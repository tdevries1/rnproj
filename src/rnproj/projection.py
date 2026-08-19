"""Core projection estimator (Layer 1).

Projects a target payoff :math:`g(S_T)` onto the span of the observed option
payoffs

.. math::

    \\{\\, 1,\\; S_T,\\; (K^P_j - S_T)^+,\\; (S_T - K^C_j)^+ \\,\\}

by weighted least squares on a state grid, then prices the projection with
the observed option prices:

.. math::

    \\hat E^Q[g] \\;=\\; \\hat\\beta_1 + \\hat\\beta_2 F_{t\\to T}
        + R_{f,t\\to T}\\Big(\\sum_j \\hat\\beta^P_j P(K_j)
        + \\sum_j \\hat\\beta^C_j C(K_j)\\Big).

Python port of ``ols_projection.m`` / ``ols_projection_sparse.m`` from the
Matlab reference implementation of De Vries (2026), with automatic state-grid
construction (see :mod:`rnproj.grids`).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._linalg import wls, wls_nonnegative

_trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy < 2 compatibility
from .chain import OptionChain
from .grids import default_grid, validate_grid

__all__ = ["project", "Projection", "BidAsk"]


@dataclass(frozen=True)
class BidAsk:
    """Bid and ask prices for conservative pricing of the replicating portfolio.

    Arrays are aligned with the chain's (sorted) put strikes followed by its
    (sorted) call strikes, after any OTM filtering. When supplied, long legs
    (positive projection weight) are priced at the ask and short legs at the
    bid, giving a conservative estimate of the replication cost.
    """

    bid: np.ndarray
    ask: np.ndarray


def _basis_matrix(s: np.ndarray, put_strikes: np.ndarray, call_strikes: np.ndarray) -> np.ndarray:
    """Design matrix [1, s, put hinges, call hinges] -- puts before calls,
    strikes ascending within each block. Must stay in lockstep with
    :func:`_basis_prices`."""
    s = np.asarray(s, dtype=float)
    return np.column_stack(
        [
            np.ones_like(s),
            s,
            np.maximum(put_strikes[None, :] - s[:, None], 0.0),
            np.maximum(s[:, None] - call_strikes[None, :], 0.0),
        ]
    )


def _basis_prices(chain: OptionChain) -> np.ndarray:
    """Risk-neutral expectations of the basis: [1, F, R_f*P..., R_f*C...].

    Same column order as :func:`_basis_matrix` (the chain stores strikes
    sorted ascending per leg).
    """
    rf = chain.gross_rate
    return np.concatenate(
        [[1.0, chain.forward], rf * chain.put_prices, rf * chain.call_prices]
    )


@dataclass(frozen=True)
class Projection:
    """Result of :func:`project`.

    Attributes
    ----------
    value : float or numpy.ndarray
        The projection estimate :math:`\\hat E^Q[g(S_T)]` (an array when
        ``g`` returned several target columns).
    beta : numpy.ndarray
        WLS coefficients, ordered [constant, underlying, puts..., calls...].
    basis_prices : numpy.ndarray
        Risk-neutral basis expectations ``[1, F, R_f*P..., R_f*C...]``.
    residual_l2 : float or numpy.ndarray
        Projection error :math:`\\|g - \\hat g\\|_{L^2(\\omega)}`, the
        data-driven half of the finite-sample bound of De Vries (2026).
    cond : float
        Condition number of the row-weighted design matrix.
    """

    value: float | np.ndarray
    beta: np.ndarray
    basis_prices: np.ndarray
    put_strikes: np.ndarray
    call_strikes: np.ndarray
    grid: np.ndarray
    weights: np.ndarray
    residual_l2: float | np.ndarray
    cond: float
    chain: OptionChain

    def fitted(self, s: np.ndarray | None = None) -> np.ndarray:
        """The projected payoff :math:`\\hat g` on ``s`` (default: the grid)."""
        X = _basis_matrix(self.grid if s is None else np.asarray(s, float),
                          self.put_strikes, self.call_strikes)
        return X @ self.beta

    def error_bound(self, chi2: float) -> float | np.ndarray:
        """Finite-sample bound :math:`\\|g-\\hat g\\|_{L^2(\\omega)}\\sqrt{\\chi^2}`.

        ``chi2`` is the chi-squared divergence :math:`\\chi^2(f^Q \\| \\omega)`
        between the (unknown) risk-neutral density and the weighting density;
        the bound holds for every risk-neutral measure within that divergence
        of :math:`\\omega`.
        """
        return self.residual_l2 * np.sqrt(chi2)

    def portfolio(self) -> Any:
        """The replicating portfolio: one row per instrument with its weight.

        Returns a :class:`pandas.DataFrame` when pandas is installed, else a
        dict of arrays.
        """
        beta = self.beta if self.beta.ndim == 1 else self.beta[:, 0]
        nP, nC = self.put_strikes.size, self.call_strikes.size
        data = {
            "instrument": np.array(
                ["bond", "underlying"] + ["put"] * nP + ["call"] * nC, dtype=object
            ),
            "strike": np.concatenate([[np.nan, np.nan], self.put_strikes, self.call_strikes]),
            "weight": beta,
        }
        try:
            import pandas as pd

            return pd.DataFrame(data)
        except ImportError:
            return data


def project(
    g: Callable[[np.ndarray], np.ndarray] | np.ndarray,
    chain: OptionChain,
    *,
    grid: np.ndarray | None = None,
    weights: Any = None,
    otm_only: bool = True,
    ridge: float = 0.0,
    nonnegative: bool = False,
    bid_ask: BidAsk | None = None,
) -> Projection:
    """Estimate :math:`E^Q[g(S_T)]` by least-squares projection.

    Parameters
    ----------
    g : callable or array
        Target payoff. A callable is evaluated on the state grid and may
        return either a vector or an ``(n_grid, m)`` matrix (one column per
        target; e.g. an indicator matrix prices a whole CDF in one solve).
        An array must be aligned with an explicitly supplied ``grid``.
    chain : OptionChain
        Observed option quotes for one maturity.
    grid : array, optional
        State grid. Default: built automatically by
        :func:`rnproj.grids.default_grid`, whose mesh is finer than half the
        smallest strike gap so the design matrix keeps full column rank.
        A supplied grid is validated (hard error if it does not cover all
        strikes; warning if its mesh is too coarse).
    weights : optional
        The weighting density :math:`\\omega`. One of: ``None`` (automatic,
        see :func:`rnproj.weights.automatic_weights`), a ``WeightSpec``, a
        density callable evaluated on the grid, or an array aligned with an
        explicitly supplied ``grid``.
    otm_only : bool
        If True (default), keep only OTM quotes, split at the forward. Set
        False to use every supplied quote as a basis function (the
        ``ols_projection_sparse`` behavior, appropriate for sparse OTC
        chains where e.g. the delta-neutral ATM strike sits above the
        forward).
    ridge : float
        Scale-invariant ridge strength (0 = plain WLS).
    nonnegative : bool
        Constrain the fitted payoff to be nonnegative in every grid state.
    bid_ask : BidAsk, optional
        Conservative pricing: long legs at ask, short legs at bid.
    """
    ch = chain.otm() if otm_only else chain
    if ch.n_options == 0:
        raise ValueError("no option quotes remain after OTM filtering.")
    strikes = np.concatenate([ch.put_strikes, ch.call_strikes])

    # ---- resolve grid and weighting density ---------------------------- #
    grid_arr, w = _resolve_grid_weights(chain, ch, grid, weights, otm_only)
    if np.any(w <= 0) or not np.all(np.isfinite(w)):
        raise ValueError("weights must be strictly positive and finite on the grid.")
    grid_arr = validate_grid(grid_arr, strikes)

    # ---- design matrix and targets -------------------------------------- #
    X = _basis_matrix(grid_arr, ch.put_strikes, ch.call_strikes)
    Y = np.asarray(g(grid_arr) if callable(g) else g, dtype=float)
    if Y.shape[0] != grid_arr.size:
        raise ValueError(
            f"target has {Y.shape[0]} rows but the grid has {grid_arr.size} points."
            + ("" if callable(g) else " Pass an explicit `grid` aligned with the array.")
        )

    # ---- weighted least squares ----------------------------------------- #
    if nonnegative:
        if ridge > 0.0:
            raise ValueError("nonnegative=True does not support ridge.")
        beta, cond = wls_nonnegative(X, Y, w)
    else:
        beta, cond = wls(X, Y, w, ridge=ridge)

    # ---- price the projection ------------------------------------------- #
    e_x = _basis_prices(ch)
    if bid_ask is not None:
        e_x = _conservative_prices(ch, beta, bid_ask)
    value = e_x @ beta

    resid = Y - X @ beta
    residual_l2 = np.sqrt(_trapezoid(w[:, None] * resid**2 if resid.ndim == 2 else w * resid**2,
                                     grid_arr, axis=0))

    return Projection(
        value=float(value) if np.ndim(value) == 0 else value,
        beta=beta,
        basis_prices=e_x,
        put_strikes=ch.put_strikes,
        call_strikes=ch.call_strikes,
        grid=grid_arr,
        weights=w,
        residual_l2=float(residual_l2) if np.ndim(residual_l2) == 0 else residual_l2,
        cond=cond,
        chain=chain,
    )


def _resolve_grid_weights(
    chain: OptionChain,
    filtered: OptionChain,
    grid: np.ndarray | None,
    weights: Any,
    otm_only: bool,
) -> tuple[np.ndarray, np.ndarray]:
    strikes = np.concatenate([filtered.put_strikes, filtered.call_strikes])

    if weights is None:
        from .weights import automatic_weights

        spec = automatic_weights(chain, otm_only=otm_only)
        weights = spec

    if hasattr(weights, "grid") and hasattr(weights, "weights"):  # WeightSpec
        spec = weights
        if grid is None:
            return np.asarray(spec.grid, float), np.asarray(spec.weights, float)
        grid = np.asarray(grid, dtype=float)
        if spec.density is not None:
            return grid, np.asarray(spec.density(grid), float)
        if grid.shape == np.shape(spec.grid) and np.allclose(grid, spec.grid):
            return grid, np.asarray(spec.weights, float)
        raise ValueError(
            "explicit `grid` differs from the WeightSpec's grid and the spec "
            "carries no density callable; drop `grid` or pass a callable."
        )

    if callable(weights):
        if grid is None:
            grid = default_grid(strikes)
        else:
            grid = np.asarray(grid, dtype=float)
        return grid, np.asarray(weights(grid), dtype=float)

    w = np.asarray(weights, dtype=float)
    if grid is None:
        raise ValueError("passing `weights` as an array requires an explicit `grid`.")
    grid = np.asarray(grid, dtype=float)
    if w.shape != grid.shape:
        raise ValueError(f"weights (len {w.size}) must match grid (len {grid.size}).")
    return grid, w


def _conservative_prices(chain: OptionChain, beta: np.ndarray, bid_ask: BidAsk) -> np.ndarray:
    if beta.ndim != 1:
        raise ValueError("bid_ask pricing supports a single target column.")
    n_opt = chain.put_strikes.size + chain.call_strikes.size
    bid = np.asarray(bid_ask.bid, dtype=float)
    ask = np.asarray(bid_ask.ask, dtype=float)
    if bid.size != n_opt or ask.size != n_opt:
        raise ValueError(
            f"bid/ask must have one entry per basis option ({n_opt}: sorted puts "
            "then sorted calls, after any OTM filtering)."
        )
    rf = chain.gross_rate
    opt = np.where(beta[2:] >= 0, rf * ask, rf * bid)
    return np.concatenate([[1.0, chain.forward], opt])
