"""Option-implied distribution: CDF and PDF by projection.

The CDF at ``x`` is the projection estimate of the indicator payoff
:math:`1\\{S_T \\le x\\}` (a whole CDF prices in one least-squares solve by
stacking indicator columns). By the pricing-consistency property of the
projection, integrating a payoff against this distribution reproduces the
projection estimate of its price.

The PDF has the closed form

.. math::

    \\hat f^Q(x) = E_X' M^{-1} \\varphi(x)\\, \\omega(x), \\qquad
    M = \\int \\varphi \\varphi' \\omega\\, ds,

with :math:`\\varphi` the option basis and :math:`E_X` its observed prices
-- the Matlab reference's ``ols_weighted_pricing_pdf.m``, generalized here
to non-uniform grids via trapezoid quadrature weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .chain import OptionChain
from .projection import Projection, _basis_matrix, _basis_prices, _resolve_grid_weights, project

__all__ = ["DistributionFit", "implied_cdf", "implied_pdf"]


@dataclass(frozen=True)
class DistributionFit:
    """An estimated risk-neutral CDF or PDF on evaluation points ``x``.

    ``raw`` holds the pre-monotonization (CDF) values; ``values`` the final
    estimate. ``fit`` is the underlying :class:`Projection` when one was
    solved (CDF), else ``None``.
    """

    kind: str
    x: np.ndarray
    values: np.ndarray
    raw: np.ndarray
    fit: Projection | None

    def __call__(self, q: np.ndarray) -> np.ndarray:
        """Linear interpolation of the estimate at points ``q``."""
        lo, hi = (0.0, 1.0) if self.kind == "cdf" else (0.0, 0.0)
        return np.interp(np.asarray(q, dtype=float), self.x, self.values, left=lo, right=hi)


def implied_cdf(
    chain: OptionChain,
    x: np.ndarray | None = None,
    *,
    monotone: str | None = "rearrange",
    n_points: int = 200,
    **kwargs: Any,
) -> DistributionFit:
    """Estimate the risk-neutral CDF of :math:`S_T`.

    Parameters
    ----------
    x : array, optional
        Evaluation points. Default: ``n_points`` evenly spaced points across
        the automatic state grid.
    monotone : {"rearrange", "constrained", None}
        ``"rearrange"`` (default): sort the raw estimates (Chernozhukov,
        Fernandez-Val & Galichon rearrangement) and clip to [0, 1] --
        dependency-free and the recommended choice.
        ``"constrained"``: re-solve each point sequentially with the
        constraint ``cdf(x_{i-1}) <= cdf(x_i) <= 1`` imposed on the
        estimate (port of ``ols_pricing_cdf_discrete.m``); slower.
        ``None``: raw projection estimates.
    **kwargs
        Passed to :func:`rnproj.project`.
    """
    if monotone not in ("rearrange", "constrained", None):
        raise ValueError(f"monotone must be 'rearrange', 'constrained', or None, got {monotone!r}.")

    fit = project(
        lambda s, _x=x: _indicator_targets(s, _x, n_points), chain, **kwargs
    )
    x_eval = _eval_points(fit.grid, x, n_points)
    raw = np.asarray(fit.value, dtype=float)

    if monotone == "rearrange":
        values = np.clip(np.sort(raw), 0.0, 1.0)
    elif monotone == "constrained":
        values = _constrained_cdf(fit, x_eval)
    else:
        values = raw
    return DistributionFit(kind="cdf", x=x_eval, values=values, raw=raw, fit=fit)


def implied_pdf(
    chain: OptionChain,
    x: np.ndarray | None = None,
    *,
    n_points: int = 200,
    grid: np.ndarray | None = None,
    weights: Any = None,
    otm_only: bool = True,
) -> DistributionFit:
    """Closed-form risk-neutral density :math:`\\hat f^Q` of :math:`S_T`.

    Evaluates :math:`E_X' M^{-1} \\varphi(x) \\omega(x)` with the weighted
    Gram matrix :math:`M` computed by trapezoid quadrature on the state
    grid. Requires the weighting density to be evaluable at ``x``; the
    automatic (VG/lognormal) weights always are.
    """
    ch = chain.otm() if otm_only else chain
    if ch.n_options == 0:
        raise ValueError("no option quotes remain after OTM filtering.")
    grid_arr, w = _resolve_grid_weights(chain, ch, grid, weights, otm_only)
    x_eval = _eval_points(grid_arr, x, n_points)

    # omega at the evaluation points: interpolate the grid values (exact when
    # x_eval lies on grid nodes; the automatic grids are dense).
    w_x = np.interp(x_eval, grid_arr, w)

    X = _basis_matrix(grid_arr, ch.put_strikes, ch.call_strikes)
    dx = np.gradient(grid_arr)  # trapezoid weights, valid on non-uniform grids
    M = X.T @ (X * (w * dx)[:, None])
    e_x = _basis_prices(ch)
    a = np.linalg.solve(M, e_x)
    phi = _basis_matrix(x_eval, ch.put_strikes, ch.call_strikes)
    values = (phi @ a) * w_x
    return DistributionFit(kind="pdf", x=x_eval, values=values, raw=values, fit=None)


# ---------------------------------------------------------------------- #
def _eval_points(grid: np.ndarray, x: np.ndarray | None, n_points: int) -> np.ndarray:
    if x is None:
        return np.linspace(grid[0], grid[-1], n_points)
    return np.asarray(x, dtype=float)


def _indicator_targets(s: np.ndarray, x: np.ndarray | None, n_points: int) -> np.ndarray:
    x_eval = _eval_points(s, x, n_points)
    return (s[:, None] <= x_eval[None, :]).astype(float)


def _constrained_cdf(fit: Projection, x_eval: np.ndarray) -> np.ndarray:
    """Sequential constrained re-solve: cdf(x_{i-1}) <= E_X beta <= 1.

    Port of ``ols_pricing_cdf_discrete.m`` using a projected-gradient-free
    QP per point (scipy trust-constr), warm-started at the unconstrained
    solution and skipped when it already satisfies the constraints.
    """
    from scipy.optimize import LinearConstraint, minimize

    X = _basis_matrix(fit.grid, fit.put_strikes, fit.call_strikes)
    sw = np.sqrt(fit.weights)[:, None]
    Xw = X * sw
    G = Xw.T @ Xw
    e_x = fit.basis_prices

    raw = np.asarray(fit.value, dtype=float)
    beta_raw = fit.beta  # (p, m)
    values = np.empty_like(raw)
    prev = 0.0
    for i in range(raw.size):
        if prev <= raw[i] <= 1.0:
            values[i] = raw[i]
            prev = raw[i]
            continue
        y = (fit.grid <= x_eval[i]).astype(float)
        c = Xw.T @ (y * sw[:, 0])
        res = minimize(
            lambda b: 0.5 * b @ G @ b - c @ b,
            x0=beta_raw[:, i],
            jac=lambda b: G @ b - c,
            method="trust-constr",
            constraints=[LinearConstraint(e_x[None, :], prev, 1.0)],
            options={"gtol": 1e-12, "xtol": 1e-14, "maxiter": 500},
        )
        values[i] = float(np.clip(e_x @ res.x, prev, 1.0))
        prev = values[i]
    return values
