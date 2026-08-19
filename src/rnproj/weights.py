"""Weighting densities (the projection prior :math:`\\omega`).

The weighting density determines the norm in which the target payoff is
projected onto the option basis. The finite-sample bound of De Vries (2026)
is :math:`\\|g-\\hat g\\|_{L^2(\\omega)} \\sqrt{\\chi^2(f^Q\\|\\omega)}`, so a
good :math:`\\omega` is one close to the risk-neutral density with tails at
least as heavy.

This module provides the automatic default used across the package. In the
current version the automatic density is a lognormal matched to the
option-implied variance; the Variance-Gamma density calibrated to the smile
(the paper's recommended default) is planned as the next iteration and will
keep the lognormal as a degeneracy fallback.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .chain import OptionChain
from .grids import default_grid

__all__ = ["WeightSpec", "automatic_weights", "lognormal_weights", "weights_from_density"]


@dataclass(frozen=True)
class WeightSpec:
    """A weighting density evaluated on a state grid.

    Attributes
    ----------
    grid : numpy.ndarray
        The state grid the density was evaluated on.
    weights : numpy.ndarray
        Density values on ``grid`` (strictly positive; normalization is
        irrelevant to the projection coefficients).
    density : callable, optional
        The density function itself, when available; lets the estimator
        re-evaluate :math:`\\omega` on a different grid.
    distribution : object, optional
        A ``scipy.stats``-like frozen distribution (used for quantile-based
        grid endpoints).
    params : dict
        Description of how the density was built (family, parameters,
        fallback flags).
    """

    grid: np.ndarray
    weights: np.ndarray
    density: Callable[[np.ndarray], np.ndarray] | None = None
    distribution: Any = None
    params: dict = field(default_factory=dict)


def lognormal_weights(
    chain: OptionChain,
    sigma: float | None = None,
    *,
    otm_only: bool = True,
    alpha: float = 1e-3,
) -> WeightSpec:
    """Lognormal weighting density with mean equal to the forward.

    ``S_T`` is lognormal with :math:`E[S_T] = F` and volatility ``sigma``
    (annualized). When ``sigma`` is omitted it is implied from the option
    prices themselves: a preliminary projection with uniform weights
    estimates :math:`\\mathrm{Var}^Q(S_T)`, and
    :math:`\\sigma_{eq} = \\sqrt{\\log(1 + \\mathrm{Var}/F^2)/T}` (the
    equivalent-lognormal volatility used by the Matlab reference).
    """
    from scipy.stats import lognorm

    F, T = chain.forward, chain.maturity
    if sigma is None:
        var = _implied_variance_uniform(chain, otm_only=otm_only)
        sigma = float(np.sqrt(np.log1p(var / F**2) / T))
        source = "implied"
    else:
        source = "user"
    if not (np.isfinite(sigma) and sigma > 0):
        raise ValueError(f"sigma must be positive and finite, got {sigma!r}.")

    s_param = sigma * np.sqrt(T)
    dist = lognorm(s=s_param, scale=F * np.exp(-0.5 * s_param**2))  # E[S_T] = F
    strikes = _basis_strikes(chain, otm_only)
    grid = default_grid(strikes, distribution=dist, alpha=alpha)
    return WeightSpec(
        grid=grid,
        weights=dist.pdf(grid),
        density=dist.pdf,
        distribution=dist,
        params={"family": "lognormal", "sigma": sigma, "sigma_source": source},
    )


def automatic_weights(chain: OptionChain, *, otm_only: bool = True) -> WeightSpec:
    """The package-default weighting density for a chain.

    Currently the variance-matched lognormal of :func:`lognormal_weights`;
    will become the smile-calibrated Variance-Gamma density (with lognormal
    fallback) in a future version.
    """
    return lognormal_weights(chain, otm_only=otm_only)


def weights_from_density(
    density: Callable[[np.ndarray], np.ndarray],
    chain_or_strikes: Any,
    *,
    distribution: Any = None,
    grid: np.ndarray | None = None,
) -> WeightSpec:
    """Wrap a user-supplied density function as a :class:`WeightSpec`.

    ``chain_or_strikes`` (an :class:`OptionChain` or a strike array) supplies
    the strikes for automatic grid construction when ``grid`` is omitted.
    """
    if grid is None:
        strikes = (
            chain_or_strikes.strikes
            if isinstance(chain_or_strikes, OptionChain)
            else np.asarray(chain_or_strikes, dtype=float)
        )
        grid = default_grid(strikes, distribution=distribution)
    else:
        grid = np.asarray(grid, dtype=float)
    return WeightSpec(
        grid=grid,
        weights=np.asarray(density(grid), dtype=float),
        density=density,
        distribution=distribution,
        params={"family": "user"},
    )


# ---------------------------------------------------------------------- #
def _basis_strikes(chain: OptionChain, otm_only: bool) -> np.ndarray:
    ch = chain.otm() if otm_only else chain
    if ch.n_options == 0:
        raise ValueError("no option quotes remain after OTM filtering.")
    return np.concatenate([ch.put_strikes, ch.call_strikes])


def _implied_variance_uniform(chain: OptionChain, *, otm_only: bool) -> float:
    """Option-implied Var^Q(S_T) from a uniform-weight seed projection."""
    from .projection import project

    F = chain.forward
    strikes = _basis_strikes(chain, otm_only)
    grid = default_grid(strikes)
    fit = project(
        lambda s: (s - F) ** 2,
        chain,
        grid=grid,
        weights=np.ones_like(grid),
        otm_only=otm_only,
    )
    # Floor at a small positive variance so a degenerate seed never produces
    # a zero-width weighting density.
    return max(float(fit.value), (1e-4 * F) ** 2)
