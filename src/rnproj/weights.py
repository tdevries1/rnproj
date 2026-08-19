"""Weighting densities (the projection prior :math:`\\omega`).

The weighting density determines the norm in which the target payoff is
projected onto the option basis. The finite-sample bound of De Vries (2026)
is :math:`\\|g-\\hat g\\|_{L^2(\\omega)} \\sqrt{\\chi^2(f^Q\\|\\omega)}`, so a
good :math:`\\omega` is one close to the risk-neutral density with tails at
least as heavy.

This module provides the automatic default used across the package: a
Variance-Gamma density calibrated to the observed smile (the paper's
recommended choice; its semi-heavy polynomial-like tails keep the
chi-squared divergence finite against realistic risk-neutral densities),
with a variance-matched lognormal fallback whenever the VG calibration
degenerates.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from ._vg import VGQuantiles, calibrate_vg, vg_price_density
from .chain import OptionChain
from .grids import default_grid

__all__ = [
    "WeightSpec",
    "automatic_weights",
    "vg_weights",
    "lognormal_weights",
    "weights_from_density",
]


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


def vg_weights(
    chain: OptionChain,
    *,
    otm_only: bool = True,
    alpha: float = 1e-3,
) -> WeightSpec:
    """Variance-Gamma weighting density calibrated to the observed smile.

    Port of ``vg_estimate_from_options.m``: OTM quotes (puts converted to
    synthetic calls by put-call parity) are fitted by pricing MSE over the
    VG parameters ``(sigma, nu, theta)`` with the martingale compensator
    enforcing :math:`E^Q[S_T] = F`. Falls back to the variance-matched
    lognormal of :func:`lognormal_weights` when

    - fewer than 4 OTM quotes are available,
    - the calibration fails or pins ``sigma``/``nu`` at its lower bound, or
    - ``T / nu < 1/2``, where the VG density develops an integrable
      singularity at the mode (a near-delta weight that breaks WLS in
      low-volatility regimes).

    The returned spec's ``params['fallback']`` records which branch was used.
    """
    F, T, r = chain.forward, chain.maturity, chain.rate

    otm_c = chain.call_strikes > F
    otm_p = chain.put_strikes < F
    k_all = np.concatenate([chain.put_strikes[otm_p], chain.call_strikes[otm_c]])
    c_all = np.concatenate(
        [
            chain.put_prices[otm_p] + (F - chain.put_strikes[otm_p]) * chain.discount,
            chain.call_prices[otm_c],
        ]
    )
    order = np.argsort(k_all, kind="stable")
    k_all, c_all = k_all[order], c_all[order]

    sigma_eq = _sigma_eq(chain, otm_only=otm_only)

    if k_all.size < 4:
        return _lognormal_fallback(chain, sigma_eq, otm_only, "too few OTM quotes")

    params, cost = calibrate_vg(k_all, c_all, r, T, F, sigma_eq)
    if params is None:
        return _lognormal_fallback(chain, sigma_eq, otm_only, "calibration failed")
    if (
        params.sigma <= 1e-3 * 1.01
        or params.nu <= 0.001 * 1.01
        or T / params.nu < 0.5
    ):
        return _lognormal_fallback(chain, sigma_eq, otm_only, "degenerate VG parameters")

    quantiles = VGQuantiles(params, T, F)
    grid = default_grid(chain.strikes, distribution=quantiles, alpha=alpha)
    return WeightSpec(
        grid=grid,
        weights=vg_price_density(params, T, F, grid),
        density=lambda s, _p=params: vg_price_density(_p, T, F, np.asarray(s, float)),
        distribution=quantiles,
        params={
            "family": "vg",
            "sigma": params.sigma,
            "nu": params.nu,
            "theta": params.theta,
            "omega": params.omega,
            "pricing_mse": cost / k_all.size,
            "fallback": False,
        },
    )


def automatic_weights(chain: OptionChain, *, otm_only: bool = True) -> WeightSpec:
    """The package-default weighting density for a chain.

    The smile-calibrated Variance-Gamma density of :func:`vg_weights`, with
    its built-in lognormal fallback on degeneracy.
    """
    return vg_weights(chain, otm_only=otm_only)


def _sigma_eq(chain: OptionChain, *, otm_only: bool) -> float:
    """BS-equivalent volatility implied by a uniform-weight seed projection."""
    F, T = chain.forward, chain.maturity
    try:
        var = _implied_variance_uniform(chain, otm_only=otm_only)
        sigma_eq = float(np.sqrt(np.log1p(var / F**2) / T))
    except (ValueError, np.linalg.LinAlgError):
        return 0.20
    if not (np.isfinite(sigma_eq) and sigma_eq > 0):
        return 0.20
    return sigma_eq


def _lognormal_fallback(
    chain: OptionChain, sigma_eq: float, otm_only: bool, reason: str
) -> WeightSpec:
    spec = lognormal_weights(chain, sigma=sigma_eq, otm_only=otm_only)
    return replace(
        spec, params={**spec.params, "fallback": True, "fallback_reason": reason}
    )


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
