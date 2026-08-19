"""Variance-Gamma density, pricing, and calibration (internal).

Port of ``vg_pdf.m`` / ``vg_price_density.m`` from the Matlab reference.
The VG process (Madan, Carr & Chang 1998) is parameterized by
``(sigma, nu, theta)``:

.. math::

    X_T = \\theta G_T + \\sigma W(G_T), \\qquad G_T \\sim
    \\Gamma(T/\\nu, \\nu),

and the price is :math:`S_T = F e^{\\omega T + X_T}` with the martingale
compensator :math:`\\omega = \\nu^{-1}\\log(1 - \\theta\\nu -
\\sigma^2\\nu/2)`, so :math:`E^Q[S_T] = F` holds automatically.

The density is evaluated by conditioning on the Gamma subordinator and
integrating the normal mixture on a fixed 800-point grid (identical to the
Matlab quadrature, so cross-language golden tests can compare tightly);
calls are priced by Carr-Madan FFT inversion of the characteristic function.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln

_N_GAMMA_GRID = 800


@dataclass(frozen=True)
class VGParams:
    sigma: float
    nu: float
    theta: float

    @property
    def omega(self) -> float:
        return float(np.log(1.0 - self.theta * self.nu - 0.5 * self.sigma**2 * self.nu) / self.nu)

    @property
    def feasible(self) -> bool:
        return 1.0 - self.theta * self.nu - 0.5 * self.sigma**2 * self.nu > 0.0


def _gamma_grid(T: float, nu: float, n_g: int = _N_GAMMA_GRID) -> tuple[np.ndarray, np.ndarray]:
    """Trapezoidal grid and density for the Gamma(T/nu, nu) subordinator.

    Same asymmetric [-8 sd, +15 sd] coverage and log-space evaluation as the
    Matlab reference (gammaln avoids overflow for large shape).
    """
    shape = T / nu
    scale = nu
    mean_g = shape * scale
    std_g = np.sqrt(shape) * scale
    g_min = max(1e-6, mean_g - 8.0 * std_g)
    g_max = mean_g + 15.0 * std_g
    g = np.linspace(g_min, g_max, n_g)
    log_fg = -shape * np.log(scale) - gammaln(shape) + (shape - 1.0) * np.log(g) - g / scale
    fg = np.exp(log_fg)
    fg[~np.isfinite(fg)] = 0.0
    return g, fg


def vg_log_density(
    x: np.ndarray, params: VGParams, T: float, *, mixture: str = "quantile"
) -> np.ndarray:
    """Density of the VG variate ``X_T`` via Gamma-subordinator mixing.

    ``mixture="quantile"`` (default) integrates the normal mixture with an
    equal-probability midpoint rule on Gamma quantiles, which stays accurate
    when the subordinator density is singular at zero (``T/nu < 1``).
    ``mixture="trapz"`` reproduces the Matlab reference's linear-grid
    trapezoid quadrature bit-for-bit; note that for ``T/nu < 1`` that
    quadrature overweights the singularity and the resulting density does
    not integrate to one (spurious mass appears at the mode).
    """
    x = np.asarray(x, dtype=float)
    if mixture == "trapz":
        g, fg = _gamma_grid(T, params.nu)
        gc = g[:, None]
        cond = np.exp(
            -0.5 * (x[None, :] - params.theta * gc) ** 2 / (params.sigma**2 * gc)
        ) / np.sqrt(2.0 * np.pi * params.sigma**2 * gc)
        trap = getattr(np, "trapezoid", getattr(np, "trapz", None))
        return trap(cond * fg[:, None], g, axis=0)
    if mixture != "quantile":
        raise ValueError(f"mixture must be 'quantile' or 'trapz', got {mixture!r}.")

    from scipy.stats import gamma as gamma_dist

    u = (np.arange(_N_GAMMA_GRID) + 0.5) / _N_GAMMA_GRID
    g = gamma_dist.ppf(u, a=T / params.nu, scale=params.nu)[:, None]
    cond = np.exp(-0.5 * (x[None, :] - params.theta * g) ** 2 / (params.sigma**2 * g)) / np.sqrt(
        2.0 * np.pi * params.sigma**2 * g
    )
    return cond.mean(axis=0)


def vg_price_density(
    params: VGParams, T: float, forward: float, s: np.ndarray, *, mixture: str = "quantile"
) -> np.ndarray:
    """VG density of the price :math:`S_T` on the grid ``s``."""
    if not params.feasible:
        raise ValueError(
            "VG parameters violate 1 - theta*nu - 0.5*sigma^2*nu > 0 (omega infinite)."
        )
    s = np.asarray(s, dtype=float)
    x_log = np.log(s / forward) - params.omega * T
    return vg_log_density(x_log, params, T, mixture=mixture) / s


def vg_call_prices(
    params: VGParams,
    strikes: np.ndarray,
    rate: float,
    T: float,
    forward: float,
    *,
    n_fft: int = 4096,
    eta: float = 0.25,
    damping: float = 1.5,
) -> np.ndarray:
    """Discounted VG call prices via Carr-Madan (1999) FFT inversion.

    Returns a large penalty vector when the parameters leave the feasible
    region (used by the calibration objective, mirroring the Matlab code).
    """
    from scipy.interpolate import PchipInterpolator

    sigma, nu, theta = params.sigma, params.nu, params.theta
    strikes = np.asarray(strikes, dtype=float)
    a = damping

    if 1.0 - theta * nu - 0.5 * sigma**2 * nu <= 0.0:
        return np.full(strikes.size, 1e10)
    # The dampened transform needs E[S^(a+1)] < inf.
    if 1.0 - (a + 1.0) * theta * nu - 0.5 * (a + 1.0) ** 2 * sigma**2 * nu <= 0.0:
        return np.full(strikes.size, 1e10)

    omega = params.omega
    lam = 2.0 * np.pi / (n_fft * eta)
    b = n_fft * lam / 2.0
    k_grid = -b + lam * np.arange(n_fft)
    v_grid = eta * np.arange(n_fft)

    u = v_grid - (a + 1.0) * 1j
    mean_log_s = np.log(forward) + omega * T
    phi = np.exp(1j * u * mean_log_s) * (
        1.0 - 1j * theta * nu * u + 0.5 * sigma**2 * nu * u**2
    ) ** (-T / nu)
    psi = np.exp(-rate * T) * phi / (a**2 + a - v_grid**2 + 1j * (2.0 * a + 1.0) * v_grid)

    # Simpson 1/3 weights: (eta/3) * [1, 4, 2, 4, ..., 4]
    sw = np.ones(n_fft)
    sw[1::2] = 4.0
    sw[2::2] = 2.0
    sw *= eta / 3.0

    c_grid = np.real(np.exp(-a * k_grid) * np.fft.fft(np.exp(1j * b * v_grid) * psi * sw)) / np.pi
    prices = PchipInterpolator(np.exp(k_grid), c_grid, extrapolate=True)(strikes)
    return np.maximum(prices, 0.0)


def calibrate_vg(
    strikes: np.ndarray,
    call_prices: np.ndarray,
    rate: float,
    T: float,
    forward: float,
    sigma_seed: float,
) -> tuple[VGParams | None, float]:
    """Calibrate (sigma, nu, theta) to observed call prices by pricing MSE.

    SLSQP from two theta starts (some smiles are right-skewed and a single
    negative start lands in the wrong basin), bounds and the martingale
    feasibility constraint as in ``vg_pdf.m``. Returns ``(params, cost)``,
    with ``params = None`` on total failure.
    """
    from scipy.optimize import minimize

    strikes = np.asarray(strikes, dtype=float)
    call_prices = np.asarray(call_prices, dtype=float)

    sigma0 = max(sigma_seed, 0.1)
    lb = np.array([1e-3, 0.001, -2.0])
    ub = np.array([3.0, 5.0, 2.0])

    def objective(x: np.ndarray) -> float:
        model = vg_call_prices(VGParams(*x), strikes, rate, T, forward)
        return float(np.sum((model - call_prices) ** 2))

    def feasibility(x: np.ndarray) -> float:
        # >= 0 required: 1 - theta*nu - 0.5*sigma^2*nu - 1e-4
        return 1.0 - x[2] * x[1] - 0.5 * x[0] ** 2 * x[1] - 1e-4

    best_x, best_cost = None, np.inf
    for theta0 in (-0.1, 0.1):
        x0 = np.array([sigma0, 0.3, theta0])
        try:
            res = minimize(
                objective,
                x0,
                method="SLSQP",
                bounds=list(zip(lb, ub)),
                constraints=[{"type": "ineq", "fun": feasibility}],
                options={"maxiter": 500, "ftol": 1e-12},
            )
        except (ValueError, RuntimeError):
            continue
        if np.isfinite(res.fun) and res.fun < best_cost and feasibility(res.x) >= 0:
            best_cost, best_x = float(res.fun), res.x
    if best_x is None:
        return None, np.inf
    return VGParams(*(float(v) for v in best_x)), best_cost


class VGQuantiles:
    """Quantile function of the VG price distribution, built by numerically
    inverting the CDF on a fine grid (matches steps 4 of the Matlab
    reference: 5000-point fine grid, cumulative trapezoid, PCHIP inverse)."""

    def __init__(self, params: VGParams, T: float, forward: float, n_fine: int = 5000):
        from scipy.interpolate import PchipInterpolator

        log_std = np.sqrt(params.sigma**2 * T + params.theta**2 * params.nu * T)
        log_center = params.omega * T
        log_range = 10.0 * log_std + abs(log_center)
        x_min = max(1e-6, forward * np.exp(log_center - log_range))
        x_max = forward * np.exp(log_center + log_range)
        x = np.linspace(x_min, x_max, n_fine)
        f = vg_price_density(params, T, forward, x)
        cdf = np.concatenate([[0.0], np.cumsum((f[1:] + f[:-1]) * 0.5 * np.diff(x))])
        cdf /= cdf[-1]
        # far tails underflow to zero, leaving flat plateaus; strip them
        cdf_u, idx = np.unique(cdf, return_index=True)
        self._inv = PchipInterpolator(cdf_u, x[idx], extrapolate=False)

    def ppf(self, q: float) -> float:
        return float(self._inv(q))
