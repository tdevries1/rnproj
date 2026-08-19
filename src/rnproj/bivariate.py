"""Bivariate FX projection: joint risk-neutral moments and tail risk.

The paper's multivariate application (Section 4.3 of De Vries 2026): for a
triangle of exchange rates :math:`S_1` (e.g. EURUSD), :math:`S_2` (GBPUSD)
and the cross rate :math:`S_1/S_2` (EURGBP), vanilla options on all three
carry joint information because of triangular parity. The joint basis on a
2-D tensor state grid is

.. math::

    \\{1,\\; s_1,\\; (K^P-s_1)^+,\\; (s_1-K^C)^+,\\; s_2,\\; (K^P-s_2)^+,
    \\; (s_2-K^C)^+,\\; s_2 (K^P - s_1/s_2)^+,\\; s_2 (s_1/s_2 - K^C)^+\\},

where the cross-rate legs are the *numeraire-adjusted* payoffs: by change
of numeraire, :math:`E^Q[S_{2,T} (S_{1,T}/S_{2,T} - K)^+]` equals the
leg-2 forward times the cross rate's gross rate times the observed
cross-rate option price. Any joint target :math:`g(s_1, s_2)` is projected
onto this basis and priced with the stacked observed prices.

Port of the FX section of ``cm_simulation8.m`` / ``empirical_application.m``
and of ``hoeffdingCovCells.m``. Following the reference, marginal
quantities use uniform weights and no OTM filter (sparse five-strike
chains), and grids are ``linspace(0.95*Kmin, 1.02*Kmax, n)`` per leg with
the package's resolution rule applied on top.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from .chain import OptionChain
from .projection import project

__all__ = [
    "FXTriangle",
    "JointProjection",
    "joint_projection",
    "implied_covariance",
    "joint_tail_probability",
    "hoeffding_decomposition",
    "CovarianceResult",
    "TailResult",
    "HoeffdingResult",
]


@dataclass(frozen=True)
class FXTriangle:
    """A triangle of option chains: two legs against a common numeraire
    currency plus the cross rate.

    ``leg1`` and ``leg2`` are quoted in the numeraire currency (e.g.
    EURUSD and GBPUSD, both in USD); ``cross`` is the ``S1/S2`` rate quoted
    in leg-2 currency (e.g. EURGBP in GBP), with its own discount/rate. The
    numeraire adjustment for the cross-rate legs is handled internally.
    """

    leg1: OptionChain
    leg2: OptionChain
    cross: OptionChain

    def __post_init__(self) -> None:
        t = (self.leg1.maturity, self.leg2.maturity, self.cross.maturity)
        if max(t) - min(t) > 1e-8:
            raise ValueError(f"the three chains must share one maturity, got {t}.")


@dataclass(frozen=True)
class JointProjection:
    """Result of :func:`joint_projection`."""

    value: float | np.ndarray
    beta: np.ndarray
    basis_prices: np.ndarray
    grid1: np.ndarray
    grid2: np.ndarray
    triangle: FXTriangle


def _leg_grid(chain: OptionChain, n_grid: int) -> np.ndarray:
    """Reference grid rule: [0.95 Kmin, 1.02 Kmax], resolution-rule floor."""
    k = chain.strikes
    lo, hi = 0.95 * k.min(), 1.02 * k.max()
    n = n_grid
    if k.size > 1:
        min_gap = float(np.min(np.diff(k)))
        n = max(n, int(np.ceil((hi - lo) / (min_gap / 2.0))) + 1)
    return np.linspace(lo, hi, n)


def _joint_system(
    tri: FXTriangle,
    grids: tuple[np.ndarray, np.ndarray] | None,
    n_grid: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Tensor grid, basis matrix, and stacked prices for the triangle."""
    l1, l2, cx = tri.leg1, tri.leg2, tri.cross
    if grids is None:
        g1, g2 = _leg_grid(l1, n_grid), _leg_grid(l2, n_grid)
    else:
        g1, g2 = (np.asarray(g, dtype=float) for g in grids)

    # cross strikes must lie inside the cross-rate range spanned by the grids
    lo, hi = g1[0] / g2[-1], g1[-1] / g2[0]
    if cx.strikes.min() < lo or cx.strikes.max() > hi:
        raise ValueError(
            f"cross-rate strikes [{cx.strikes.min():.6g}, {cx.strikes.max():.6g}] fall "
            f"outside the cross-rate range [{lo:.6g}, {hi:.6g}] of the joint grid."
        )

    x1, x2 = (a.ravel() for a in np.meshgrid(g1, g2))
    ratio = x1 / x2
    phi = np.column_stack(
        [
            np.ones_like(x1),
            x1,
            np.maximum(l1.put_strikes[None, :] - x1[:, None], 0.0),
            np.maximum(x1[:, None] - l1.call_strikes[None, :], 0.0),
            x2,
            np.maximum(l2.put_strikes[None, :] - x2[:, None], 0.0),
            np.maximum(x2[:, None] - l2.call_strikes[None, :], 0.0),
            x2[:, None] * np.maximum(cx.put_strikes[None, :] - ratio[:, None], 0.0),
            x2[:, None] * np.maximum(ratio[:, None] - cx.call_strikes[None, :], 0.0),
        ]
    )
    cross_factor = l2.forward * cx.gross_rate  # numeraire adjustment
    prices = np.concatenate(
        [
            [1.0, l1.forward],
            l1.gross_rate * l1.put_prices,
            l1.gross_rate * l1.call_prices,
            [l2.forward],
            l2.gross_rate * l2.put_prices,
            l2.gross_rate * l2.call_prices,
            cross_factor * cx.put_prices,
            cross_factor * cx.call_prices,
        ]
    )
    return g1, g2, x1, phi, prices


def joint_projection(
    g: Callable[[np.ndarray, np.ndarray], np.ndarray],
    tri: FXTriangle,
    *,
    grids: tuple[np.ndarray, np.ndarray] | None = None,
    weights: Any = None,
    n_grid: int = 700,
) -> JointProjection:
    """Project a joint payoff ``g(s1, s2)`` onto the triangle's option basis.

    Parameters
    ----------
    g : callable
        Joint target payoff, evaluated elementwise on the flattened tensor
        grid; may return an ``(n, m)`` matrix for several targets at once.
    grids : (array, array), optional
        Marginal state grids; default per the reference rule
        ``linspace(0.95 Kmin, 1.02 Kmax, n_grid)`` per leg (with the
        resolution rule as a floor).
    weights : optional
        ``None`` (uniform, the reference default), a pair of marginal
        weight arrays (product density), or a callable ``w(s1, s2)``.
    """
    g1, g2, x1, phi, prices = _joint_system(tri, grids, n_grid)
    x2 = np.meshgrid(g1, g2)[1].ravel()

    if weights is None:
        w = None
    elif callable(weights):
        w = np.asarray(weights(x1, x2), dtype=float)
    else:
        w1, w2 = (np.asarray(a, dtype=float) for a in weights)
        if w1.shape != g1.shape or w2.shape != g2.shape:
            raise ValueError("marginal weight arrays must match the marginal grids.")
        w = (w2[:, None] * w1[None, :]).ravel()

    y = np.asarray(g(x1, x2), dtype=float)
    if w is None:
        beta, *_ = np.linalg.lstsq(phi, y, rcond=None)
    else:
        if np.any(w <= 0):
            raise ValueError("weights must be strictly positive on the grid.")
        sw = np.sqrt(w)[:, None]
        beta, *_ = np.linalg.lstsq(phi * sw, y * (sw if y.ndim == 2 else sw[:, 0]), rcond=None)

    value = prices @ beta
    return JointProjection(
        value=float(value) if np.ndim(value) == 0 else value,
        beta=beta,
        basis_prices=prices,
        grid1=g1,
        grid2=g2,
        triangle=tri,
    )


# ---------------------------------------------------------------------- #
@dataclass(frozen=True)
class CovarianceResult:
    covariance: float
    correlation: float
    variance1: float
    variance2: float
    fit: JointProjection


def implied_covariance(tri: FXTriangle, **kwargs: Any) -> CovarianceResult:
    """Risk-neutral covariance and correlation of ``(S1, S2)``.

    Projects :math:`(s_1 - F_1)(s_2 - F_2)` onto the joint basis; marginal
    variances come from univariate projections on each leg (uniform
    weights, no OTM filter, per the reference implementation).
    """
    F1, F2 = tri.leg1.forward, tri.leg2.forward
    fit = joint_projection(lambda s1, s2: (s1 - F1) * (s2 - F2), tri, **kwargs)
    var1 = _marginal_variance(tri.leg1, fit.grid1)
    var2 = _marginal_variance(tri.leg2, fit.grid2)
    cov = float(fit.value)
    return CovarianceResult(
        covariance=cov,
        correlation=cov / np.sqrt(var1 * var2),
        variance1=var1,
        variance2=var2,
        fit=fit,
    )


@dataclass(frozen=True)
class TailResult:
    joint: float
    marginal1: float
    marginal2: float
    independent: float
    conditional: float
    fit: JointProjection


def joint_tail_probability(
    tri: FXTriangle, a1: float = 0.97, a2: float = 0.97, **kwargs: Any
) -> TailResult:
    """Joint crash probability :math:`Q(S_1/F_1 \\le a_1, S_2/F_2 \\le a_2)`.

    ``independent`` is the product of the marginal probabilities (the
    no-dependence benchmark); the gap between ``joint`` and ``independent``
    is the dependence channel. ``conditional`` is joint / marginal2.
    """
    F1, F2 = tri.leg1.forward, tri.leg2.forward
    fit = joint_projection(
        lambda s1, s2: ((s1 / F1 <= a1) & (s2 / F2 <= a2)).astype(float), tri, **kwargs
    )
    m1 = _marginal_tail(tri.leg1, fit.grid1, a1)
    m2 = _marginal_tail(tri.leg2, fit.grid2, a2)
    joint = float(fit.value)
    return TailResult(
        joint=joint,
        marginal1=m1,
        marginal2=m2,
        independent=m1 * m2,
        conditional=joint / m2 if m2 > 0 else np.nan,
        fit=fit,
    )


@dataclass(frozen=True)
class HoeffdingResult:
    """Cellwise Hoeffding decomposition of the risk-neutral covariance.

    ``cells[i, j]`` is the contribution of return-space cell
    ``[x_edges[j], x_edges[j+1]] x [y_edges[i], y_edges[i+1]]`` to
    :math:`\\iint (F_{12} - F_1 F_2)\\,dx\\,dy` (in return units); ``share``
    is each cell's fraction of ``total``.
    """

    x_edges: np.ndarray
    y_edges: np.ndarray
    cells: np.ndarray
    joint_cells: np.ndarray
    marginal_cells: np.ndarray
    total: float
    share: np.ndarray


def hoeffding_decomposition(
    tri: FXTriangle,
    x_edges: np.ndarray | None = None,
    y_edges: np.ndarray | None = None,
    *,
    n_edges: int = 4,
    **kwargs: Any,
) -> HoeffdingResult:
    """Localize the risk-neutral covariance across the joint distribution.

    Hoeffding's identity writes the covariance of the *returns* as
    :math:`\\iint (F_{12}(x, y) - F_1(x) F_2(y))\\,dx\\,dy`; evaluating the
    integrand cellwise (2-D trapezoid) shows which region of the joint
    distribution drives dependence. Edges are in return units
    (:math:`S_i/F_i`); default: ``n_edges`` points spanning each grid.
    Port of ``hoeffdingCovCells.m``.
    """
    grids = kwargs.pop("grids", None)
    n_grid = kwargs.pop("n_grid", 700)
    if grids is None:
        g1, g2 = _leg_grid(tri.leg1, n_grid), _leg_grid(tri.leg2, n_grid)
    else:
        g1, g2 = (np.asarray(g, dtype=float) for g in grids)
    F1, F2 = tri.leg1.forward, tri.leg2.forward
    if x_edges is None:
        x_edges = np.linspace(g1[0], g1[-1], n_edges) / F1
    else:
        x_edges = np.asarray(x_edges, dtype=float)
    if y_edges is None:
        y_edges = np.linspace(g2[0], g2[-1], n_edges) / F2
    else:
        y_edges = np.asarray(y_edges, dtype=float)

    # joint CDF at every edge node, one multi-target solve
    xx, yy = np.meshgrid(x_edges, y_edges)
    nodes = np.column_stack([xx.ravel(), yy.ravel()])
    fit_cdf = joint_projection(
        lambda s1, s2: (
            ((s1 / F1)[:, None] <= nodes[None, :, 0])
            & ((s2 / F2)[:, None] <= nodes[None, :, 1])
        ).astype(float),
        tri,
        grids=(g1, g2),
        **kwargs,
    )
    Fxy = np.asarray(fit_cdf.value).reshape(yy.shape)

    Fx = np.array([_marginal_tail(tri.leg1, g1, a) for a in x_edges])
    Fy = np.array([_marginal_tail(tri.leg2, g2, a) for a in y_edges])
    Fprod = Fy[:, None] * Fx[None, :]

    ny, nx = len(y_edges), len(x_edges)
    joint_cells = np.zeros((ny - 1, nx - 1))
    marg_cells = np.zeros((ny - 1, nx - 1))
    for i in range(ny - 1):
        dy = y_edges[i + 1] - y_edges[i]
        for j in range(nx - 1):
            dx = x_edges[j + 1] - x_edges[j]
            joint_cells[i, j] = dx * dy * (
                Fxy[i, j] + Fxy[i, j + 1] + Fxy[i + 1, j] + Fxy[i + 1, j + 1]
            ) / 4.0
            marg_cells[i, j] = dx * dy * (
                Fprod[i, j] + Fprod[i, j + 1] + Fprod[i + 1, j] + Fprod[i + 1, j + 1]
            ) / 4.0
    cells = joint_cells - marg_cells
    total = float(cells.sum())
    return HoeffdingResult(
        x_edges=x_edges,
        y_edges=y_edges,
        cells=cells,
        joint_cells=joint_cells,
        marginal_cells=marg_cells,
        total=total,
        share=cells / total if total != 0 else np.full_like(cells, np.nan),
    )


# ---------------------------------------------------------------------- #
def _marginal_variance(chain: OptionChain, grid: np.ndarray) -> float:
    F = chain.forward
    fit = project(
        lambda s: (s - F) ** 2, chain, grid=grid, weights=np.ones_like(grid), otm_only=False
    )
    return float(fit.value)


def _marginal_tail(chain: OptionChain, grid: np.ndarray, a: float) -> float:
    F = chain.forward
    fit = project(
        lambda s: (s / F <= a).astype(float),
        chain,
        grid=grid,
        weights=np.ones_like(grid),
        otm_only=False,
    )
    return float(fit.value)
