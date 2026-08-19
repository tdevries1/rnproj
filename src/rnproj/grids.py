"""State-grid construction.

The projection estimator is a weighted least-squares fit on a state grid, so
the grid is a numerical device, not a modeling choice. Two requirements make
a grid safe:

1. **Coverage** -- the grid must extend beyond every observed strike (with
   padding), otherwise the outermost hinge payoffs are truncated and the
   basis loses its linear-extrapolation budget.
2. **Resolution** -- the mesh must be finer than the smallest gap between
   consecutive strikes. Between two adjacent strikes both hinge payoffs are
   affine, so with fewer than two interior grid points the corresponding
   design-matrix columns become collinear with the constant and the linear
   term, and the design matrix loses full column rank.

:func:`default_grid` is the single grid-construction authority used by every
layer of the package.
"""

from __future__ import annotations

import warnings
from typing import Any, Protocol

import numpy as np

__all__ = ["default_grid", "validate_grid"]

#: Hard cap on automatic grid size; a warning is raised when it binds.
MAX_GRID_SIZE = 2_000_000


class _HasPpf(Protocol):
    def ppf(self, q: float) -> float: ...


def default_grid(
    strikes: Any,
    *,
    distribution: _HasPpf | None = None,
    alpha: float = 1e-3,
    strike_padding: float = 0.05,
    n_min: int = 1000,
    safety_factor: float = 2.0,
) -> np.ndarray:
    """Build a uniform state grid for the projection estimator.

    Parameters
    ----------
    strikes : array_like
        All observed strikes (puts and calls together).
    distribution : object with ``.ppf``, optional
        A distribution for the underlying (e.g. the weighting density). When
        given, the grid endpoints are pushed out to its ``alpha`` and
        ``1 - alpha`` quantiles if those lie beyond the padded strike range.
    alpha : float
        Tail mass left outside the grid on each side when ``distribution``
        is given.
    strike_padding : float
        Relative padding beyond the outermost strikes (default 5%).
    n_min : int
        Minimum number of grid points.
    safety_factor : float
        The mesh is at most ``min(diff(strikes)) / safety_factor``, so with
        the default there are at least two grid points between any pair of
        consecutive strikes.

    Returns
    -------
    numpy.ndarray
        Uniform, strictly increasing grid covering all strikes.
    """
    strikes = np.unique(np.asarray(strikes, dtype=float))
    if strikes.size == 0:
        raise ValueError("default_grid needs at least one strike.")
    if np.any(strikes <= 0) or not np.all(np.isfinite(strikes)):
        raise ValueError("strikes must be positive and finite.")

    lo = (1.0 - strike_padding) * strikes[0]
    hi = (1.0 + strike_padding) * strikes[-1]
    if distribution is not None:
        lo = min(lo, float(distribution.ppf(alpha)))
        hi = max(hi, float(distribution.ppf(1.0 - alpha)))
    lo = max(lo, 1e-12)

    # Resolution rule: mesh finer than the smallest strike gap.
    if strikes.size > 1:
        min_gap = float(np.min(np.diff(strikes)))
        h_max = min_gap / safety_factor
        n = int(np.ceil((hi - lo) / h_max)) + 1
    else:
        n = n_min
    n = max(n, n_min)
    if n > MAX_GRID_SIZE:
        warnings.warn(
            f"Automatic grid would need {n} points to resolve the smallest "
            f"strike gap; capping at {MAX_GRID_SIZE}. Nearly coincident "
            "strikes may make the design matrix ill-conditioned.",
            stacklevel=2,
        )
        n = MAX_GRID_SIZE

    return np.linspace(lo, hi, n)


def validate_grid(grid: Any, strikes: np.ndarray, *, safety_factor: float = 2.0) -> np.ndarray:
    """Validate a user-supplied state grid against the observed strikes.

    Raises on missing strike coverage (hard error, as in the Matlab
    reference) and warns when the mesh is coarser than
    ``min(diff(strikes)) / safety_factor``.
    """
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 1 or grid.size < 2:
        raise ValueError("grid must be a one-dimensional array with at least 2 points.")
    if np.any(np.diff(grid) <= 0):
        raise ValueError("grid must be strictly increasing.")
    if strikes.size:
        if strikes.min() < grid[0] or strikes.max() > grid[-1]:
            raise ValueError(
                f"state grid [{grid[0]:.6g}, {grid[-1]:.6g}] does not cover all "
                f"observed strikes [{strikes.min():.6g}, {strikes.max():.6g}]. "
                "Widen the grid (or omit it to use the automatic grid)."
            )
        distinct = np.unique(strikes)
        if distinct.size > 1:
            min_gap = float(np.min(np.diff(distinct)))
            if float(np.max(np.diff(grid))) > min_gap / safety_factor:
                warnings.warn(
                    "grid mesh is coarser than half the smallest strike gap; "
                    "the design matrix may lose full column rank. Use a finer "
                    "grid (or omit it to use the automatic grid).",
                    stacklevel=2,
                )
    return grid
