"""rnproj: risk-neutral expectations from option prices by projection.

Implements the projection estimator of De Vries (2026), "A Projection
Approach for Estimating Risk-Neutral Expectations": project the target
payoff onto the span of observed option payoffs by weighted least squares,
then price the projection with observed option prices. A finite-sample
near-optimality bound controls the estimation error, and the same machinery
delivers option-implied moments, distributions, and (bivariate) dependence.

Quickstart::

    import rnproj

    chain = rnproj.OptionChain.from_arrays(
        strikes, calls=call_prices, puts=put_prices,
        forward=5321.4, maturity=30 / 365, rate=0.043,
    )
    fit = rnproj.expectation(lambda s: (s / chain.forward - 1) ** 2, chain)
    print(fit.value)          # E^Q[(S_T/F - 1)^2]
    print(fit.portfolio())    # the replicating option portfolio
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .carr_madan import carr_madan, carr_madan_cdf, carr_madan_sparse
from .chain import OptionChain
from .distribution import DistributionFit, implied_cdf, implied_pdf
from .grids import default_grid
from .moments import Moments, implied_moments, svix, vix
from .projection import BidAsk, Projection, project
from .weights import (
    WeightSpec,
    automatic_weights,
    lognormal_weights,
    vg_weights,
    weights_from_density,
)

__version__ = "0.1.0"

__all__ = [
    "OptionChain",
    "Projection",
    "BidAsk",
    "WeightSpec",
    "Moments",
    "DistributionFit",
    "project",
    "expectation",
    "implied_moments",
    "implied_cdf",
    "implied_pdf",
    "vix",
    "svix",
    "carr_madan",
    "carr_madan_sparse",
    "carr_madan_cdf",
    "default_grid",
    "automatic_weights",
    "vg_weights",
    "lognormal_weights",
    "weights_from_density",
    "__version__",
]


def expectation(
    g: Callable[[np.ndarray], np.ndarray] | np.ndarray,
    chain: OptionChain,
    **kwargs: Any,
) -> Projection:
    """Estimate :math:`E^Q[g(S_T)]` with automatic grid and weighting density.

    Convenience alias for :func:`rnproj.projection.project`; see there for
    all options.
    """
    return project(g, chain, **kwargs)
