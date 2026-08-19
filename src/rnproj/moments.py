"""Option-implied moments and variance indices.

Conventions (matching the paper's empirical code):

- ``vix(chain)**2  = (2/T) * E^Q[log(F/S_T)]`` -- the risk-neutral entropy
  measure behind the CBOE VIX. The CBOE index quotes ``100 * vix(chain)``.
- ``svix(chain)**2 = (1/T) * (E^Q[(S_T/F)^2] - 1)`` -- Martin's (2017) SVIX,
  the annualized risk-neutral variance of the simple return. (With zero
  dividends these coincide with the spot-return definitions
  ``(2/T)(log R_f - E log R)`` and ``(E[R^2] - R_f^2)/(T R_f^2)``.)
- :func:`implied_moments` returns central moments of the log return
  ``log(S_T/F)`` by default (the Bakshi-Kapadia-Madan convention), or of the
  simple return ``S_T/F - 1``, or raw price moments ``E[S_T^k]``.

All quantities are estimated with a single projection solve (one column per
required power).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Any

import numpy as np

from .chain import OptionChain
from .projection import Projection, project

__all__ = ["Moments", "implied_moments", "vix", "svix"]


@dataclass(frozen=True)
class Moments:
    """Option-implied moments of one maturity.

    Attributes
    ----------
    of : str
        ``"log"`` (log return :math:`\\log(S_T/F)`), ``"simple"``
        (:math:`S_T/F - 1`), or ``"price"`` (:math:`S_T`).
    raw : numpy.ndarray
        Raw moments :math:`E^Q[x^k]` for k = 1..kmax.
    mean, variance, skewness, kurtosis : float
        Central moments and standardized shape measures (kurtosis is the
        plain standardized fourth moment; 3 = normal).
    fit : Projection
        The underlying projection (portfolio weights, diagnostics).
    """

    of: str
    raw: np.ndarray
    mean: float
    variance: float
    skewness: float
    kurtosis: float
    fit: Projection

    def central(self, k: int) -> float:
        """Central moment :math:`E^Q[(x - E^Q x)^k]` from the raw moments."""
        if k > self.raw.size:
            raise ValueError(f"only moments up to order {self.raw.size} were estimated.")
        mu = self.mean
        raw_ext = np.concatenate([[1.0], self.raw])  # raw_ext[j] = E[x^j]
        return float(
            sum(comb(k, j) * raw_ext[j] * (-mu) ** (k - j) for j in range(k + 1))
        )


def implied_moments(
    chain: OptionChain,
    *,
    orders: int = 4,
    of: str = "log",
    **kwargs: Any,
) -> Moments:
    """Estimate option-implied moments up to ``orders``.

    Parameters
    ----------
    orders : int
        Highest moment to estimate (>= 2).
    of : {"log", "simple", "price"}
        The variable whose moments are estimated: log return
        ``log(S_T/F)`` (default, the Bakshi-Kapadia-Madan convention),
        simple return
        ``S_T/F - 1`` (whose risk-neutral mean is exactly 0), or the raw
        price ``S_T``.
    **kwargs
        Passed to :func:`rnproj.project` (``weights``, ``grid``,
        ``otm_only``, ...).
    """
    if orders < 2:
        raise ValueError("orders must be at least 2.")
    if of not in ("log", "simple", "price"):
        raise ValueError(f"of must be 'log', 'simple', or 'price', got {of!r}.")

    F = chain.forward
    powers = np.arange(1, orders + 1)

    def transform(s: np.ndarray) -> np.ndarray:
        if of == "log":
            x = np.log(s / F)
        elif of == "simple":
            x = s / F - 1.0
        else:
            x = s
        return x[:, None] ** powers

    fit = project(transform, chain, **kwargs)
    raw = np.asarray(fit.value, dtype=float)

    mean = raw[0]
    raw_ext = np.concatenate([[1.0], raw])
    central = [
        float(sum(comb(k, j) * raw_ext[j] * (-mean) ** (k - j) for j in range(k + 1)))
        for k in range(2, orders + 1)
    ]
    variance = central[0]
    skewness = central[1] / variance**1.5 if orders >= 3 else np.nan
    kurtosis = central[2] / variance**2 if orders >= 4 else np.nan

    return Moments(
        of=of,
        raw=raw,
        mean=float(mean),
        variance=float(variance),
        skewness=float(skewness),
        kurtosis=float(kurtosis),
        fit=fit,
    )


def vix(chain: OptionChain, **kwargs: Any) -> float:
    """The VIX-style annualized volatility (decimal units).

    ``vix**2 = (2/T) * E^Q[log(F/S_T)]``, the risk-neutral entropy measure
    underlying the CBOE VIX; the index itself quotes ``100 * vix``.
    """
    fit = project(lambda s: np.log(chain.forward / s), chain, **kwargs)
    return float(np.sqrt(2.0 / chain.maturity * fit.value))


def svix(chain: OptionChain, **kwargs: Any) -> float:
    """Martin's (2017) SVIX annualized volatility (decimal units).

    ``svix**2 = (1/T) * (E^Q[(S_T/F)^2] - 1)``, the annualized risk-neutral
    variance of the simple return.
    """
    fit = project(lambda s: (s / chain.forward) ** 2, chain, **kwargs)
    return float(np.sqrt((fit.value - 1.0) / chain.maturity))
