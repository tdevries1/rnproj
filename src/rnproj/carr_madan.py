"""Carr-Madan benchmark estimators.

The textbook alternative the projection estimator is benchmarked against:

.. math::

    E^Q[g(S_T)] = g(F) + R_f \\Big( \\int_0^F g''(K) P(K)\\,dK
                  + \\int_F^\\infty g''(K) C(K)\\,dK \\Big),

discretized on the observed OTM strikes (trapezoid or Simpson). Ports of
``carr_madan_pricing.m``, ``carr_madan_pricing_sparse.m`` and
``carr_madan_cdf.m``; unlike the Matlab reference, the forward always comes
from the chain (never inferred as ``exp(r*T)*spot``).

These functions exist so every example and simulation can show the
projection and the textbook approach side by side; they are not the
package's recommended estimator.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .chain import OptionChain

__all__ = ["carr_madan", "carr_madan_sparse", "carr_madan_cdf"]

_trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))


def carr_madan(
    g: Callable[[np.ndarray], np.ndarray],
    d2g: Callable[[np.ndarray], np.ndarray],
    chain: OptionChain,
    *,
    method: str = "trapezoid",
) -> float:
    """Carr-Madan estimate of :math:`E^Q[g(S_T)]` from OTM quotes.

    ``d2g`` is the second derivative :math:`g''`. Returns ``nan`` when
    neither wing has at least two OTM strikes; falls back to a single wing
    when only one does (as in the Matlab reference).
    """
    if method == "trapezoid":
        quad = _trapezoid
    elif method == "simpson":
        from scipy.integrate import simpson

        quad = simpson
    else:
        raise ValueError(f"method must be 'trapezoid' or 'simpson', got {method!r}.")

    ch = chain.otm()
    kp, pp = ch.put_strikes, ch.put_prices
    kc, cp = ch.call_strikes, ch.call_prices

    def wing(strikes: np.ndarray, prices: np.ndarray) -> float:
        return float(quad(np.asarray(d2g(strikes), dtype=float) * prices, strikes))

    if kp.size < 2 and kc.size < 2:
        return float("nan")
    total = 0.0
    if kp.size >= 2:
        total += wing(kp, pp)
    if kc.size >= 2:
        total += wing(kc, cp)
    return float(g(np.asarray([chain.forward]))[0] + chain.gross_rate * total)


def carr_madan_sparse(
    g: Callable[[np.ndarray], np.ndarray],
    d2g: Callable[[np.ndarray], np.ndarray],
    chain: OptionChain,
) -> float:
    """Sparse-chain Carr-Madan: parity-convert ITM quotes, Riemann sum.

    Instead of discarding in-the-money quotes, converts them to the OTM side
    by put-call parity (building a full put curve below the forward and a
    full call curve above), then integrates with a central-difference
    Riemann sum. Port of ``carr_madan_pricing_sparse.m``.
    """
    F, df = chain.forward, chain.discount

    put_k = np.concatenate([chain.put_strikes, chain.call_strikes])
    put_p = np.concatenate(
        [chain.put_prices, chain.call_prices - df * (F - chain.call_strikes)]
    )
    below = put_k <= F
    put_k, put_p = put_k[below], put_p[below]
    order = np.argsort(put_k, kind="stable")
    put_k, put_p = put_k[order], put_p[order]

    call_k = np.concatenate([chain.call_strikes, chain.put_strikes])
    call_p = np.concatenate(
        [chain.call_prices, chain.put_prices + df * (F - chain.put_strikes)]
    )
    above = call_k > F
    call_k, call_p = call_k[above], call_p[above]
    order = np.argsort(call_k, kind="stable")
    call_k, call_p = call_k[order], call_p[order]

    def riemann(strikes: np.ndarray, prices: np.ndarray) -> float:
        if strikes.size < 2:
            return 0.0
        dk = np.empty_like(strikes)
        dk[0] = strikes[1] - strikes[0]
        dk[-1] = strikes[-1] - strikes[-2]
        dk[1:-1] = (strikes[2:] - strikes[:-2]) / 2.0
        return float(np.sum(np.asarray(d2g(strikes), dtype=float) * prices * dk))

    if put_k.size < 2 and call_k.size < 2:
        return float("nan")
    total = riemann(put_k, put_p) + riemann(call_k, call_p)
    return float(g(np.asarray([F]))[0] + chain.gross_rate * total)


def carr_madan_cdf(chain: OptionChain, x: np.ndarray) -> np.ndarray:
    """Risk-neutral CDF on ``x`` from finite-difference option-price slopes.

    ``F(K) = R_f dP/dK`` on the OTM put wing and ``F(K) = 1 + R_f dC/dK``
    on the OTM call wing, linearly interpolated onto ``x`` and anchored at
    ``(min(x), 0)`` and ``(max(x), 1)``. Returns all-``nan`` when either
    wing has fewer than two strikes. Port of ``carr_madan_cdf.m``.
    """
    x = np.asarray(x, dtype=float)
    ch = chain.otm()
    if ch.put_strikes.size < 2 or ch.call_strikes.size < 2:
        return np.full(x.size, np.nan)

    rf = chain.gross_rate
    cdf_puts = rf * _matlab_gradient(ch.put_prices, ch.put_strikes)
    cdf_calls = 1.0 + rf * _matlab_gradient(ch.call_prices, ch.call_strikes)

    k = np.concatenate([ch.put_strikes, ch.call_strikes])
    v = np.concatenate([cdf_puts, cdf_calls])
    xp = np.concatenate([[x.min()], k, [x.max()]])
    fp = np.concatenate([[0.0], v, [1.0]])
    return np.clip(np.interp(x, xp, fp), 0.0, 1.0)


def _matlab_gradient(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Matlab-style gradient: simple central differences on the interior
    ((y[i+1]-y[i-1]) / (x[i+1]-x[i-1])), one-sided at the ends. numpy's
    ``gradient`` uses a different second-order formula for uneven spacing,
    so this is implemented explicitly for cross-language reproducibility."""
    out = np.empty_like(y, dtype=float)
    out[0] = (y[1] - y[0]) / (x[1] - x[0])
    out[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
    if y.size > 2:
        out[1:-1] = (y[2:] - y[:-2]) / (x[2:] - x[:-2])
    return out
