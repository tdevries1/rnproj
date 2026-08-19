"""Option-chain container.

A single maturity of European option quotes. The forward price is mandatory:
carry, dividends, and foreign interest all live in the forward, and the
package never infers ``F = exp(r*T)*spot``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

__all__ = ["OptionChain"]


def _as_1d_float(x: Any, name: str) -> np.ndarray:
    arr = np.atleast_1d(np.asarray(x, dtype=float))
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {arr.shape}.")
    return arr


@dataclass(frozen=True)
class OptionChain:
    """European option quotes for one underlying and one maturity.

    Parameters
    ----------
    call_strikes, call_prices : array_like
        Call strikes and prices (same length; may be empty).
    put_strikes, put_prices : array_like
        Put strikes and prices (same length; may be empty).
    forward : float
        Forward price :math:`F_{t \\to T}` of the underlying for delivery at T.
    maturity : float
        Time to maturity T in year fractions.
    rate : float, optional
        Continuously-compounded risk-free rate. Give exactly one of
        ``rate`` and ``discount``.
    discount : float, optional
        Discount factor :math:`e^{-rT}` (equals 1 / gross risk-free rate).
    spot : float, optional
        Spot price; informational only, never used in computations.
    """

    call_strikes: np.ndarray
    call_prices: np.ndarray
    put_strikes: np.ndarray
    put_prices: np.ndarray
    forward: float
    maturity: float
    rate: float = field(default=None)  # type: ignore[assignment]
    discount: float = field(default=None)  # type: ignore[assignment]
    spot: float | None = None

    def __post_init__(self) -> None:
        for prices_name, strikes_name in (
            ("call_prices", "call_strikes"),
            ("put_prices", "put_strikes"),
        ):
            strikes = _as_1d_float(getattr(self, strikes_name), strikes_name)
            prices = _as_1d_float(getattr(self, prices_name), prices_name)
            if strikes.shape != prices.shape:
                raise ValueError(
                    f"{strikes_name} and {prices_name} must have the same length "
                    f"({strikes.size} vs {prices.size})."
                )
            if np.any(strikes <= 0) or not np.all(np.isfinite(strikes)):
                raise ValueError(f"{strikes_name} must be positive and finite.")
            if np.any(prices < 0) or not np.all(np.isfinite(prices)):
                raise ValueError(f"{prices_name} must be nonnegative and finite.")
            order = np.argsort(strikes, kind="stable")
            object.__setattr__(self, strikes_name, strikes[order])
            object.__setattr__(self, prices_name, prices[order])

        if not (np.isfinite(self.forward) and self.forward > 0):
            raise ValueError(f"forward must be positive and finite, got {self.forward!r}.")
        if not (np.isfinite(self.maturity) and self.maturity > 0):
            raise ValueError(f"maturity must be positive and finite, got {self.maturity!r}.")

        if (self.rate is None) == (self.discount is None):
            raise ValueError("Give exactly one of `rate` and `discount`.")
        if self.rate is None:
            if not (np.isfinite(self.discount) and self.discount > 0):
                raise ValueError(f"discount must be positive and finite, got {self.discount!r}.")
            object.__setattr__(self, "rate", -np.log(self.discount) / self.maturity)
        else:
            if not np.isfinite(self.rate):
                raise ValueError(f"rate must be finite, got {self.rate!r}.")
            object.__setattr__(self, "discount", float(np.exp(-self.rate * self.maturity)))

    # ------------------------------------------------------------------ #
    @property
    def gross_rate(self) -> float:
        """Gross risk-free rate :math:`R_{f,t\\to T} = e^{rT} = 1/\\text{discount}`."""
        return 1.0 / self.discount

    @property
    def strikes(self) -> np.ndarray:
        """Sorted union of all put and call strikes (distinct values)."""
        return np.unique(np.concatenate([self.put_strikes, self.call_strikes]))

    @property
    def n_options(self) -> int:
        return self.call_strikes.size + self.put_strikes.size

    # ------------------------------------------------------------------ #
    def otm(self) -> OptionChain:
        """Copy with only out-of-the-money quotes, split at the forward.

        Puts with :math:`K \\le F` and calls with :math:`K > F` are kept; all
        other quotes are dropped. Mirrors the OTM filter of the Matlab
        reference (``ols_projection.m``).
        """
        keep_p = self.put_strikes <= self.forward
        keep_c = self.call_strikes > self.forward
        # discount=None: validation re-runs on replace and requires exactly
        # one of rate/discount; the discount is re-derived from the rate.
        return replace(
            self,
            put_strikes=self.put_strikes[keep_p],
            put_prices=self.put_prices[keep_p],
            call_strikes=self.call_strikes[keep_c],
            call_prices=self.call_prices[keep_c],
            discount=None,
        )

    # ------------------------------------------------------------------ #
    @classmethod
    def from_arrays(
        cls,
        strikes: Any,
        calls: Any = None,
        puts: Any = None,
        *,
        forward: float,
        maturity: float,
        rate: float | None = None,
        discount: float | None = None,
        spot: float | None = None,
    ) -> OptionChain:
        """Build a chain from one strike array with call and/or put prices.

        ``calls`` and ``puts`` are price arrays aligned with ``strikes``;
        missing quotes may be marked ``nan`` and are dropped per leg.
        """
        strikes = _as_1d_float(strikes, "strikes")

        def leg(prices: Any, name: str) -> tuple[np.ndarray, np.ndarray]:
            if prices is None:
                return np.empty(0), np.empty(0)
            prices = _as_1d_float(prices, name)
            if prices.shape != strikes.shape:
                raise ValueError(f"{name} must be aligned with strikes.")
            ok = np.isfinite(prices)
            return strikes[ok], prices[ok]

        kc, c = leg(calls, "calls")
        kp, p = leg(puts, "puts")
        return cls(
            call_strikes=kc,
            call_prices=c,
            put_strikes=kp,
            put_prices=p,
            forward=forward,
            maturity=maturity,
            rate=rate,
            discount=discount,
            spot=spot,
        )
