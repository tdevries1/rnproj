"""FX quote conversion: Bloomberg-style (ATM, RR, BF) vol quotes to strikes
and Garman-Kohlhagen prices.

OTC FX options are quoted as at-the-money (delta-neutral) volatility, risk
reversals, and butterflies at fixed deltas. These helpers reconstruct the
smile, solve the delta convention for strikes, and price the quotes, so a
five-point OTC surface becomes an :class:`rnproj.OptionChain` in one call.
Formulas match ``fx_make_prices.m`` from the Matlab reference (spot-delta
convention, ATM delta-neutral strike, GK forward-form pricing).
"""

from __future__ import annotations

import numpy as np

from .chain import OptionChain

__all__ = [
    "garman_kohlhagen",
    "atm_delta_neutral_strike",
    "smile_from_quotes",
    "strike_from_spot_delta",
    "chain_from_fx_quotes",
]


def garman_kohlhagen(forward, strike, vol, maturity, domestic_df, kind="call"):
    """Garman-Kohlhagen option price in forward form.

    ``domestic_df`` is the domestic discount factor (the price currency's).
    All inputs broadcast elementwise.
    """
    from scipy.stats import norm

    F, K, sig, T, dd = (np.asarray(a, dtype=float) for a in
                        (forward, strike, vol, maturity, domestic_df))
    srt = sig * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sig**2 * T) / srt
    d2 = d1 - srt
    if kind == "call":
        return dd * (F * norm.cdf(d1) - K * norm.cdf(d2))
    if kind == "put":
        return dd * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    raise ValueError(f"kind must be 'call' or 'put', got {kind!r}.")


def atm_delta_neutral_strike(forward, vol, maturity):
    """The delta-neutral ATM strike :math:`K = F e^{\\sigma^2 T / 2}`.

    Note this sits *above* the forward, which is why sparse FX chains should
    be projected with ``otm_only=False``.
    """
    return np.asarray(forward, float) * np.exp(
        0.5 * np.asarray(vol, float) ** 2 * np.asarray(maturity, float)
    )


def smile_from_quotes(atm, rr, bf):
    """Call and put vols at one delta from (ATM, risk-reversal, butterfly).

    ``sigma_call = atm + (bf + rr)/2`` and ``sigma_put = atm + (bf - rr)/2``
    (the convention of the Matlab reference). All inputs in decimal vols.
    """
    atm, rr, bf = (np.asarray(a, dtype=float) for a in (atm, rr, bf))
    return atm + 0.5 * (bf + rr), atm + 0.5 * (bf - rr)


def strike_from_spot_delta(delta, vol, forward, maturity, foreign_df, kind="call"):
    """Strike for a given *spot delta* magnitude (e.g. 0.10, 0.25).

    FX spot-delta conventions: call :math:`\\Delta_S = D_f N(d_1)`, put
    :math:`\\Delta_S = -D_f N(-d_1)` with ``foreign_df`` the foreign
    discount factor; converted to forward delta and inverted for K.
    """
    from scipy.stats import norm

    delta, sig, F, T, df = (np.asarray(a, dtype=float) for a in
                            (delta, vol, forward, maturity, foreign_df))
    if kind == "call":
        delta_f = np.clip(np.abs(delta) / df, 1e-6, 1 - 1e-6)
    elif kind == "put":
        delta_f = np.clip(1.0 - np.abs(delta) / df, 1e-6, 1 - 1e-6)
    else:
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}.")
    d1 = norm.ppf(delta_f)
    srt = sig * np.sqrt(T)
    return F * np.exp(-srt * d1 + 0.5 * sig**2 * T)


def chain_from_fx_quotes(
    *,
    atm: float,
    rr25: float,
    bf25: float,
    rr10: float,
    bf10: float,
    forward: float,
    maturity: float,
    domestic_df: float,
    foreign_df: float,
    spot: float | None = None,
) -> OptionChain:
    """Build the standard five-strike OTC chain from Bloomberg-style quotes.

    Strikes: 10-delta put, 25-delta put, delta-neutral ATM (entered as a
    put, matching the Matlab reference), 25-delta call, 10-delta call. All
    vols in decimals; ``domestic_df`` prices the options and sets the
    chain's discount, ``foreign_df`` drives the spot-delta convention.

    Project chains built this way with ``otm_only=False``: the ATM strike
    lies above the forward and an OTM filter would discard it.
    """
    c25, p25 = smile_from_quotes(atm, rr25, bf25)
    c10, p10 = smile_from_quotes(atm, rr10, bf10)

    k_atm = float(atm_delta_neutral_strike(forward, atm, maturity))
    k_c25 = float(strike_from_spot_delta(0.25, c25, forward, maturity, foreign_df, "call"))
    k_p25 = float(strike_from_spot_delta(0.25, p25, forward, maturity, foreign_df, "put"))
    k_c10 = float(strike_from_spot_delta(0.10, c10, forward, maturity, foreign_df, "call"))
    k_p10 = float(strike_from_spot_delta(0.10, p10, forward, maturity, foreign_df, "put"))

    put_strikes = np.array([k_p10, k_p25, k_atm])
    put_vols = np.array([p10, p25, atm], dtype=float)
    call_strikes = np.array([k_c25, k_c10])
    call_vols = np.array([c25, c10], dtype=float)

    return OptionChain(
        put_strikes=put_strikes,
        put_prices=garman_kohlhagen(forward, put_strikes, put_vols, maturity,
                                    domestic_df, "put"),
        call_strikes=call_strikes,
        call_prices=garman_kohlhagen(forward, call_strikes, call_vols, maturity,
                                     domestic_df, "call"),
        forward=forward,
        maturity=maturity,
        discount=domestic_df,
        spot=spot,
    )
