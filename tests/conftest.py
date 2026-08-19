"""Shared test fixtures: synthetic Black-Scholes option chains."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from rnproj import OptionChain


def black_prices(forward, strikes, maturity, rate, sigma):
    """Discounted Black (1976) call and put prices from the forward."""
    strikes = np.asarray(strikes, dtype=float)
    df = np.exp(-rate * maturity)
    srt = sigma * np.sqrt(maturity)
    d1 = np.log(forward / strikes) / srt + 0.5 * srt
    d2 = d1 - srt
    calls = df * (forward * norm.cdf(d1) - strikes * norm.cdf(d2))
    puts = df * (strikes * norm.cdf(-d2) - forward * norm.cdf(-d1))
    return calls, puts


def make_bs_chain(
    forward=100.0,
    sigma=0.2,
    maturity=1.0 / 12.0,
    rate=0.02,
    strikes=None,
):
    """A Black-Scholes chain quoting calls and puts at every strike."""
    if strikes is None:
        strikes = np.linspace(60.0, 150.0, 40)
    calls, puts = black_prices(forward, strikes, maturity, rate, sigma)
    return OptionChain.from_arrays(
        strikes, calls=calls, puts=puts, forward=forward, maturity=maturity, rate=rate
    )


def make_vg_chain(
    sigma=0.18,
    nu=0.4,
    theta=-0.12,
    forward=100.0,
    maturity=0.25,
    rate=0.02,
    strikes=None,
):
    """A chain priced under a Variance-Gamma model (calls via FFT, puts via parity)."""
    from rnproj._vg import VGParams, vg_call_prices

    if strikes is None:
        strikes = np.linspace(70.0, 140.0, 15)
    strikes = np.asarray(strikes, dtype=float)
    calls = vg_call_prices(VGParams(sigma, nu, theta), strikes, rate, maturity, forward)
    puts = calls - (forward - strikes) * np.exp(-rate * maturity)
    return OptionChain.from_arrays(
        strikes, calls=calls, puts=puts, forward=forward, maturity=maturity, rate=rate
    )


@pytest.fixture
def bs_chain():
    return make_bs_chain()


@pytest.fixture
def vg_chain():
    return make_vg_chain()


@pytest.fixture
def bs_sparse_chain():
    """Five FX-style strikes, poorly centered on the forward."""
    return make_bs_chain(strikes=np.array([85.0, 93.0, 100.5, 108.0, 118.0]))
