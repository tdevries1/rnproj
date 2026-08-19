"""FX quote conversion: delta round trips and chain construction."""

import numpy as np
import pytest
from scipy.stats import norm

from rnproj import project
from rnproj.fx import (
    atm_delta_neutral_strike,
    chain_from_fx_quotes,
    garman_kohlhagen,
    smile_from_quotes,
    strike_from_spot_delta,
)

F, T = 1.10, 1.0 / 12.0
DD, DF = 0.998, 0.997  # domestic / foreign discount factors


class TestFormulas:
    def test_gk_put_call_parity(self):
        k = np.array([1.05, 1.10, 1.15])
        c = garman_kohlhagen(F, k, 0.1, T, DD, "call")
        p = garman_kohlhagen(F, k, 0.1, T, DD, "put")
        np.testing.assert_allclose(c - p, DD * (F - k), rtol=1e-12)

    def test_atm_dn_strike_above_forward(self):
        k = atm_delta_neutral_strike(F, 0.1, T)
        assert k > F
        assert k == pytest.approx(F * np.exp(0.5 * 0.01 * T), rel=1e-14)

    def test_smile_reconstruction(self):
        c, p = smile_from_quotes(atm=0.10, rr=-0.02, bf=0.004)
        assert c - p == pytest.approx(-0.02, rel=1e-12)  # RR = call - put
        assert c == pytest.approx(0.10 + 0.5 * (0.004 - 0.02), rel=1e-12)

    def test_delta_round_trip(self):
        # the strike solved for a 25-delta call must have spot delta 0.25
        sigma = 0.11
        k = strike_from_spot_delta(0.25, sigma, F, T, DF, "call")
        srt = sigma * np.sqrt(T)
        d1 = (np.log(F / k) + 0.5 * sigma**2 * T) / srt
        assert DF * norm.cdf(d1) == pytest.approx(0.25, rel=1e-10)

    def test_put_delta_round_trip(self):
        sigma = 0.12
        k = strike_from_spot_delta(0.10, sigma, F, T, DF, "put")
        srt = sigma * np.sqrt(T)
        d1 = (np.log(F / k) + 0.5 * sigma**2 * T) / srt
        assert -DF * norm.cdf(-d1) == pytest.approx(-0.10, rel=1e-10)


class TestChainConstruction:
    def make(self):
        return chain_from_fx_quotes(
            atm=0.10, rr25=-0.015, bf25=0.003, rr10=-0.025, bf10=0.008,
            forward=F, maturity=T, domestic_df=DD, foreign_df=DF,
        )

    def test_five_strikes_ordered(self):
        chain = self.make()
        assert chain.put_strikes.size == 3 and chain.call_strikes.size == 2
        strikes = np.concatenate([chain.put_strikes, chain.call_strikes])
        # 10dP < 25dP < ATM-DN and ATM-DN above forward; calls above puts' top
        assert np.all(np.diff(chain.put_strikes) > 0)
        assert chain.put_strikes[-1] > F  # the ATM-DN strike
        assert strikes.min() < F < strikes.max()

    def test_prices_positive_and_arbitrage_sane(self):
        chain = self.make()
        assert np.all(chain.put_prices > 0) and np.all(chain.call_prices > 0)
        # farther-OTM options are cheaper
        assert chain.put_prices[0] < chain.put_prices[1]
        assert chain.call_prices[1] < chain.call_prices[0]

    def test_projection_on_fx_chain(self):
        # flat-smile chain: implied variance should be near atm^2 * T
        chain = chain_from_fx_quotes(
            atm=0.10, rr25=0.0, bf25=0.0, rr10=0.0, bf10=0.0,
            forward=F, maturity=T, domestic_df=1.0, foreign_df=1.0,
        )
        fit = project(lambda s: (s / F - 1.0) ** 2, chain, otm_only=False)
        assert fit.value == pytest.approx(np.expm1(0.10**2 * T), rel=5e-2)
