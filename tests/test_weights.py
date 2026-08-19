"""Weighting densities: VG calibration, lognormal fallback, grid rules."""

import numpy as np
import pytest

from conftest import make_bs_chain
from rnproj import automatic_weights, lognormal_weights, project, vg_weights
from rnproj._vg import VGParams, VGQuantiles, calibrate_vg, vg_call_prices, vg_price_density

TRAP = getattr(np, "trapezoid", getattr(np, "trapz", None))


class TestVGCore:
    def test_density_integrates_to_one(self):
        params = VGParams(sigma=0.18, nu=0.4, theta=-0.12)
        F, T = 100.0, 0.25
        s = np.linspace(1.0, 400.0, 20000)
        pdf = vg_price_density(params, T, F, s)
        assert TRAP(pdf, s) == pytest.approx(1.0, abs=1e-4)

    def test_density_martingale_mean(self):
        params = VGParams(sigma=0.18, nu=0.4, theta=-0.12)
        F, T = 100.0, 0.25
        s = np.linspace(0.5, 500.0, 40000)
        pdf = vg_price_density(params, T, F, s)
        assert TRAP(s * pdf, s) == pytest.approx(F, rel=1e-3)

    def test_fft_call_prices_match_density_pricing(self):
        params = VGParams(sigma=0.18, nu=0.4, theta=-0.12)
        F, T, r = 100.0, 0.25, 0.02
        strikes = np.array([80.0, 95.0, 100.0, 110.0, 125.0])
        fft_prices = vg_call_prices(params, strikes, r, T, F)
        s = np.linspace(0.5, 600.0, 60000)
        pdf = vg_price_density(params, T, F, s)
        for k, p_fft in zip(strikes, fft_prices):
            quad = np.exp(-r * T) * TRAP(np.maximum(s - k, 0.0) * pdf, s)
            assert p_fft == pytest.approx(quad, rel=2e-3)

    def test_calibration_self_consistency(self):
        # Prices generated from known VG params must be recovered.
        true = VGParams(sigma=0.18, nu=0.4, theta=-0.12)
        F, T, r = 100.0, 0.25, 0.02
        strikes = np.linspace(70.0, 140.0, 15)
        prices = vg_call_prices(true, strikes, r, T, F)
        fitted, cost = calibrate_vg(strikes, prices, r, T, F, sigma_seed=0.2)
        assert fitted is not None
        assert cost < 1e-6
        assert fitted.sigma == pytest.approx(true.sigma, rel=1e-2)
        assert fitted.nu == pytest.approx(true.nu, rel=5e-2)
        assert fitted.theta == pytest.approx(true.theta, rel=5e-2)

    def test_quantiles_bracket_forward(self):
        params = VGParams(sigma=0.18, nu=0.4, theta=-0.12)
        q = VGQuantiles(params, 0.25, 100.0)
        assert q.ppf(0.001) < 100.0 < q.ppf(0.999)
        assert q.ppf(0.001) < q.ppf(0.5) < q.ppf(0.999)


class TestWeightSpecs:
    def test_vg_weights_on_vg_chain(self, vg_chain):
        spec = vg_weights(vg_chain)
        assert spec.params["family"] == "vg"
        assert not spec.params["fallback"]
        assert np.all(spec.weights > 0)
        # grid rules: covers strikes with padding, resolves min strike gap
        strikes = vg_chain.strikes
        assert spec.grid[0] <= strikes.min() * 0.95 + 1e-9
        assert spec.grid[-1] >= strikes.max() * 1.05 - 1e-9
        assert np.max(np.diff(spec.grid)) <= np.min(np.diff(strikes)) / 2 + 1e-12

    def test_automatic_weights_projection_accuracy(self, bs_chain):
        F, T = bs_chain.forward, bs_chain.maturity
        fit = project(lambda s: s**2, bs_chain)  # automatic weights
        assert fit.value == pytest.approx(F**2 * np.exp(0.2**2 * T), rel=1e-4)

    def test_fallback_on_few_strikes(self):
        chain = make_bs_chain(strikes=np.array([95.0, 100.5, 106.0]))
        spec = vg_weights(chain)
        assert spec.params["fallback"]
        assert spec.params["family"] == "lognormal"

    def test_bs_chain_falls_back_to_lognormal(self, bs_chain):
        # A flat BS smile is the nu -> 0 limit of VG: the calibration pins
        # nu at its bound, and the designed fallback picks the lognormal --
        # which IS the correct weighting density for this chain.
        spec = automatic_weights(bs_chain)
        assert spec.params["family"] == "lognormal"
        assert spec.params["fallback"]

    def test_lognormal_sigma_inferred(self, bs_chain):
        spec = lognormal_weights(bs_chain)
        assert spec.params["sigma"] == pytest.approx(0.2, rel=1e-2)

    def test_automatic_weights_is_vg_on_vg_chain(self, vg_chain):
        spec = automatic_weights(vg_chain)
        assert spec.params["family"] == "vg"


class TestFallbackRegimes:
    def test_low_vol_regime_still_works(self):
        # Floor-pinned FX style: tiny vol, few strikes.
        strikes = np.array([0.97, 0.99, 1.0, 1.01, 1.03])
        chain = make_bs_chain(
            forward=1.0, sigma=0.02, maturity=1.0 / 12.0, rate=0.0, strikes=strikes
        )
        fit = project(lambda s: (s - 1.0) ** 2, chain, otm_only=False)
        assert fit.value == pytest.approx(0.02**2 / 12.0, rel=5e-2)
