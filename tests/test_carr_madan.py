"""Carr-Madan benchmark: analytic checks and agreement with the projection."""

import numpy as np
import pytest
from scipy.stats import norm

from conftest import make_bs_chain
from rnproj import carr_madan, carr_madan_cdf, carr_madan_sparse, project

SIGMA = 0.2


class TestCarrMadan:
    def test_second_moment_dense(self):
        chain = make_bs_chain(strikes=np.linspace(40.0, 250.0, 400))
        F, T = chain.forward, chain.maturity
        est = carr_madan(lambda s: s**2, lambda s: 2.0 * np.ones_like(s), chain)
        # CM carries tail-truncation + discretization error even on a dense
        # chain; ~2e-4 relative here (the projection passes 1e-5 on 40 strikes)
        assert est == pytest.approx(F**2 * np.exp(SIGMA**2 * T), rel=5e-4)

    def test_simpson_at_least_as_good(self):
        chain = make_bs_chain(strikes=np.linspace(40.0, 250.0, 200))
        F, T = chain.forward, chain.maturity
        truth = F**2 * np.exp(SIGMA**2 * T)
        tz = carr_madan(lambda s: s**2, lambda s: 2.0 * np.ones_like(s), chain)
        sp = carr_madan(
            lambda s: s**2, lambda s: 2.0 * np.ones_like(s), chain, method="simpson"
        )
        assert abs(sp - truth) <= abs(tz - truth) * 1.5

    def test_nan_when_too_few_strikes(self):
        chain = make_bs_chain(strikes=np.array([99.0, 101.0]))
        est = carr_madan(lambda s: s**2, lambda s: 2.0 * np.ones_like(s), chain)
        assert np.isnan(est)

    def test_sparse_parity_conversion(self):
        chain = make_bs_chain(strikes=np.linspace(60.0, 150.0, 40))
        F, T = chain.forward, chain.maturity
        est = carr_madan_sparse(lambda s: s**2, lambda s: 2.0 * np.ones_like(s), chain)
        assert est == pytest.approx(F**2 * np.exp(SIGMA**2 * T), rel=5e-3)

    def test_projection_beats_cm_on_sparse_strikes(self, bs_sparse_chain):
        F, T = bs_sparse_chain.forward, bs_sparse_chain.maturity
        truth = F**2 * np.exp(SIGMA**2 * T)
        cm = carr_madan_sparse(
            lambda s: s**2, lambda s: 2.0 * np.ones_like(s), bs_sparse_chain
        )
        proj = project(lambda s: s**2, bs_sparse_chain, otm_only=False).value
        assert abs(proj - truth) < abs(cm - truth)

    def test_prop4_dense_grid_agreement(self):
        # Proposition 4: with dense uniform strikes, projection ~ Carr-Madan.
        chain = make_bs_chain(strikes=np.linspace(50.0, 200.0, 300))
        cm = carr_madan(lambda s: np.log(s), lambda s: -1.0 / s**2, chain)
        proj = project(lambda s: np.log(s), chain).value
        assert proj == pytest.approx(cm, rel=1e-4)


class TestCarrMadanCDF:
    def test_matches_lognormal_cdf(self):
        chain = make_bs_chain(strikes=np.linspace(60.0, 160.0, 100))
        F, T = chain.forward, chain.maturity
        x = np.linspace(70.0, 140.0, 30)
        est = carr_madan_cdf(chain, x)
        srt = SIGMA * np.sqrt(T)
        truth = norm.cdf((np.log(x / F) + 0.5 * srt**2) / srt)
        # the put/call wing stitch at the forward costs ~2e-2 locally
        assert np.max(np.abs(est - truth)) < 2.5e-2
        away_from_atm = np.abs(x - F) > 10
        assert np.max(np.abs(est[away_from_atm] - truth[away_from_atm])) < 5e-3

    def test_nan_when_one_wing_missing(self):
        chain = make_bs_chain(strikes=np.array([105.0, 110.0, 120.0]))  # calls only OTM
        est = carr_madan_cdf(chain, np.linspace(80, 130, 10))
        assert np.all(np.isnan(est))
