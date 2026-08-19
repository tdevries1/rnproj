"""Implied moments and variance indices against Black-Scholes analytics."""

import numpy as np
import pytest

from rnproj import implied_moments, svix, vix

SIGMA = 0.2


class TestIndices:
    def test_vix_matches_bs(self, bs_chain):
        # Under BS, (2/T) E[log(F/S_T)] = sigma^2 exactly.
        assert vix(bs_chain) == pytest.approx(SIGMA, rel=1e-4)

    def test_svix_matches_bs(self, bs_chain):
        # Under BS, E[(S_T/F)^2] = exp(sigma^2 T).
        T = bs_chain.maturity
        expected = np.sqrt(np.expm1(SIGMA**2 * T) / T)
        assert svix(bs_chain) == pytest.approx(expected, rel=1e-4)

    def test_sparse_chain_indices_close(self, bs_sparse_chain):
        assert vix(bs_sparse_chain, otm_only=False) == pytest.approx(SIGMA, rel=5e-3)
        T = bs_sparse_chain.maturity
        expected = np.sqrt(np.expm1(SIGMA**2 * T) / T)
        assert svix(bs_sparse_chain, otm_only=False) == pytest.approx(expected, rel=5e-3)


class TestImpliedMoments:
    def test_log_moments_bs(self, bs_chain):
        # log(S_T/F) ~ N(-sigma^2 T/2, sigma^2 T)
        T = bs_chain.maturity
        m = implied_moments(bs_chain)
        assert m.mean == pytest.approx(-0.5 * SIGMA**2 * T, rel=1e-3)
        assert m.variance == pytest.approx(SIGMA**2 * T, rel=1e-3)
        assert m.skewness == pytest.approx(0.0, abs=2e-2)
        assert m.kurtosis == pytest.approx(3.0, rel=2e-2)

    def test_simple_moments_mean_zero(self, bs_chain):
        m = implied_moments(bs_chain, of="simple")
        # E[S_T/F - 1] = 0 exactly by construction (affine target).
        assert m.mean == pytest.approx(0.0, abs=1e-10)
        T = bs_chain.maturity
        assert m.variance == pytest.approx(np.expm1(SIGMA**2 * T), rel=1e-3)

    def test_price_moments(self, bs_chain):
        F, T = bs_chain.forward, bs_chain.maturity
        m = implied_moments(bs_chain, of="price", orders=2)
        assert m.raw[0] == pytest.approx(F, rel=1e-10)
        assert m.raw[1] == pytest.approx(F**2 * np.exp(SIGMA**2 * T), rel=1e-4)

    def test_central_helper_consistent(self, bs_chain):
        m = implied_moments(bs_chain)
        assert m.central(2) == pytest.approx(m.variance, rel=1e-12)

    def test_invalid_args(self, bs_chain):
        with pytest.raises(ValueError, match="at least 2"):
            implied_moments(bs_chain, orders=1)
        with pytest.raises(ValueError, match="must be"):
            implied_moments(bs_chain, of="weird")
