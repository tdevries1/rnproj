"""Exactness invariants and core behavior of the Layer-1 projection."""

import numpy as np
import pytest

from conftest import make_bs_chain
from rnproj import OptionChain, project


class TestExactness:
    """Payoffs inside the basis span must be priced exactly."""

    def test_affine_payoff(self, bs_chain):
        a, b = 3.7, -0.21
        fit = project(lambda s: a + b * s, bs_chain)
        assert fit.value == pytest.approx(a + b * bs_chain.forward, rel=1e-10)

    def test_observed_call_priced_exactly(self, bs_chain):
        ch = bs_chain.otm()
        k = ch.call_strikes[3]
        price = ch.call_prices[3]
        fit = project(lambda s: np.maximum(s - k, 0.0), bs_chain)
        assert fit.value == pytest.approx(ch.gross_rate * price, rel=1e-9)

    def test_observed_put_priced_exactly(self, bs_chain):
        ch = bs_chain.otm()
        k = ch.put_strikes[2]
        price = ch.put_prices[2]
        fit = project(lambda s: np.maximum(k - s, 0.0), bs_chain)
        assert fit.value == pytest.approx(ch.gross_rate * price, rel=1e-9)

    def test_put_call_parity_recovered(self, bs_chain):
        # A call payoff at a put strike K is (K-s)^+ + s - K: in the span.
        ch = bs_chain.otm()
        k = ch.put_strikes[2]
        put_price = ch.put_prices[2]
        fit = project(lambda s: np.maximum(s - k, 0.0), bs_chain)
        parity = ch.gross_rate * put_price + ch.forward - k
        assert fit.value == pytest.approx(parity, rel=1e-9)

    def test_residual_zero_for_spanned_payoff(self, bs_chain):
        fit = project(lambda s: 2.0 + 0.5 * s, bs_chain)
        assert fit.residual_l2 == pytest.approx(0.0, abs=1e-8)


class TestAccuracy:
    def test_second_moment_bs(self, bs_chain):
        # E^Q[S^2] = F^2 exp(sigma^2 T) under Black-Scholes.
        F, T, sigma = bs_chain.forward, bs_chain.maturity, 0.2
        fit = project(lambda s: s**2, bs_chain)
        assert fit.value == pytest.approx(F**2 * np.exp(sigma**2 * T), rel=1e-5)

    def test_sparse_chain_second_moment(self, bs_sparse_chain):
        F, T, sigma = bs_sparse_chain.forward, bs_sparse_chain.maturity, 0.2
        fit = project(lambda s: s**2, bs_sparse_chain, otm_only=False)
        assert fit.value == pytest.approx(F**2 * np.exp(sigma**2 * T), rel=1e-3)

    def test_scale_invariance(self, bs_chain):
        c = 250.0
        scaled = OptionChain(
            call_strikes=bs_chain.call_strikes * c,
            call_prices=bs_chain.call_prices * c,
            put_strikes=bs_chain.put_strikes * c,
            put_prices=bs_chain.put_prices * c,
            forward=bs_chain.forward * c,
            maturity=bs_chain.maturity,
            rate=bs_chain.rate,
        )
        fit = project(lambda s: s**2, bs_chain)
        fit_scaled = project(lambda s: s**2, scaled)
        assert fit_scaled.value == pytest.approx(c**2 * fit.value, rel=1e-8)


class TestVectorized:
    def test_matrix_target_matches_columnwise(self, bs_chain):
        thresholds = np.array([85.0, 100.0, 115.0])
        fit = project(lambda s: (s[:, None] <= thresholds).astype(float), bs_chain)
        assert fit.value.shape == (3,)
        for j, x in enumerate(thresholds):
            single = project(lambda s, x=x: (s <= x).astype(float), bs_chain)
            assert fit.value[j] == pytest.approx(single.value, rel=1e-12)
        # CDF-like output should be increasing in the threshold
        assert np.all(np.diff(fit.value) > 0)


class TestAutoGrid:
    def test_auto_grid_covers_and_resolves(self, bs_sparse_chain):
        fit = project(lambda s: s**2, bs_sparse_chain, otm_only=False)
        strikes = bs_sparse_chain.strikes
        assert fit.grid[0] < strikes.min() and fit.grid[-1] > strikes.max()
        min_gap = np.min(np.diff(strikes))
        assert np.max(np.diff(fit.grid)) <= min_gap / 2 + 1e-12

    def test_full_rank_with_close_strikes(self):
        strikes = np.array([90.0, 99.9, 100.0, 100.1, 110.0])
        chain = make_bs_chain(strikes=strikes)
        fit = project(lambda s: s**2, chain, otm_only=False)
        assert np.isfinite(fit.cond)
        # every basis payoff must still be priced exactly (full column rank)
        k = 99.9
        idx = np.where(chain.call_strikes == k)[0][0]
        f2 = project(lambda s: np.maximum(s - k, 0.0), chain, otm_only=False)
        assert f2.value == pytest.approx(chain.gross_rate * chain.call_prices[idx], rel=1e-6)

    def test_coarse_user_grid_warns(self, bs_chain):
        grid = np.linspace(50.0, 160.0, 30)
        with pytest.warns(UserWarning, match="coarser"):
            project(lambda s: s**2, bs_chain, grid=grid, weights=np.ones(30))

    def test_grid_not_covering_strikes_raises(self, bs_chain):
        grid = np.linspace(80.0, 120.0, 5000)
        with pytest.raises(ValueError, match="does not cover"):
            project(lambda s: s**2, bs_chain, grid=grid, weights=np.ones(5000))


class TestArgumentHandling:
    def test_array_weights_require_grid(self, bs_chain):
        with pytest.raises(ValueError, match="requires an explicit"):
            project(lambda s: s**2, bs_chain, weights=np.ones(100))

    def test_callable_weights_on_auto_grid(self, bs_chain):
        fit = project(lambda s: s**2, bs_chain, weights=lambda s: np.ones_like(s))
        assert np.allclose(fit.weights, 1.0)

    def test_rate_discount_exclusive(self):
        with pytest.raises(ValueError, match="exactly one"):
            OptionChain.from_arrays(
                [100.0], calls=[1.0], forward=100.0, maturity=0.1, rate=0.02, discount=0.99
            )

    def test_nonpositive_weights_rejected(self, bs_chain):
        grid = np.linspace(50, 160, 5000)
        with pytest.raises(ValueError, match="strictly positive"):
            project(lambda s: s**2, bs_chain, grid=grid, weights=np.zeros(5000))


class TestConstraintsAndPortfolio:
    def test_nonnegative_fitted_payoff(self, bs_sparse_chain):
        F = bs_sparse_chain.forward
        fit = project(
            lambda s: (s - F) ** 2, bs_sparse_chain, otm_only=False, nonnegative=True
        )
        assert np.all(fit.fitted() >= -1e-7)

    def test_portfolio_weights_sum(self, bs_chain):
        fit = project(lambda s: s**2, bs_chain)
        port = fit.portfolio()
        weights = port["weight"] if isinstance(port, dict) else port["weight"].to_numpy()
        assert np.allclose(weights, fit.beta)

    def test_error_bound_scales(self, bs_sparse_chain):
        fit = project(lambda s: np.log(s), bs_sparse_chain, otm_only=False)
        assert fit.error_bound(4.0) == pytest.approx(2.0 * fit.residual_l2)
