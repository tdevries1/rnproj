"""Implied CDF/PDF: sanity properties and lognormal analytics."""

import numpy as np
import pytest
from scipy.stats import norm

from conftest import make_bs_chain
from rnproj import implied_cdf, implied_pdf

SIGMA = 0.2
TRAP = getattr(np, "trapezoid", getattr(np, "trapz", None))


def lognormal_cdf(x, forward, sigma, maturity):
    srt = sigma * np.sqrt(maturity)
    return norm.cdf((np.log(x / forward) + 0.5 * srt**2) / srt)


class TestImpliedCDF:
    def test_matches_lognormal(self, bs_chain):
        F, T = bs_chain.forward, bs_chain.maturity
        cdf = implied_cdf(bs_chain)
        inner = (cdf.x > 70) & (cdf.x < 140)
        truth = lognormal_cdf(cdf.x[inner], F, SIGMA, T)
        assert np.max(np.abs(cdf.values[inner] - truth)) < 5e-3

    def test_rearranged_is_monotone_in_range(self, bs_sparse_chain):
        cdf = implied_cdf(bs_sparse_chain, otm_only=False)
        assert np.all(np.diff(cdf.values) >= 0)
        assert cdf.values.min() >= 0.0 and cdf.values.max() <= 1.0

    def test_constrained_is_monotone_in_range(self, bs_sparse_chain):
        cdf = implied_cdf(bs_sparse_chain, otm_only=False, monotone="constrained", n_points=60)
        assert np.all(np.diff(cdf.values) >= -1e-12)
        assert cdf.values.min() >= 0.0 and cdf.values.max() <= 1.0

    def test_limits(self, bs_chain):
        cdf = implied_cdf(bs_chain)
        assert cdf.values[0] < 5e-3
        assert cdf.values[-1] > 1 - 5e-3

    def test_callable_interpolation(self, bs_chain):
        cdf = implied_cdf(bs_chain)
        q = cdf(np.array([bs_chain.forward]))
        assert 0.4 < q[0] < 0.6

    def test_pricing_consistency(self, bs_chain):
        # Integrating a payoff against the implied distribution reproduces
        # (approximately) the direct projection estimate of its price.
        from rnproj import project

        cdf = implied_cdf(bs_chain, monotone=None, n_points=400)
        pdf_vals = np.gradient(cdf.values, cdf.x)
        e_indirect = TRAP(cdf.x**2 * pdf_vals, cdf.x)
        e_direct = project(lambda s: s**2, bs_chain).value
        assert e_indirect == pytest.approx(e_direct, rel=1e-2)


class TestImpliedPDF:
    def test_matches_lognormal(self, bs_chain):
        F, T = bs_chain.forward, bs_chain.maturity
        pdf = implied_pdf(bs_chain)
        srt = SIGMA * np.sqrt(T)
        truth = np.exp(
            -0.5 * ((np.log(pdf.x / F) + 0.5 * srt**2) / srt) ** 2
        ) / (pdf.x * srt * np.sqrt(2 * np.pi))
        inner = (pdf.x > 75) & (pdf.x < 135)
        scale = truth[inner].max()
        assert np.max(np.abs(pdf.values[inner] - truth[inner])) / scale < 2e-2

    def test_integrates_to_one(self, bs_chain):
        pdf = implied_pdf(bs_chain, n_points=800)
        assert TRAP(pdf.values, pdf.x) == pytest.approx(1.0, abs=2e-2)

    def test_mean_is_forward(self, bs_chain):
        pdf = implied_pdf(bs_chain, n_points=800)
        mean = TRAP(pdf.x * pdf.values, pdf.x)
        assert mean == pytest.approx(bs_chain.forward, rel=1e-2)


class TestSparseChainDistribution:
    def test_five_strike_cdf_reasonable(self):
        strikes = np.array([85.0, 93.0, 100.5, 108.0, 118.0])
        chain = make_bs_chain(strikes=strikes)
        cdf = implied_cdf(chain, otm_only=False)
        truth = lognormal_cdf(cdf.x, chain.forward, SIGMA, chain.maturity)
        # sparse strikes: looser tolerance, but shape must be right
        assert np.max(np.abs(cdf.values - truth)) < 5e-2
