"""Bivariate FX projection against an analytic bivariate-lognormal model.

Under Q (numeraire currency), (S1, S2) are joint lognormal with means
(F1, F2), vols (sigma1, sigma2) and log-correlation rho. Then:

- vanilla options on each leg have Black prices;
- the numeraire-adjusted cross payoff S2*(S1/S2 - K)^+ = (S1 - K*S2)^+ is
  an exchange option with Margrabe's formula, giving the cross chain's
  observed price via  C3 = E^Q[(S1 - K S2)^+] / (F2 * Rf3);
- Cov(S1, S2) = F1 F2 (exp(rho s1 s2 T) - 1) analytically;
- the joint crash probability is a bivariate normal CDF.
"""

import numpy as np
import pytest
from scipy.stats import multivariate_normal, norm

from conftest import black_prices
from rnproj import (
    FXTriangle,
    OptionChain,
    hoeffding_decomposition,
    implied_covariance,
    joint_tail_probability,
)

F1, F2 = 1.10, 1.30
SIG1, SIG2 = 0.10, 0.08
T = 1.0 / 12.0
N_GRID = 150


def margrabe_undiscounted(k):
    """E^Q[(S1 - k*S2)^+] for joint lognormal (S1, S2), rho given globally."""
    sx = np.sqrt(SIG1**2 + SIG2**2 - 2 * RHO * SIG1 * SIG2)
    srt = sx * np.sqrt(T)
    d1 = (np.log(F1 / (k * F2)) + 0.5 * srt**2) / srt
    return F1 * norm.cdf(d1) - k * F2 * norm.cdf(d1 - srt)


def leg_chain(forward, sigma):
    z = np.array([-1.6, -0.8, 0.0, 0.8, 1.6])
    strikes = forward * np.exp(z * sigma * np.sqrt(T))
    calls, puts = black_prices(forward, strikes, T, 0.0, sigma)
    return OptionChain(
        put_strikes=strikes[:3],
        put_prices=puts[:3],
        call_strikes=strikes[3:],
        call_prices=calls[3:],
        forward=forward,
        maturity=T,
        rate=0.0,
    )


def make_triangle(rho):
    global RHO
    RHO = rho
    leg1, leg2 = leg_chain(F1, SIG1), leg_chain(F2, SIG2)
    f3 = F1 / F2
    sx = np.sqrt(SIG1**2 + SIG2**2 - 2 * rho * SIG1 * SIG2)
    z = np.array([-1.6, -0.8, 0.0, 0.8, 1.6])
    k3 = f3 * np.exp(z * max(sx, 0.02) * np.sqrt(T))
    call_prices = np.array([margrabe_undiscounted(k) for k in k3]) / F2  # Rf3 = 1
    # parity in numeraire-adjusted units: E^Q[S2 (S3-K)^+] - E^Q[S2 (K-S3)^+]
    #   = E^Q[S1 - K S2] = F1 - K F2  =>  C3 - P3 = (F1 - K F2)/F2
    put_prices = call_prices - (F1 - k3 * F2) / F2
    cross = OptionChain(
        put_strikes=k3[:3],
        put_prices=put_prices[:3],
        call_strikes=k3[3:],
        call_prices=call_prices[3:],
        forward=f3,
        maturity=T,
        rate=0.0,
    )
    return FXTriangle(leg1=leg1, leg2=leg2, cross=cross)


def true_cov(rho):
    return F1 * F2 * np.expm1(rho * SIG1 * SIG2 * T)


def true_joint_tail(rho, a1, a2):
    z1 = (np.log(a1) + 0.5 * SIG1**2 * T) / (SIG1 * np.sqrt(T))
    z2 = (np.log(a2) + 0.5 * SIG2**2 * T) / (SIG2 * np.sqrt(T))
    return multivariate_normal(cov=[[1, rho], [rho, 1]]).cdf([z1, z2])


class TestExactness:
    """Payoffs inside the joint basis span must be priced exactly."""

    def test_cross_rate_leg_priced_exactly(self):
        from rnproj import joint_projection

        tri = make_triangle(0.6)
        k = tri.cross.call_strikes[0]
        fit = joint_projection(
            lambda s1, s2: s2 * np.maximum(s1 / s2 - k, 0.0), tri, n_grid=N_GRID
        )
        truth = tri.leg2.forward * tri.cross.gross_rate * tri.cross.call_prices[0]
        assert fit.value == pytest.approx(truth, rel=1e-10)

    def test_marginal_leg_priced_exactly(self):
        from rnproj import joint_projection

        tri = make_triangle(0.6)
        k = tri.leg1.call_strikes[0]
        fit = joint_projection(lambda s1, s2: np.maximum(s1 - k, 0.0), tri, n_grid=N_GRID)
        assert fit.value == pytest.approx(
            tri.leg1.gross_rate * tri.leg1.call_prices[0], rel=1e-10
        )

    def test_forwards_priced_exactly(self):
        from rnproj import joint_projection

        tri = make_triangle(0.6)
        fit = joint_projection(lambda s1, s2: s1 + 0.0 * s2, tri, n_grid=N_GRID)
        assert fit.value == pytest.approx(tri.leg1.forward, rel=1e-10)
        fit = joint_projection(lambda s1, s2: s2 + 0.0 * s1, tri, n_grid=N_GRID)
        assert fit.value == pytest.approx(tri.leg2.forward, rel=1e-10)


class TestCovariance:
    def test_positive_dependence(self):
        # 5 strikes per leg: the paper's simulations find ~10-20% relative
        # accuracy for the covariance in this design.
        tri = make_triangle(0.6)
        res = implied_covariance(tri, n_grid=N_GRID)
        assert res.covariance == pytest.approx(true_cov(0.6), rel=0.20)
        assert 0.4 < res.correlation < 0.85

    def test_independence_gives_near_zero(self):
        tri = make_triangle(0.0)
        res = implied_covariance(tri, n_grid=N_GRID)
        assert abs(res.covariance) < 0.15 * abs(true_cov(0.6))

    def test_marginal_variances(self):
        tri = make_triangle(0.6)
        res = implied_covariance(tri, n_grid=N_GRID)
        assert res.variance1 == pytest.approx(F1**2 * np.expm1(SIG1**2 * T), rel=5e-2)
        assert res.variance2 == pytest.approx(F2**2 * np.expm1(SIG2**2 * T), rel=5e-2)

    def test_negative_dependence(self):
        tri = make_triangle(-0.5)
        res = implied_covariance(tri, n_grid=N_GRID)
        assert res.covariance < 0


class TestJointTail:
    def test_joint_crash_probability(self):
        tri = make_triangle(0.6)
        a = 0.97
        res = joint_tail_probability(tri, a, a, n_grid=N_GRID)
        assert res.joint == pytest.approx(true_joint_tail(0.6, a, a), abs=0.02)
        assert res.marginal1 == pytest.approx(
            norm.cdf((np.log(a) + 0.5 * SIG1**2 * T) / (SIG1 * np.sqrt(T))), abs=0.02
        )
        assert res.independent == pytest.approx(res.marginal1 * res.marginal2, rel=1e-12)

    def test_dependence_channel_sign(self):
        # Corner indicators are the hardest target for the finite basis and
        # raw estimates can fall below 0 (the paper's empirical code clamps
        # them); the ORDERING across dependence regimes is what must hold.
        a = 0.97
        res_dep = joint_tail_probability(make_triangle(0.6), a, a, n_grid=N_GRID)
        res_ind = joint_tail_probability(make_triangle(0.0), a, a, n_grid=N_GRID)
        assert res_dep.joint > res_ind.joint + 0.01
        assert res_dep.joint > res_dep.independent - 0.005


class TestHoeffding:
    def test_cells_sum_to_total(self):
        tri = make_triangle(0.6)
        res = hoeffding_decomposition(tri, n_edges=4, n_grid=N_GRID)
        assert res.cells.sum() == pytest.approx(res.total, rel=1e-12)
        assert res.cells.shape == (3, 3)
        assert res.share.sum() == pytest.approx(1.0, rel=1e-9)

    def test_total_sign_matches_dependence(self):
        assert hoeffding_decomposition(make_triangle(0.6), n_grid=N_GRID).total > 0
        assert hoeffding_decomposition(make_triangle(-0.5), n_grid=N_GRID).total < 0


class TestValidation:
    def test_maturity_mismatch_raises(self):
        tri = make_triangle(0.6)
        bad = OptionChain(
            put_strikes=tri.cross.put_strikes,
            put_prices=tri.cross.put_prices,
            call_strikes=tri.cross.call_strikes,
            call_prices=tri.cross.call_prices,
            forward=tri.cross.forward,
            maturity=0.5,
            rate=0.0,
        )
        with pytest.raises(ValueError, match="share one maturity"):
            FXTriangle(leg1=tri.leg1, leg2=tri.leg2, cross=bad)

    def test_cross_strikes_outside_grid_raises(self):
        tri = make_triangle(0.6)
        from rnproj import joint_projection

        g1 = np.linspace(1.05, 1.15, 200)
        g2 = np.linspace(1.25, 1.35, 200)
        bad_cross = OptionChain(
            put_strikes=np.array([0.5]),
            put_prices=np.array([0.001]),
            call_strikes=np.array([1.4]),
            call_prices=np.array([0.001]),
            forward=tri.cross.forward,
            maturity=T,
            rate=0.0,
        )
        bad_tri = FXTriangle(leg1=tri.leg1, leg2=tri.leg2, cross=bad_cross)
        with pytest.raises(ValueError, match="cross-rate strikes"):
            joint_projection(lambda a, b: a * b, bad_tri, grids=(g1, g2))
