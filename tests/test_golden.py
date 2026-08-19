"""Golden tests: Python vs the Matlab reference implementation.

Fixture files in ``tests/golden/`` are generated once by running
``matlab/export_goldens.m`` in Matlab. Each output carries its own
relative tolerance. Cases are skipped (not failed) while a fixture file
has not been generated yet.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from rnproj import OptionChain, carr_madan, carr_madan_cdf, carr_madan_sparse, implied_cdf, project
from rnproj.distribution import implied_pdf

GOLDEN_DIR = Path(__file__).parent / "golden"


def load(case):
    path = GOLDEN_DIR / f"{case}.json"
    if not path.exists():
        pytest.skip(f"golden fixture {case}.json not generated yet (run matlab/export_goldens.m)")
    with open(path) as f:
        return json.load(f)


def check(name, actual, expected_spec):
    expected = np.asarray(expected_spec["value"], dtype=float)
    rtol = float(expected_spec["rtol"])
    np.testing.assert_allclose(
        np.asarray(actual, dtype=float),
        expected,
        rtol=max(rtol, 1e-16),
        atol=rtol * max(1.0, float(np.max(np.abs(expected)))) * 1e-3,
        err_msg=f"golden mismatch for {name}",
    )


def chain_from_inputs(inp):
    return OptionChain.from_arrays(
        np.asarray(inp["strikes"]),
        calls=np.asarray(inp["call_prices"]),
        puts=np.asarray(inp["put_prices"]),
        forward=inp["forward"],
        maturity=inp["maturity"],
        rate=inp["rate"],
    )


class TestBSDense:
    def test_projection_outputs(self):
        gold = load("bs_dense")
        inp, out = gold["inputs"], gold["outputs"]
        chain = chain_from_inputs(inp)
        grid = np.asarray(inp["grid"])
        w = np.asarray(inp["weights"])
        F = inp["forward"]

        targets = {
            "E_S2": lambda s: s**2,
            "E_S3": lambda s: s**3,
            "E_S4": lambda s: s**4,
            "E_neglogSF": lambda s: np.log(F / s),
            "E_SF2": lambda s: (s / F) ** 2,
        }
        for name, g in targets.items():
            fit = project(g, chain, grid=grid, weights=w)
            check(name, fit.value, out[name])

        fit = project(lambda s: s**2, chain, grid=grid, weights=w)
        check("beta_S2", fit.beta, out["beta_S2"])

        x_cdf = np.asarray(out["x_cdf"])
        fit = project(lambda s: (s[:, None] <= x_cdf[None, :]).astype(float),
                      chain, grid=grid, weights=w)
        check("cdf_raw", fit.value, out["cdf_raw"])

    def test_pdf_closed_form(self):
        gold = load("bs_dense")
        inp, out = gold["inputs"], gold["outputs"]
        chain = chain_from_inputs(inp)
        grid = np.asarray(inp["grid"])
        w = np.asarray(inp["weights"])
        idx = np.asarray(out["pdf_index_1based"], dtype=int) - 1
        pdf = implied_pdf(chain, x=grid[idx], grid=grid, weights=w)
        check("pdf_values", pdf.values, out["pdf_values"])


class TestBSSparse:
    def test_sparse_projection(self):
        gold = load("bs_sparse")
        inp, out = gold["inputs"], gold["outputs"]
        chain = chain_from_inputs(inp)
        grid = np.asarray(inp["grid"])
        w = np.asarray(inp["weights"])
        F = inp["forward"]

        fit = project(lambda s: s**2, chain, grid=grid, weights=w, otm_only=False)
        check("E_S2", fit.value, out["E_S2"])
        check("beta_S2", fit.beta, out["beta_S2"])
        fit = project(lambda s: np.log(F / s), chain, grid=grid, weights=w, otm_only=False)
        check("E_neglogSF", fit.value, out["E_neglogSF"])


class TestVGFixedParams:
    def test_grid_rules_and_density(self):
        from rnproj._vg import VGParams, VGQuantiles, vg_price_density
        from rnproj.grids import default_grid

        gold = load("vg_fixed_params")
        inp, out = gold["inputs"], gold["outputs"]
        params = VGParams(inp["sigma"], inp["nu"], inp["theta"])
        F, T = inp["forward"], inp["maturity"]
        strikes = np.concatenate([inp["put_strikes"], inp["call_strikes"]])

        check("omega", params.omega, out["omega"])
        q = VGQuantiles(params, T, F, mixture="trapz")
        grid = default_grid(strikes, distribution=q)
        check("grid_min", grid[0], out["grid_min"])
        check("grid_max", grid[-1], out["grid_max"])
        assert grid.size == int(out["n_grid"]["value"])

        idx = np.asarray(out["pdf_index_1based"], dtype=int) - 1
        pdf = vg_price_density(params, T, F, grid[idx], mixture="trapz")
        check("pdf_values", pdf, out["pdf_values"])


class TestCarrMadan:
    def test_pricing_and_cdf(self):
        gold = load("carr_madan")
        inp, out = gold["inputs"], gold["outputs"]
        chain = chain_from_inputs(inp)

        two = lambda s: 2.0 * np.ones_like(s)  # noqa: E731
        sq = lambda s: s**2  # noqa: E731
        check("cm_S2_trapz", carr_madan(sq, two, chain), out["cm_S2_trapz"])
        check("cm_S2_simpson", carr_madan(sq, two, chain, method="simpson"),
              out["cm_S2_simpson"])
        check("cm_logS", carr_madan(np.log, lambda s: -1.0 / s**2, chain), out["cm_logS"])
        check("cm_S2_sparse", carr_madan_sparse(sq, two, chain), out["cm_S2_sparse"])

        x_cdf = np.asarray(out["x_cdf"])
        check("cm_cdf", carr_madan_cdf(chain, x_cdf), out["cm_cdf"])


class TestFXQuotes:
    def test_strikes_and_prices(self):
        from rnproj.fx import chain_from_fx_quotes

        gold = load("fx_quotes")
        inp, out = gold["inputs"], gold["outputs"]
        chain = chain_from_fx_quotes(
            atm=inp["atm"], rr25=inp["rr25"], bf25=inp["bf25"],
            rr10=inp["rr10"], bf10=inp["bf10"],
            forward=inp["forward"], maturity=inp["maturity"],
            domestic_df=inp["domestic_df"], foreign_df=inp["foreign_df"],
        )
        check("put_strikes", chain.put_strikes, out["put_strikes"])
        check("call_strikes", chain.call_strikes, out["call_strikes"])
        check("put_prices", chain.put_prices, out["put_prices"])
        check("call_prices", chain.call_prices, out["call_prices"])


class TestFXBivariate:
    def make_triangle_and_grids(self, inp):
        from rnproj import FXTriangle

        def chain(K, C, P, forward):
            K, C, P = (np.asarray(a) for a in (K, C, P))
            return OptionChain(
                put_strikes=K[:3], put_prices=P[:3],
                call_strikes=K[3:], call_prices=C[3:],
                forward=forward, maturity=inp["maturity"], rate=0.0,
            )

        tri = FXTriangle(
            leg1=chain(inp["K1"], inp["C1"], inp["P1"], inp["F1"]),
            leg2=chain(inp["K2"], inp["C2"], inp["P2"], inp["F2"]),
            cross=chain(inp["K3"], inp["C3"], inp["P3"], inp["F1"] / inp["F2"]),
        )
        n = int(inp["n_grid"])
        K1, K2 = np.asarray(inp["K1"]), np.asarray(inp["K2"])
        grids = (
            np.linspace(0.95 * K1.min(), 1.02 * K1.max(), n),
            np.linspace(0.95 * K2.min(), 1.02 * K2.max(), n),
        )
        return tri, grids

    def test_covariance_and_tail(self):
        from rnproj import implied_covariance, joint_tail_probability

        gold = load("fx_bivariate")
        inp, out = gold["inputs"], gold["outputs"]
        tri, grids = self.make_triangle_and_grids(inp)

        cov = implied_covariance(tri, grids=grids)
        check("cov_hat", cov.covariance, out["cov_hat"])
        check("corr_hat", cov.correlation, out["corr_hat"])
        check("var1", cov.variance1, out["var1"])
        check("var2", cov.variance2, out["var2"])

        tail = joint_tail_probability(tri, inp["a1"], inp["a2"], grids=grids)
        check("tail_risk", tail.joint, out["tail_risk"])
        check("marginal1", tail.marginal1, out["marginal1"])
        check("marginal2", tail.marginal2, out["marginal2"])

    def test_hoeffding_cells(self):
        from rnproj import hoeffding_decomposition

        gold = load("fx_bivariate")
        inp, out = gold["inputs"], gold["outputs"]
        tri, grids = self.make_triangle_and_grids(inp)
        res = hoeffding_decomposition(tri, n_edges=4, grids=grids)
        check("hoeffding_total", res.total, out["hoeffding_total"])
        # Matlab flattens column-major
        check("hoeffding_cells", res.cells.flatten(order="F"), out["hoeffding_cells"])


class TestConstrainedCDF:
    def test_sequential_constrained(self):
        gold = load("cdf_constrained")
        inp, out = gold["inputs"], gold["outputs"]
        chain = chain_from_inputs(inp)
        grid = np.asarray(inp["grid"])
        cdf = implied_cdf(
            chain,
            x=grid,
            grid=grid,
            weights=np.ones_like(grid),
            monotone="constrained",
        )
        expected = np.asarray(out["cdf"]["value"], dtype=float)
        # Matlab's ols_pricing_cdf_discrete initializes cdf(1) = 0 and never
        # solves the first point; compare from the second point on, with an
        # absolute floor for the near-zero left tail (different QP solvers).
        # The Matlab lsqlin interior-point solver stops at slightly different
        # points than exact least squares in the flat far-left tail, which
        # changes where the monotonicity constraint binds (plateau vs. slow
        # rise); the gap also varies across BLAS builds (~5e-4 observed on
        # macOS arm64). Assert the structure (monotone, in range) and
        # CDF-scale closeness rather than solver-noise-level agreement.
        assert np.all(np.diff(cdf.values) >= -1e-12)
        assert cdf.values.min() >= 0.0 and cdf.values.max() <= 1.0
        assert cdf.values[0] <= max(expected[1], 1e-3)
        np.testing.assert_allclose(cdf.values[1:], expected[1:], rtol=1e-3, atol=1.5e-3)
