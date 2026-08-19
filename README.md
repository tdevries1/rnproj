# rnproj

**Risk-neutral expectations from option prices by least-squares projection.**

`rnproj` implements the projection estimator of [De Vries (2026),
*Recovering Risk-Neutral Moments from Options*](https://arxiv.org/abs/2601.14852):
instead of discretizing the Carr-Madan spanning integral, the target payoff
`g(S_T)` is projected onto the span of the observed option payoffs by
weighted least squares, and the projection is priced with the observed
option prices:

```
E^Q[g(S_T)]  ≈  β₁ + β₂·F + R_f·( Σⱼ βⱼᴾ·P(Kⱼ) + Σⱼ βⱼᶜ·C(Kⱼ) )
```

where β solves an ordinary weighted least-squares problem on a state grid.

**Why use it instead of Carr-Madan?**

- **Automatically computes**: implied moments (BKM),
  VIX/SVIX-style indices, implied distributions.
- **Markedly more accurate with sparse or truncated strikes**, where the
  discretized CM formula worsens (order-of-magnitude smaller errors
  in the paper's simulations). With dense strikes the two coincide.
- Exact by construction for anything the options span (put-call parity,
  affine payoffs, observed option payoffs).
- Returns an **investable replicating portfolio**: the coefficients are
  position sizes in the quoted options.
- Comes with a **finite-sample error bound**:
  `|E^Q g − E^Q ĝ| ≤ ‖g − ĝ‖_{L²(ω)} · √χ²(f^Q‖ω)`, whose first factor is
  computed from the data (`fit.residual_l2`).
- **Bivariate FX dependence**: risk-neutral
  covariance, correlation, and joint crash probabilities for a triangle of
  exchange rates, using cross-rate options via triangular parity.

## Installation

```bash
pip install rnproj
```

Requires Python ≥ 3.10; depends only on numpy and scipy.

## Quickstart

```python
import rnproj

chain = rnproj.OptionChain.from_arrays(
    strikes, calls=call_prices, puts=put_prices,
    forward=5321.4, maturity=30 / 365, rate=0.043,
)

m = rnproj.implied_moments(chain)      # BKM-style log-return moments
print(m.variance, m.skewness, m.kurtosis)
print(rnproj.vix(chain), rnproj.svix(chain))

cdf = rnproj.implied_cdf(chain)        # option-implied distribution
pdf = rnproj.implied_pdf(chain)

# any payoff, one call; automatic state grid and weighting density
fit = rnproj.expectation(lambda s: (s / chain.forward - 1) ** 2, chain)
fit.value          # the estimate
fit.portfolio()    # the replicating option portfolio
fit.residual_l2    # projection error (the error-bound ingredient)
```

The state grid and the weighting density ω (a Variance-Gamma density
calibrated to the smile, with a lognormal fallback) are chosen
automatically; both can be overridden:

```python
fit = rnproj.project(g, chain, weights=my_density_callable)      # custom ω
fit = rnproj.project(g, chain, grid=my_grid, weights=my_weights) # full control
fit = rnproj.project(g, chain, otm_only=False)                   # sparse OTC chains
```

For chains quoted OTC-style (ATM/risk-reversal/butterfly vols), build the
five-strike chain directly:

```python
chain = rnproj.fx.chain_from_fx_quotes(
    atm=0.093, rr25=-0.012, bf25=0.0031, rr10=-0.022, bf10=0.0102,
    forward=1.0834, maturity=1 / 12, domestic_df=0.996, foreign_df=0.998,
)
```

## FX dependence (bivariate)

For two currencies against a common numeraire plus their cross rate,
vanilla options on all three legs estimate risk-neutral dependence:

```python
tri = rnproj.FXTriangle(leg1=eurusd_chain, leg2=gbpusd_chain, cross=eurgbp_chain)

rnproj.implied_covariance(tri).correlation      # risk-neutral correlation
res = rnproj.joint_tail_probability(tri, 0.97, 0.97)
res.joint, res.independent                       # dependence vs marginal channel
rnproj.hoeffding_decomposition(tri)              # where the covariance lives
```

This is the machinery behind the paper's SNB floor event study, where about
two thirds of the change in joint crash risk around both floor events came
from the dependence channel.

## Examples

Executable notebooks in [`examples/`](examples/):

1. [Quickstart: implied moments, VIX/SVIX, implied distribution](examples/01_quickstart_implied_moments.ipynb)
2. [Projection vs Carr-Madan with sparse strikes](examples/02_vix_svix_vs_carr_madan.ipynb)
3. [FX dependence and an SNB-style event study](examples/03_fx_dependence_snb.ipynb)

## Citing

If you use this package in academic work, please cite the paper:

> De Vries, Tjeerd (2026). *Recovering Risk-Neutral Moments from Options*.
> arXiv:2601.14852.

```bibtex
@misc{devries2026recovering,
  author        = {De Vries, Tjeerd},
  title         = {Recovering Risk-Neutral Moments from Options},
  year          = {2026},
  eprint        = {2601.14852},
  archivePrefix = {arXiv},
  primaryClass  = {q-fin.GN},
  url           = {https://arxiv.org/abs/2601.14852}
}
```

## License

MIT
