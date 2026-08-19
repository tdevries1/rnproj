# rnproj

**Risk-neutral expectations from option prices by least-squares projection.**

`rnproj` implements the projection estimator of De Vries (2026), *A
Projection Approach for Estimating Risk-Neutral Expectations*. It is a
drop-in replacement for hand-rolled Carr-Madan code (implied moments,
VIX/SVIX, implied distributions), markedly more accurate on sparse strike
chains, and the only available estimator of bivariate risk-neutral FX
dependence from vanilla options.

## Install

```bash
pip install rnproj
```

## Quickstart

```python
import rnproj

chain = rnproj.OptionChain.from_arrays(
    strikes, calls=call_prices, puts=put_prices,
    forward=5321.4, maturity=30 / 365, rate=0.043,
)

m = rnproj.implied_moments(chain)          # BKM-style moments
rnproj.vix(chain), rnproj.svix(chain)      # variance indices
cdf = rnproj.implied_cdf(chain)            # implied distribution

fit = rnproj.expectation(lambda s: (s / chain.forward - 1) ** 2, chain)
fit.value          # the estimate E^Q[g(S_T)]
fit.portfolio()    # the replicating option portfolio
fit.residual_l2    # projection error, feeds the finite-sample bound
```

Continue with [Theory in five minutes](theory.md), or jump straight to the
executable notebooks in the repository's `examples/` folder.
