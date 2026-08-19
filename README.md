# rnproj

**Risk-neutral expectations from option prices by least-squares projection.**

`rnproj` implements the projection estimator of De Vries (2026), *A Projection
Approach for Estimating Risk-Neutral Expectations*: instead of the Carr-Madan
spanning integral, the target payoff `g(S_T)` is projected onto the span of
the observed option payoffs by weighted least squares, and the projection is
priced with the observed option prices. The estimator

- is a **drop-in replacement** for hand-rolled Carr-Madan code (implied
  moments, VIX/SVIX-style indices, tail probabilities),
- is **markedly more accurate with sparse or truncated strikes**, where the
  discretized spanning integral degrades,
- returns an **investable replicating portfolio** (the projection
  coefficients are position sizes in the quoted options),
- comes with a **finite-sample error bound** from the paper.

## Installation

```bash
pip install rnproj
```

Requires Python ≥ 3.10, numpy, scipy.

## Quickstart

```python
import rnproj

chain = rnproj.OptionChain.from_arrays(
    strikes, calls=call_prices, puts=put_prices,
    forward=5321.4, maturity=30 / 365, rate=0.043,
)

# E^Q of any payoff, with automatic state grid and weighting density:
fit = rnproj.expectation(lambda s: (s / chain.forward - 1) ** 2, chain)
print(fit.value)         # the risk-neutral expectation
print(fit.portfolio())   # the replicating option portfolio
print(fit.residual_l2)   # projection error (finite-sample bound ingredient)
```

Implied moments, distribution estimation, the Carr-Madan benchmark, and the
bivariate FX dependence estimator are under active development toward v0.1.

## Citing

If you use this package in academic work, please cite:

> De Vries, T. (2026). *A Projection Approach for Estimating Risk-Neutral
> Expectations.* Working paper, HEC Paris.

## License

MIT
