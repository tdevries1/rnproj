# Migrating from Carr-Madan code

Most empirical pipelines compute option-implied quantities with some
variant of the discretized spanning integral

```python
# the usual hand-rolled pattern
e_g = g(F) + Rf * (np.trapz(d2g(Kp) * P, Kp) + np.trapz(d2g(Kc) * C, Kc))
```

Each such call maps to one `rnproj` call — without the second derivative,
the OTM bookkeeping, or the interpolation/extrapolation layer:

| You compute today | With rnproj |
| --- | --- |
| Bakshi-Kapadia-Madan (2003) implied variance / skew / kurtosis | `rnproj.implied_moments(chain)` |
| VIX-style index | `rnproj.vix(chain)` |
| SVIX (Martin 2017) | `rnproj.svix(chain)` |
| \(E^Q[g(S_T)]\) for any g | `rnproj.expectation(g, chain)` |
| Breeden-Litzenberger density | `rnproj.implied_pdf(chain)` |
| Risk-neutral CDF / tail probabilities | `rnproj.implied_cdf(chain)` |
| Interpolate the IV surface first | not needed - raw quotes go in directly |

Things to know when switching:

1. **The forward is explicit.** Build the chain with the true forward
   (from put-call parity or the futures curve); `rnproj` never assumes
   \(F = e^{rT} S_t\).
2. **No smile fitting.** Raw quotes enter directly; there is no
   interpolation step whose settings need documenting.
3. **OTM filtering is automatic** (`otm_only=True` splits at the forward).
   For sparse OTC chains where every quote matters, pass
   `otm_only=False`.
4. **The Carr-Madan benchmark is included** (`rnproj.carr_madan`,
   `rnproj.carr_madan_sparse`, `rnproj.carr_madan_cdf`) so you can report
   both estimators side by side.
5. **Diagnostics come with the estimate**: `fit.residual_l2` (how well the
   options span your payoff), `fit.cond` (design conditioning),
   `fit.portfolio()` (the actual replicating positions).
