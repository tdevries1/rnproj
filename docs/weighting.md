# Choosing the weighting density

The weighting density \(\omega\) is the norm in which the payoff is
approximated: the projection concentrates accuracy where \(\omega\) puts
mass. The error bound
\(\|g-\hat g\|_{L^2(\omega)} \sqrt{\chi^2(f^Q\|\omega)}\) makes the
trade-off explicit — the ideal \(\omega\) is close to the risk-neutral
density itself, with tails at least as heavy so the divergence stays
finite.

## The automatic default

`weights=None` (the default everywhere) uses:

1. **Variance-Gamma density calibrated to the smile**
   (`rnproj.vg_weights`): the VG family (Madan-Carr-Chang 1998) captures
   skew and semi-heavy tails with three parameters; the martingale
   compensator pins \(E^\omega[S_T] = F\). Calibration minimizes pricing
   MSE over the OTM quotes.
2. **Lognormal fallback** (`rnproj.lognormal_weights`) whenever VG
   degenerates: fewer than 4 OTM quotes, parameters at their bounds (a
   flat smile is the \(\nu \to 0\) limit of VG, so BS-like chains fall
   back *by design*), or \(T/\nu < 1/2\) where the VG density develops an
   integrable singularity at the mode. The fallback sigma is implied from
   the chain's own option prices.

Which branch ran is recorded in `spec.params["family"]` and
`spec.params["fallback"]`.

## Overriding

```python
# any density function
fit = rnproj.project(g, chain, weights=my_pdf_callable)

# an explicit array on an explicit grid
fit = rnproj.project(g, chain, grid=grid, weights=w)

# uniform weights (the domain choice becomes the prior)
fit = rnproj.project(g, chain, weights=lambda s: np.ones_like(s))
```

!!! note "Robustness"
    The weighting density affects the replicating portfolio only at second
    order (Proposition 4 of the paper); reasonable choices give very
    similar estimates. It matters most for payoffs with mass far from the
    money (tail probabilities, high moments).
