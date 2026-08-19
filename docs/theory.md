# Theory in five minutes

## The estimator

Let \(F\) be the forward, \(R_f\) the gross risk-free rate, and
\(P(K_j), C(K_j)\) observed put and call prices for one maturity. The
projection estimator answers: *what is the best replication of the payoff
\(g(S_T)\) using only the traded instruments?*

The basis is the set of traded payoffs

$$
\mathcal{F} = \{\, 1,\; S_T,\; (K^P_j - S_T)^+,\; (S_T - K^C_j)^+ \,\},
$$

whose risk-neutral expectations are known from market prices:
\(E^Q[1] = 1\), \(E^Q[S_T] = F\), \(E^Q[(K-S_T)^+] = R_f P(K)\),
\(E^Q[(S_T-K)^+] = R_f C(K)\).

Project \(g\) onto \(\operatorname{span}(\mathcal{F})\) in
\(L^2(\omega)\) for a weighting density \(\omega\): on a state grid
\(s_1, \dots, s_n\), solve the weighted least squares problem

$$
\hat\beta = \arg\min_\beta \sum_i \omega(s_i)\,\big(\varphi(s_i)'\beta - g(s_i)\big)^2,
$$

and price the fitted combination:

$$
\widehat{E^Q[g]} = \hat\beta_1 + \hat\beta_2 F
  + R_f \Big( \textstyle\sum_j \hat\beta^P_j P(K_j) + \sum_j \hat\beta^C_j C(K_j) \Big).
$$

Because \(1\) and \(S_T\) are in the basis, anything affine is priced
exactly; because each option payoff is in the basis, each observed option
is priced exactly; put-call parity is automatic.

## The error bound

The estimation error is controlled by a finite-sample inequality
(Proposition 2 of the paper):

$$
\big| E^Q[g] - E^Q[\hat g] \big| \;\le\;
  \|g - \hat g\|_{L^2(\omega)} \cdot \sqrt{\chi^2\!\big(f^Q \,\|\, \omega\big)} .
$$

The first factor is the projection residual — **computable from the data**
(`fit.residual_l2`). The second is the chi-squared divergence between the
unknown risk-neutral density and the weighting density; it is finite
whenever \(\omega\) has tails at least as heavy as \(f^Q\). The bound holds
simultaneously for *every* risk-neutral measure within that divergence, so
it also quantifies how much prices pin down the expectation
(`fit.error_bound(chi2)`).

## Relation to Carr-Madan

With a dense, regular strike grid the projection weights converge to the
Carr-Madan quadrature weights \(h\,g''(K_j)\) (Proposition 4), so the two
estimators agree asymptotically. In finite samples they differ exactly
where it matters: with few strikes, truncated ranges, or noisy quotes, the
Carr-Madan discretization error grows while the projection continues to
price the best available replication.

## The state grid

The grid is a numerical device, not a modeling choice — the WLS sum is a
Riemann approximation of the \(L^2(\omega)\) inner product, so the mesh
cancels from \(\hat\beta\). Two safety rules make any grid valid, and
`rnproj` enforces both automatically:

1. **Coverage** — the grid extends beyond every strike (with padding).
2. **Resolution** — the mesh is finer than half the smallest strike gap;
   otherwise two adjacent hinge payoffs are affine on every grid point they
   share and the design matrix loses full column rank.

See `rnproj.grids.default_grid`; pass `grid=` to override.
