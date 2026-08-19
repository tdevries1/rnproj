# FX dependence

The bivariate module estimates **risk-neutral dependence between two
currencies from vanilla options alone** — the paper's novel contribution
and the machinery behind its SNB floor event study.

## Setup

Take two rates against a common numeraire, say EURUSD (\(S_1\)) and GBPUSD
(\(S_2\)), plus the cross rate EURGBP (\(S_1/S_2\)). By triangular parity,
options on the cross rate are joint payoffs of \((S_1, S_2)\): with a
change of numeraire,

$$
E^Q\!\big[S_{2,T}\,(S_{1,T}/S_{2,T} - K)^+\big] = F_2 \, R_f^{(3)} \, C_3(K),
$$

where \(C_3\) is the observed cross-rate call. The joint basis therefore
contains the two marginal option families *plus* the numeraire-adjusted
cross-rate payoffs, and any joint target \(g(s_1, s_2)\) can be projected
onto it — same estimator, two dimensions.

## Usage

```python
import rnproj
from rnproj import FXTriangle

tri = FXTriangle(leg1=eurusd_chain, leg2=gbpusd_chain, cross=eurgbp_chain)

# covariance / correlation
cov = rnproj.implied_covariance(tri)
cov.covariance, cov.correlation

# joint crash probability and its decomposition
res = rnproj.joint_tail_probability(tri, 0.97, 0.97)
res.joint          # Q(S1/F1 <= 0.97, S2/F2 <= 0.97)
res.independent    # marginal-channel benchmark Q1 * Q2
res.joint - res.independent   # the dependence channel ("wedge")

# localize the covariance across the joint distribution
h = rnproj.hoeffding_decomposition(tri)
h.cells, h.share, h.total

# any joint payoff
fit = rnproj.joint_projection(lambda s1, s2: (s1 - s2) ** 2, tri)
```

Chains from OTC quote conventions (ATM / risk reversal / butterfly) are
built with `rnproj.fx.chain_from_fx_quotes`; see the API reference.

!!! warning "Sparse chains"
    OTC FX chains have five strikes, and the delta-neutral ATM strike lies
    *above* the forward. The bivariate machinery therefore uses all quotes
    without an OTM filter, matching the paper's empirical implementation.
    Raw joint tail estimates can fall slightly below 0 for extreme corner
    events; clamp at 0 as the paper does.
