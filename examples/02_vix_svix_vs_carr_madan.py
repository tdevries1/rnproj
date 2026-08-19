# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Projection vs Carr-Madan with sparse strikes
#
# The Carr-Madan spanning integral $E^Q[g] = g(F) + R_f \int g''(K)\,
# O(K)\,dK$ must be discretized on the observed strikes. With few strikes
# the discretization and truncation errors dominate. The projection
# estimator prices the *best least-squares replication* of the payoff
# instead, and is exact for anything in the option span by construction.
#
# This is Monte Carlo evidence in the style of the paper's Section 5:
# random strike subsets of a Black-Scholes chain, comparing relative errors
# for SVIX ($E^Q[R^2]$-based) and VIX ($E^Q[\log R]$-based).

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

import rnproj

rng = np.random.default_rng(42)
forward, maturity, rate, sigma = 100.0, 1 / 12, 0.02, 0.20
truth_vix = sigma
truth_svix = np.sqrt(np.expm1(sigma**2 * maturity) / maturity)


def bs_chain(strikes):
    df = np.exp(-rate * maturity)
    srt = sigma * np.sqrt(maturity)
    d1 = np.log(forward / strikes) / srt + 0.5 * srt
    calls = df * (forward * norm.cdf(d1) - strikes * norm.cdf(d1 - srt))
    puts = df * (strikes * norm.cdf(-(d1 - srt)) - forward * norm.cdf(-d1))
    return rnproj.OptionChain.from_arrays(
        strikes, calls=calls, puts=puts, forward=forward, maturity=maturity, rate=rate
    )


# %% [markdown]
# ## Experiment: random sparse strike sets

# %%
n_strike_grid = [5, 8, 12, 20, 30]
n_mc = 60
err = {("proj", "vix"): [], ("proj", "svix"): [], ("cm", "vix"): [], ("cm", "svix"): []}

for n_k in n_strike_grid:
    e = {k: [] for k in err}
    for _ in range(n_mc):
        strikes = np.sort(rng.uniform(75.0, 130.0, n_k))
        chain = bs_chain(strikes)
        try:
            vix_p = rnproj.vix(chain, otm_only=False)
            svix_p = rnproj.svix(chain, otm_only=False)
            vix_cm = np.sqrt(max(
                2 / maturity * (np.log(np.exp(rate * maturity))
                                - (rnproj.carr_madan(np.log, lambda s: -1 / s**2, chain)
                                   - np.log(forward * np.exp(-rate * maturity)))), 0))
            e2_cm = rnproj.carr_madan(
                lambda s: (s / forward) ** 2,
                lambda s: 2 / forward**2 * np.ones_like(s), chain)
            svix_cm = np.sqrt(max((e2_cm - 1.0) / maturity, 0))
        except (ValueError, np.linalg.LinAlgError):
            continue
        e[("proj", "vix")].append(abs(vix_p - truth_vix) / truth_vix)
        e[("proj", "svix")].append(abs(svix_p - truth_svix) / truth_svix)
        e[("cm", "vix")].append(abs(vix_cm - truth_vix) / truth_vix)
        e[("cm", "svix")].append(abs(svix_cm - truth_svix) / truth_svix)
    for k in err:
        err[k].append(np.nanmean(e[k]) if e[k] else np.nan)

# %% [markdown]
# ## Results
#
# Mean relative error by number of strikes (log scale). The projection is
# typically an order of magnitude more accurate on sparse chains, and the
# two methods converge as strikes become dense -- the paper's Proposition
# on Carr-Madan equivalence.

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for ax, measure in zip(axes, ["svix", "vix"]):
    ax.semilogy(n_strike_grid, err[("proj", measure)], "o-", label="projection")
    ax.semilogy(n_strike_grid, err[("cm", measure)], "s--", label="Carr-Madan")
    ax.set_title(measure.upper())
    ax.set_xlabel("number of strikes")
    ax.legend()
axes[0].set_ylabel("mean relative error")
plt.tight_layout()

# %%
print("mean relative errors with 5 strikes:")
print(f"  SVIX  projection {err[('proj', 'svix')][0]:.2e}   CM {err[('cm', 'svix')][0]:.2e}")
print(f"  VIX   projection {err[('proj', 'vix')][0]:.2e}   CM {err[('cm', 'vix')][0]:.2e}")
