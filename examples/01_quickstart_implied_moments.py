# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Quickstart: option-implied moments with `rnproj`
#
# `rnproj` estimates risk-neutral expectations $E^Q[g(S_T)]$ from an option
# chain by **projection**: the target payoff is regressed onto the span of
# the observed option payoffs $\{1, S_T, (K-S_T)^+, (S_T-K)^+\}$, and the
# fitted combination is priced with the observed option prices. This is the
# estimator of De Vries (2026), *A Projection Approach for Estimating
# Risk-Neutral Expectations*, and it is a drop-in replacement for
# Carr-Madan-style spanning integrals (BKM moments, VIX/SVIX, tail
# probabilities).
#
# This notebook uses a synthetic chain so it runs anywhere; replace the
# synthetic prices with your own quotes (e.g. from OptionMetrics).

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

import rnproj

# %% [markdown]
# ## Build an option chain
#
# A chain is one maturity of quotes. The **forward is mandatory**: carry
# and dividends live in the forward, and `rnproj` never infers it from the
# spot. Here we simulate an SPX-style chain from a Black-Scholes model with
# 20% volatility.

# %%
forward, maturity, rate, sigma = 5300.0, 30 / 365, 0.043, 0.20

strikes = np.linspace(4200.0, 6200.0, 41)
df = np.exp(-rate * maturity)
srt = sigma * np.sqrt(maturity)
d1 = np.log(forward / strikes) / srt + 0.5 * srt
calls = df * (forward * norm.cdf(d1) - strikes * norm.cdf(d1 - srt))
puts = df * (strikes * norm.cdf(-(d1 - srt)) - forward * norm.cdf(-d1))

chain = rnproj.OptionChain.from_arrays(
    strikes, calls=calls, puts=puts, forward=forward, maturity=maturity, rate=rate
)
chain.n_options

# %% [markdown]
# ## Any risk-neutral expectation in one call
#
# `rnproj.expectation` picks the state grid and the weighting density
# automatically (a Variance-Gamma density calibrated to the smile, with a
# lognormal fallback). The result carries the estimate, the replicating
# portfolio, and diagnostics.

# %%
fit = rnproj.expectation(lambda s: (s / forward - 1.0) ** 2, chain)
print("E^Q[(S_T/F - 1)^2] =", fit.value)
print("truth under BS     =", np.expm1(sigma**2 * maturity))
print("projection error ||g - g_hat|| =", fit.residual_l2)

# %% [markdown]
# The projection coefficients are an **investable portfolio** of the quoted
# options replicating the target payoff:

# %%
port = fit.portfolio()  # pandas DataFrame if pandas is installed, else dict
if isinstance(port, dict):
    for i in range(8):
        print(f"{port['instrument'][i]:<11} K={port['strike'][i]:9.2f}  "
              f"weight={port['weight'][i]:+.5f}")
else:
    print(port.head(8))

# %% [markdown]
# ## Implied moments, VIX and SVIX

# %%
m = rnproj.implied_moments(chain)  # log-return moments, BKM convention
print(f"variance  {m.variance:.6f}   (BS truth {sigma**2 * maturity:.6f})")
print(f"skewness  {m.skewness:+.4f}   (BS truth 0)")
print(f"kurtosis  {m.kurtosis:.4f}   (BS truth 3)")
print(f"VIX       {100 * rnproj.vix(chain):.2f}   (BS truth {100 * sigma:.2f})")
print(f"SVIX      {100 * rnproj.svix(chain):.2f}")

# %% [markdown]
# ## The option-implied distribution
#
# Projecting indicator payoffs $1\{S_T \le x\}$ gives the risk-neutral CDF
# (one least-squares solve for every $x$ at once); a closed form gives the
# density.

# %%
cdf = rnproj.implied_cdf(chain)
pdf = rnproj.implied_pdf(chain)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
axes[0].plot(cdf.x, cdf.values)
axes[0].set_title("implied CDF")
axes[1].plot(pdf.x, pdf.values)
axes[1].set_title("implied PDF")
for ax in axes:
    ax.axvline(forward, color="gray", lw=0.5)
    ax.set_xlabel("$S_T$")
plt.tight_layout()

# %% [markdown]
# ## Why not Carr-Madan?
#
# With dense strikes the two approaches agree (the paper proves the
# equivalence asymptotically). The projection's advantage appears with
# **sparse or truncated strikes** -- see the second example notebook.

# %%
cm = rnproj.carr_madan(lambda s: s**2, lambda s: 2.0 * np.ones_like(s), chain)
proj = rnproj.expectation(lambda s: s**2, chain)
truth = forward**2 * np.exp(sigma**2 * maturity)
print(f"E^Q[S^2]: projection error {abs(proj.value - truth):.4f}, "
      f"Carr-Madan error {abs(cm - truth):.4f}")
