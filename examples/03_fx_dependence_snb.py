# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # FX dependence from a triangle of option chains
#
# The unique feature of the projection approach: **risk-neutral dependence
# between two currencies from vanilla options alone**. For a triangle such
# as EURUSD, GBPUSD, and the cross rate EURGBP, triangular parity makes
# cross-rate options informative about the *joint* distribution: the
# numeraire-adjusted payoff $S_2 (S_1/S_2 - K)^+ = (S_1 - K S_2)^+$ is a
# two-asset payoff priced by an observed one-asset option.
#
# The paper's application: around the SNB's EUR/CHF floor events (Sep 2011
# introduction, Jan 2015 removal), roughly two thirds of the change in
# joint crash risk came from the **dependence channel**, not the marginals.
# Here we reproduce the mechanics on synthetic quotes with a "floor on" vs
# "floor off" dependence regime.

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

import rnproj
from rnproj import FXTriangle, OptionChain

F1, F2 = 1.10, 1.30  # EURUSD, GBPUSD forwards (USD numeraire)
SIG1, SIG2 = 0.10, 0.08
T = 1 / 12


# %% [markdown]
# ## Synthetic triangle under a joint-lognormal Q
#
# Legs are Black-priced; the cross chain is priced with Margrabe's exchange
# option formula, so the triangle is internally consistent for any
# log-correlation $\rho$. With real data you would instead build each chain
# from Bloomberg-style quotes via `rnproj.fx.chain_from_fx_quotes`.

# %%
def leg_chain(forward, sigma):
    z = np.array([-1.6, -0.8, 0.0, 0.8, 1.6])  # 5-strike OTC-style layout
    strikes = forward * np.exp(z * sigma * np.sqrt(T))
    df, srt = 1.0, sigma * np.sqrt(T)
    d1 = np.log(forward / strikes) / srt + 0.5 * srt
    calls = df * (forward * norm.cdf(d1) - strikes * norm.cdf(d1 - srt))
    puts = calls - (forward - strikes)
    return OptionChain(
        put_strikes=strikes[:3], put_prices=puts[:3],
        call_strikes=strikes[3:], call_prices=calls[3:],
        forward=forward, maturity=T, rate=0.0,
    )


def make_triangle(rho):
    leg1, leg2 = leg_chain(F1, SIG1), leg_chain(F2, SIG2)
    f3 = F1 / F2
    sx = np.sqrt(SIG1**2 + SIG2**2 - 2 * rho * SIG1 * SIG2)
    z = np.array([-1.6, -0.8, 0.0, 0.8, 1.6])
    k3 = f3 * np.exp(z * max(sx, 0.02) * np.sqrt(T))
    srt = sx * np.sqrt(T)
    d1 = (np.log(F1 / (k3 * F2)) + 0.5 * srt**2) / srt
    calls = (F1 * norm.cdf(d1) - k3 * F2 * norm.cdf(d1 - srt)) / F2
    puts = calls - (F1 - k3 * F2) / F2
    cross = OptionChain(
        put_strikes=k3[:3], put_prices=puts[:3],
        call_strikes=k3[3:], call_prices=calls[3:],
        forward=f3, maturity=T, rate=0.0,
    )
    return FXTriangle(leg1=leg1, leg2=leg2, cross=cross)


# %% [markdown]
# ## Implied correlation across dependence regimes

# %%
rhos = np.linspace(-0.6, 0.9, 7)
est = [rnproj.implied_covariance(make_triangle(r), n_grid=200).correlation for r in rhos]

plt.figure(figsize=(5, 3.5))
plt.plot(rhos, est, "o-", label="projection estimate")
plt.plot(rhos, rhos, "k--", lw=0.8, label="45°")
plt.xlabel(r"true $\rho$")
plt.ylabel("implied correlation")
plt.legend()
plt.tight_layout()

# %% [markdown]
# ## Joint crash risk: marginal channel vs dependence channel
#
# $Q(S_1/F_1 \le 0.97,\ S_2/F_2 \le 0.97)$ decomposed against the
# independence benchmark $Q_1 \cdot Q_2$. An SNB-floor-removal-style event
# is a jump from low to high dependence with roughly unchanged marginals:
# the whole change in joint risk is the dependence channel.

# %%
a = 0.97
low, high = 0.28, 0.87  # the paper's estimated correlations around the 2015 removal
res_low = rnproj.joint_tail_probability(make_triangle(low), a, a, n_grid=200)
res_high = rnproj.joint_tail_probability(make_triangle(high), a, a, n_grid=200)

for name, res in [("floor on (rho=0.28)", res_low), ("floor off (rho=0.87)", res_high)]:
    print(f"{name:22s} joint {max(res.joint, 0):.4f}  "
          f"independent {res.independent:.4f}  "
          f"dependence wedge {max(res.joint, 0) - res.independent:+.4f}")

delta_joint = max(res_high.joint, 0) - max(res_low.joint, 0)
delta_marg = res_high.independent - res_low.independent
print(f"\nchange in joint crash risk: {delta_joint:+.4f}")
print(f"  marginal channel:         {delta_marg:+.4f}")
print(f"  dependence channel:       {delta_joint - delta_marg:+.4f} "
      f"({100 * (delta_joint - delta_marg) / delta_joint:.0f}% of the change)")

# %% [markdown]
# ## Where does the dependence live? (Hoeffding decomposition)
#
# Hoeffding's identity localizes the covariance across the joint
# distribution: each cell's contribution to
# $\iint (F_{12} - F_1 F_2)\,dx\,dy$.

# %%
res = rnproj.hoeffding_decomposition(make_triangle(high), n_edges=5, n_grid=200)

plt.figure(figsize=(4.6, 3.8))
plt.pcolormesh(res.x_edges, res.y_edges, res.share, cmap="RdBu_r",
               vmin=-np.abs(res.share).max(), vmax=np.abs(res.share).max())
plt.colorbar(label="share of covariance")
plt.xlabel("$S_1/F_1$")
plt.ylabel("$S_2/F_2$")
plt.title("cellwise covariance contributions")
plt.tight_layout()
print(f"total (return-space, truncated box): {res.total:.6f}")
