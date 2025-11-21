
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid as cumtrapz

def safe_log(x, eps=1e-12):
    return np.log(np.clip(x, eps, None))

# Parameters
t = np.arange(0, 11, 1, dtype=float)
gammaBH_base = 0.08
P_cross = 1.0
r0 = np.log(2.0)  # since I0=0.5
tau = 5.0
desiredgammapeak = 0.25
k = desiredgammapeak * np.e

# Baseline
gammaBH = gammaBH_base * t
intgamma = cumtrapz(gammaBH * P_cross, t, initial=0.0)
rexact = r0 + 0.5 * gammaBH_base * t**2
rfromint = r0 + intgamma
ainfo = rexact / r0
# To avoid division by zero at t=0 for Hinfo, we can clip or handle it.
# rexact starts at r0 > 0, so it is fine.
Hinfo = gammaBH / rexact

# Effective equation of state
b = gammaBH_base
weffanalytic = np.full_like(t, np.nan)
mask_tpos = t > 0
# Formula: w_eff = -2/3 - (2/3) * (r / (b * t^2))
weffanalytic[mask_tpos] = -2/3 - (2/3) * (rexact[mask_tpos] / (b * t[mask_tpos]**2))

# Discrete derivative check
lnH = safe_log(Hinfo)[1:]
lna = safe_log(ainfo)[1:]
# Use gradient with respect to values of ln_a? 
# np.gradient assumes constant spacing if only one array provided.
# But ln_a is not constantly spaced.
# So we should do diff / diff.
dlnH = np.diff(lnH)
dlna = np.diff(lna)
# Avoid division by zero if dlna is 0 (should not be as a grows)
dlna = np.where(dlna==0, 1e-12, dlna)
dlnHdlna = dlnH / dlna
# Note: dlnHdlna will have length len(t)-2.
# The discrete check in the docx code used `np.gradient(lnH, ln_a)`.
# np.gradient with two arguments computes gradient of first with respect to second.
dlnHdlna_grad = np.gradient(lnH, lna)
weffdiscrete = -1 - (2/3) * dlnHdlna_grad

# Alternative history
gammaBHalt = k * (t/tau) * np.exp(-t/tau)
intgammaalt = cumtrapz(gammaBHalt * P_cross, t, initial=0.0)
ralt = r0 + intgammaalt
Hinfoalt = gammaBHalt / ralt

# Plots
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
axes[0,0].semilogy(t, ainfo, 'r-o', label='ainfo')
axes[0,0].set_title('Chain-Break Driven Expansion'); axes[0,0].legend()

axes[0,1].plot(t, Hinfo, 'k-o', label='Hinfo (baseline)')
axes[0,1].plot(t, Hinfoalt, 'C0-s', label='H_info (alt)')
axes[0,1].set_title('Informational Expansion Rate'); axes[0,1].legend()

axes[1,0].plot(t[mask_tpos], weffanalytic[mask_tpos], 'm-o', label='w_eff (analytic)')
# Plot discrete with offset to match time indices
axes[1,0].plot(t[1:], weffdiscrete, 'g--s', label='w_eff (discrete)')
axes[1,0].axhline(-1.0, color='gray', linestyle=':', linewidth=1)
axes[1,0].set_title('Effective Equation of State'); axes[1,0].legend()

axes[1,1].plot(t, gammaBH, 'C3-o', label='gammaBH (baseline)')
axes[1,1].plot(t, gammaBHalt, 'C1-s', label='gamma_BH (alt)')
axes[1,1].set_title('Chain-Break Histories'); axes[1,1].legend()

plt.tight_layout()
plt.savefig('fig7_2.pdf')
plt.show()
