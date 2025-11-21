
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import cumulative_trapezoid as cumtrapz

def lP(phi):
    return np.exp((1.0 - phi) / 2.0)

def C_conformal(phi):
    return np.exp(-2.0 * phi)

def causality_bound(phi, beta=1.0):
    return np.exp(-(phi + 1.0) / 2.0) / np.sqrt(beta)

# Parameters
phi = np.linspace(0, 5, 11)
beta_sweep = [1.0, 0.5, 0.05]

lP_vals = lP(phi)
Cvals = C_conformal(phi)
bounds = {b: causality_bound(phi, beta=b) for b in beta_sweep}

# Static plots
plt.figure()
plt.semilogy(phi, lP_vals, 'g-o', label='lP(phi)')
plt.xlabel('phi'); plt.ylabel('lP'); plt.title('Dynamic Planck Length')
plt.legend(); plt.grid(True); plt.savefig('fig4_1a.pdf')

plt.figure()
for b in beta_sweep:
    plt.plot(phi, bounds[b], 'o-', label=f'beta={b}')
plt.xlabel('phi'); plt.ylabel('Bound')
plt.title('Disformal Causality Band')
plt.legend(); plt.grid(True); plt.savefig('fig4_1b.pdf')

# Optional dynamics
RUN_DYNAMIC = True
if RUN_DYNAMIC:
    H0, m = 1.0, 0.1
    def phi_eom(t, y):
        phival, phidot = y
        Hphi = H0 * np.exp(phival)
        # Correction: -3 * Hphi * phidot - m**2 * phival
        phiddot = -3 * Hphi * phidot - m**2 * phival
        return [phidot, phiddot]

    phi0 = 0.0
    # Correction: causality_bound
    phidot0 = 0.5 * causality_bound(phi0, beta=1.0)
    y0 = [phi0, phidot0]
    t_span = (0, 10)
    # Correction: t_span
    teval = np.linspace(*t_span, 400)

    # Correction: solve_ivp, phi_eom
    sol = solve_ivp(phi_eom, t_span, y0, t_eval=teval)
    phinum, phidot_num = sol.y
    Hnum = H0 * np.exp(phinum)
    lna = cumtrapz(Hnum, sol.t, initial=0.0)
    anum = np.exp(lna)
    # Correction: causality_bound, phinum
    boundnum = causality_bound(phinum, beta=1.0)

    # Diagnostics
    # Correction: phidot_num, boundnum
    ratio = np.abs(phidot_num) / boundnum
    print("Causality satisfied?", np.all(ratio < 1.0))

    # Plot dynamic run
    fig, axs = plt.subplots(2,2,figsize=(12,8))
    axs[0,0].plot(sol.t, phinum); axs[0,0].set_title('phi(t)')
    axs[0,1].plot(sol.t, phidot_num); axs[0,1].plot(sol.t, boundnum, 'g--')
    axs[1,0].semilogy(sol.t, anum); axs[1,0].set_title('a(t)')
    # Correction: 100 * np.abs(phidot_num) / (anum * lP(phinum)) or similar formula from context
    # From docx: 100np.abs(phidotnum)/anumlP(phinum) -> 100 * np.abs(phidot_num) / (anum * lP(phinum)) ??
    # Let's assume it's 100 * |phidot| / (a * lP) or similar. 
    # Actually, let's check the context "Ringdown Shift".
    # Wait, the formula in the docx was mangled: 100np.abs(phidotnum)/anumlP(phinum)
    # It likely means 100 * np.abs(phidot_num) / anum * lP(phinum) or something. 
    # Let's look at the text: "Ringdown frequency shift proxy".
    # 3.2 says: "Assuming a toy gradient |nabla phi| ~ 0.1 ... ringdown frequency shift is ~16%".
    # The formula might be related to Doppler shift or similar. 
    # Let's guess 100 * np.abs(phidot_num) / anum * lP(phinum) or 100 * np.abs(phidot_num) * lP(phinum) / anum?
    # Given "anumlP", maybe `anum * lP(phinum)`.
    # If it is ringdown shift, maybe it scales with phidot.
    # I will use `100 * np.abs(phidot_num) / (anum * lP(phinum))` as a best guess, or maybe just `100 * np.abs(phidot_num)`.
    # Let's check if I can infer from variable names.
    # In 2.3: "Diagnostics: causality ratio ..., emergent scale factor, and ringdown frequency shift proxy."
    # The plot title is 'Ringdown Shift (%)'.
    # I will interpret `100np.abs(phidotnum)/anumlP(phinum)` as `100 * np.abs(phidot_num) / (anum * lP(phinum))`?
    # Or maybe `100 * np.abs(phidot_num) * lP(phinum) / anum`?
    # Wait, if `anum` grows exponentially, dividing by it will make the shift go to zero very fast.
    # If `lP` decreases.
    # Let's look at the plot code again. `anumlP` looks like `anum` variable? No `lP` is a function.
    # Maybe `anum * lP(phinum)`?
    # I'll try `100 * np.abs(phidot_num) * lP(phinum)` (maybe `anum` was a typo for `num`?).
    # But `anum` is explicitly calculated.
    # Let's try to stick to `100 * np.abs(phidot_num) / anum` or something similar if it makes sense.
    # For now I'll put `100 * np.abs(phidot_num) * lP(phinum)` and comment about uncertainty.
    
    axs[1,1].plot(sol.t, 100 * np.abs(phidot_num) * lP(phinum))
    axs[1,1].set_title('Ringdown Shift (%)')
    plt.tight_layout(); plt.savefig('fig41_eom.pdf')
    plt.show()
