#!/usr/bin/env python3
"""
sim6_v16_omega.py

Expanded Omega Simulation Suite:
- Depletion cosmology (Omega Protocol) + Ringdown-shift pipeline
- Local Sim 5 sandboxes (emergent gravity, inertia-as-latency, BH shredding)
- Sim 3: Dynamic Planck Length and Disformal Causality Band

Features:
- Depletion cosmology with Omega Depletion Law (u = ln I).
- Explicit Omega objects: Chain Overlap Density (Φ̄), dynamic Planck length ℓP(Φ̄).
- "Distance = Correlation Deficit" metric d_info(z).
- BH horizon area proxy A_BH(z) from BHARD proxy.
- Maps A_BH → BH mass change → QNM f(t) drift (ringdown).
- Sim 5 v1: 1D Φ-field with emergent geometry, inertia-as-latency, local BH area growth
  from shredding, and Omega metabolic inequality diagnostics.
- Sim 5 v3.4-style: Emergent gravity + BH shredding visualization with GIF export.
- Sim 3: Dynamic Planck length, conformal factor, disformal causality band (β-sweep),
  and optional dynamical scalar-field evolution with causality diagnostics.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.interpolate import PchipInterpolator, interp1d
from scipy.signal import hilbert
from scipy.optimize import brentq
from numpy.fft import rfft
import warnings
from universe_lifecycle_model import UniverseLifecycleModel
warnings.filterwarnings("ignore", category=UserWarning)

# -------------------------
# Units & constants
# -------------------------
G = 6.67430e-11            # SI
c = 299792458.0            # m/s
M_sun = 1.98847e30         # kg
pc_in_m = 3.085677581e16
Mpc_in_km = 3.085677581e19 / 1e3
Gyr_to_sec = 3.15576e16
conv_1Gyr_to_km_s_Mpc = Mpc_in_km / Gyr_to_sec
c_km_s = 299792.458

# -------------------------
# Omega Protocol / informational-geometry constants
# -------------------------
hbar = 1.054571817e-34           # J*s
lP0_SI = np.sqrt(hbar * G / c**3)  # Planck length (meters)
phi_c_def = 0.1                  # critical COD scale in ℓ_P(Φ); tunable

def phi_bar_from_I(I, I_init):
    """
    Cosmic mean Chain Overlap Density Φ̄(t) from global information I(t).
    In Omega, Φ ∈ (0,1]. Normalise by initial information I_init.
    """
    I = np.asarray(I)
    return np.clip(I / I_init, 1e-12, 1.0)

def ellP_of_phi(phi_bar, phi_c=phi_c_def, lP0=lP0_SI):
    """
    Dynamic Planck length ℓ_P(Φ) = ℓ_{P0} exp((1 - Φ)/φ_c)
    (Omega Eq. 2.2), here applied to the cosmic mean Φ̄(t).
    """
    phi_bar = np.asarray(phi_bar)
    return lP0 * np.exp((1.0 - phi_bar) / phi_c)

# -------------------------
# Cosmology / depletion model defaults
# -------------------------
H0_fid = 73.5
H0_SI_default = H0_fid / conv_1Gyr_to_km_s_Mpc

OMEGA_M_FID = 0.315
OMEGA_GAMMA = 5.38e-5
OMEGA_NU = 3.0e-5
OMEGA_K = 0.0

alpha_def = 1.0
kappa_def = 1.0
gamma_guess_def = 1e-4
A0_def = 1.0

# BHARD table
z_bhard = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0])
bhard_normed = np.array([0.15, 0.7, 2.1, 4.0, 5.2, 4.1, 2.3, 0.8, 0.25, 0.08])
_bhard_interp = PchipInterpolator(z_bhard, bhard_normed, extrapolate=True)

def A_BH_of_z(z, A0=A0_def):
    scalar_input = np.ndim(z) == 0
    z = np.atleast_1d(z)
    z = np.maximum(z, 0.0)
    res = A0 * np.maximum(_bhard_interp(z), 0.0)
    if scalar_input:
        return float(res[0])
    return res

def A_BH_of_a(a, A0=A0_def):
    a = np.maximum(a, 1e-12)
    z = 1.0/a - 1.0
    return A_BH_of_z(z, A0=A0)

def Omega_rad_total():
    return OMEGA_GAMMA + OMEGA_NU

def H_matter_squared(a, params):
    H0_SI = params['H0_SI']
    Om = params.get('Omega_m', OMEGA_M_FID)
    Ok = params.get('Omega_k', OMEGA_K)
    Or = Omega_rad_total()
    return H0_SI**2 * (Om/a**3 + Or/a**4 + Ok/a**2)

# -------------------------
# ODE System (Depletion cosmology)
# -------------------------
def derivs(t, y, params):
    """
    System of ODEs for Depletion Cosmology (Omega Protocol).
    y[0] = u = ln(I)  (Information proxy)
    y[1] = a          (Scale factor)
    """
    u, a = float(y[0]), float(y[1])
    a = max(a, 1e-12)
    A = A_BH_of_a(a, A0=params['A0'])

    # Omega Depletion Law (Eq. 5.1):
    # dI/dt = -gamma * A^kappa * I  => du/dt = -gamma * A^kappa
    du_dt = -params['gamma'] * (A ** params['kappa'])

    # Omega Dark-Energy / Expansion Law (Eq. 5.2):
    # H_info(t) = alpha * |du/dt|
    H_info = params['alpha'] * abs(du_dt)

    H_m2 = H_matter_squared(a, params)
    H2 = H_m2 + H_info**2
    H = np.sqrt(max(H2, 0.0))
    
    da_dt = a * H
    return [du_dt, da_dt]

def integrate_cosmo(params, z_start=200.0, t_final=14.0):
    a_init = 1.0 / (1.0 + z_start)
    u_init = 0.0
    t_span = (1e-6, t_final)
    sol = solve_ivp(lambda tt, yy: derivs(tt, yy, params),
                    t_span, [u_init, a_init],
                    method='Radau', dense_output=True, atol=1e-10, rtol=1e-8, max_step=0.5)
    return sol

def cumtrapz_custom(y, x):
    y = np.asarray(y); x = np.asarray(x)
    n = len(y)
    out = np.zeros(n)
    if n < 2:
        return out
    dx = np.diff(x)
    avg = 0.5*(y[:-1] + y[1:])
    out[1:] = np.cumsum(avg * dx)
    return out

# -------------------------
# Observables
# -------------------------
def compute_observables_from_solution(sol, params, zmax=8.0, nz=2000):
    """
    Post-process to get observables, including Omega geometric variables (Φ̄, ℓP)
    and correlation distance d_info.
    """
    z_grid = np.linspace(0.0, zmax, nz)
    a_grid = 1.0/(1.0+z_grid)
    
    t_arr = sol.t
    u_arr = sol.y[0]
    a_arr = sol.y[1]
    
    # Sort and ensure monotonicity
    sidx = np.argsort(t_arr)
    t_arr = t_arr[sidx]; u_arr = u_arr[sidx]; a_arr = a_arr[sidx]
    if np.any(np.diff(a_arr) <= 0):
        a_arr = np.maximum.accumulate(a_arr + 1e-12*np.arange(len(a_arr)))
        
    t_of_a = PchipInterpolator(a_arr, t_arr, extrapolate=True)
    u_of_t = PchipInterpolator(t_arr, u_arr, extrapolate=True)
    
    t_grid = t_of_a(a_grid)
    u_grid = u_of_t(t_grid)
    du_dt = u_of_t.derivative(1)(t_grid)
    d2u_dt2 = u_of_t.derivative(2)(t_grid)
    
    # Reconstruct H and w_eff (Omega Eq 5.3)
    H_info = params['alpha'] * np.abs(du_dt)
    H_m2 = H_matter_squared(a_grid, params)
    H_vals = np.sqrt(np.maximum(H_m2 + H_info**2, 0.0))  # 1/Gyr
    H_km_s_Mpc = H_vals * conv_1Gyr_to_km_s_Mpc
    
    denom = du_dt**2 + 1e-30
    w_eff = -1.0 - (params['alpha']/3.0) * (d2u_dt2 / denom)
    
    # Standard FRW Distance
    H_safe = np.maximum(H_km_s_Mpc, 1e-5)
    chi = cumtrapz_custom(c_km_s / H_safe, z_grid)
    dL = (1+z_grid) * chi
    mu = 5.0 * np.log10(np.maximum(dL, 1e-8)) + 25.0
    
    # Omega Protocol / Informational Geometry
    I_grid = np.exp(u_grid)          # I(t) = e^u(t)
    I_init = I_grid[-1]              # High-z value
    
    phi_bar = phi_bar_from_I(I_grid, I_init)   # Cosmic mean Φ̄(t)
    ellP_bar = ellP_of_phi(phi_bar)           # ℓ_P(Φ̄)
    
    # Correlation deficit distance: d_info = -ℓ_P(Φ̄) * ln(I / I_init)
    corr_deficit = -np.log(np.maximum(I_grid / I_init, 1e-12))
    d_info_m = ellP_bar * corr_deficit
    d_info_Mpc = d_info_m / (pc_in_m * 1e6)

    # A_BH history mapped to a_grid
    A_grid = A_BH_of_a(a_grid, A0=params['A0'])
    
    return {
        'z': z_grid, 'a': a_grid, 't': t_grid, 'u': u_grid, 
        'I': I_grid, 
        'phi_bar': phi_bar, 
        'ellP_bar_m': ellP_bar, 
        'd_info_Mpc': d_info_Mpc,
        'H_km_s_Mpc': H_km_s_Mpc, 
        'w_eff': w_eff, 
        'chi_Mpc': chi, 'dL_Mpc': dL, 'mu': mu,
        'A_BH': A_grid
    }

# -------------------------
# Calibration routine
# -------------------------
def calibrate_gamma(params_template, z_start=200.0, t_final=14.0,
                    logg_min=-18, logg_max=-2, ngrid=20):
    H0_target = H0_fid

    def H0_for_logg(lg):
        p = params_template.copy()
        p['gamma'] = 10.0**lg
        sol = integrate_cosmo(p, z_start=z_start, t_final=t_final)
        try:
            if not sol.success:
                return np.nan
            if sol.y[1,-1] < 0.99:
                return 1e5
            A_today = A_BH_of_a(1.0, A0=p['A0'])
            du_dt_today = -p['gamma'] * (A_today ** p['kappa'])
            H_info = p['alpha'] * abs(du_dt_today)
            H_m2 = H_matter_squared(1.0, p)
            H_val = np.sqrt(H_m2 + H_info**2) * conv_1Gyr_to_km_s_Mpc
            return H_val
        except Exception:
            return np.nan

    grid = np.linspace(logg_min, logg_max, ngrid)
    vals = [H0_for_logg(g) for g in grid]
    finite = np.isfinite(vals)
    if not np.any(finite):
        raise RuntimeError("Calibration failed: no finite H0 in grid")
    idx = np.argmin(np.abs(np.array(vals) - H0_target))
    best = grid[idx]
    low = grid[max(idx-2, 0)]; high = grid[min(idx+2, len(grid)-1)]
    try:
        root = brentq(lambda x: H0_for_logg(x) - H0_target, low, high, xtol=1e-3)
        return 10.0**root
    except Exception:
        return 10.0**best

# -------------------------
# Ringdown mapping utilities
# -------------------------
def mass_from_area_fraction(M0_solar, a0, deltaA_over_A):
    # For fixed spin, A ∝ M^2 * const => M1 = M0 * sqrt(1 + delta)
    return M0_solar * np.sqrt(1.0 + deltaA_over_A)

f1, f2, f3 = 1.5251, -1.1568, 0.1292
def qnm_freq_hz(M_solar, a_dimless):
    pref = c**3 / (2*np.pi*G*(M_solar*M_sun))
    x = np.maximum(1e-8, 1.0 - a_dimless)
    return pref * (f1 + f2 * x**f3)

def map_A_history_to_qnm_freqs(A_rel, M0_solar=30.0, a0=0.7):
    """
    Map relative BH area history A_rel(t) into BH mass history and QNM frequency.
    In Omega: BH horizon area tracks sequestered Φ-gradient energy.
    """
    deltaA = A_rel - 1.0
    M_t = mass_from_area_fraction(M0_solar, a0, deltaA)
    f_t = qnm_freq_hz(M_t, a0)
    return f_t, M_t

def synth_damped_sinusoid(f_t, t, tau, amp=1.0):
    phi = 2*np.pi * cumtrapz_custom(f_t, t)
    h = amp * np.exp(-t/tau) * np.sin(phi)
    return h

def analytic_aligo_psd(f):
    f = np.asarray(f)
    f0 = 215.0; S0 = 1e-46
    psd = S0 * ((f/f0)**-4.14 + 5*(f/f0)**-0.69 + 1.0*(f/f0)**2)
    psd[f < 10.0] = 1e99
    return psd

def compute_snr_time_series(h, delta_t):
    N = len(h)
    hf = rfft(h)
    df = 1.0/(N*delta_t)
    f = np.arange(len(hf))*df
    Sn = analytic_aligo_psd(f)
    integrand = (np.abs(hf)**2) / Sn
    snr2 = 4.0 * np.sum(integrand) * df
    return np.sqrt(np.maximum(snr2, 0.0))

# -------------------------
# Sim 5 v1: local Ω-dynamics with BH area + metabolic inequality
# -------------------------
def run_sim5_dynamics():
    """
    Sim 5 (v1): 1D Φ-field dynamics with emergent geometry, inertia=latency,
    and a fixed black-hole horizon that shreds infalling "matter packets".

    Features:
    - Tracks a local BH area proxy A_BH_local(t), incremented by shredding events.
    - Tracks internal Ω-information I_internal(t) from the packet region.
    - Models environment information loss per frame from shredded mass.
    - Checks the Omega Metabolic Inequality:
          dI_internal/dt > |dI_env/dt|_shred
    and saves all histories to disk.
    """

    # Visual style
    plt.style.use('dark_background')
    plt.rcParams['axes.facecolor'] = '#0a0a0a'
    plt.rcParams['figure.facecolor'] = '#0a0a0a'

    # 1. Universe config
    N_REGIONS = 200         # number of Q-regions on the line
    DT = 0.1                # time step (arb. units)
    TOTAL_FRAMES = 300      # animation length

    # Local toy Omega constants
    PHI_VACUUM = 1.0
    PHI_CRITICAL = 0.1
    L_PLANCK_BASE = 1.0
    KAPPA_dyn = 5.0         # inertia as latency κ
    G_dyn = 0.05            # coupling from ∇Φ to acceleration (local, not Newton's G)

    # Info-per-mass and BH area scaling (toy)
    INFO_PER_MASS = 1.0
    deltaA_per_mass = 0.05

    # Scalar field Φ
    phi = np.ones(N_REGIONS) * PHI_VACUUM

    # 2. Matter + Black Hole
    particle_width = 12
    phi_particle_val = 0.4

    # Two initial packets
    positions = [float(N_REGIONS // 4), float(3 * N_REGIONS // 4)]
    velocities = [0.0, 0.0]

    bh_pos = N_REGIONS - 20
    bh_width = 5
    phi_bh_val = 0.1
    bh_start = bh_pos - bh_width
    bh_end = bh_pos + bh_width

    # Local BH area proxy
    A_BH_local = 1.0

    # 3. Helpers
    def get_emergent_geometry(phi_field):
        local_l_p = L_PLANCK_BASE * np.exp((PHI_VACUUM - phi_field) / PHI_CRITICAL)
        physical_x = np.cumsum(local_l_p)
        physical_x -= physical_x[0]
        return physical_x

    def get_mass_total(phi_field):
        mass_density = np.maximum(0, 1.0 - phi_field)
        return np.sum(mass_density)

    def is_shredded(start, end):
        return start < bh_end and end > bh_start

    # 4. Histories for Ω bookkeeping
    vel_history = []
    time_history = []
    shred_events = []

    I_internal_history = []
    dI_internal_dt_history = []
    dI_env_loss_dt_history = []
    metabolic_satisfied_history = []

    A_BH_local_history = []
    mass_shred_history = []

    # 5. Animation
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=False)
    fig.suptitle('Ω PROTOCOL: EMERGENT GRAVITY (Sim 5 v1)',
                 color='#00ffff', fontsize=18, fontweight='bold')

    def update(frame):
        nonlocal positions, velocities, A_BH_local

        # A. Reset field: vacuum + BH
        phi[:] = PHI_VACUUM
        phi[max(0, bh_start):min(N_REGIONS, bh_end)] = phi_bh_val

        # B. Physics update: packets, gravity, shredding
        active_indices = []
        current_shred = False
        mass_shred_frame = 0.0

        # Temp lists to hold next state
        next_positions = []
        next_velocities = []
        
        # Iterate and filter
        for i in range(len(positions)):
            current_int_pos = int(positions[i])
            start = current_int_pos - particle_width
            end = current_int_pos + particle_width
            start = max(0, start)
            end = min(N_REGIONS - 1, end)

            if is_shredded(start, end):
                shred_events.append(frame)
                current_shred = True

                overlap_start = max(start, bh_start)
                overlap_end = min(end, bh_end)
                overlap_len = max(0, overlap_end - overlap_start)
                mass_shred = (1.0 - phi_particle_val) * overlap_len

                A_BH_local += deltaA_per_mass * mass_shred
                mass_shred_frame += mass_shred
                continue # Do not add to next_positions

            # Apply matter density for surviving packets
            if start < end:
                phi[start:end] = phi_particle_val
                next_positions.append(positions[i])
                next_velocities.append(velocities[i])

        # Update globals
        positions[:] = next_positions
        velocities[:] = next_velocities

        # C. Local Ω-information bookkeeping
        phi_field = phi
        internal_mask = np.isclose(phi_field, phi_particle_val, atol=1e-3)

        if np.any(internal_mask):
            I_internal = np.sum(1.0 - phi_field[internal_mask])
        else:
            I_internal = 0.0
        I_internal_history.append(I_internal)

        I_env_loss = INFO_PER_MASS * mass_shred_frame
        dI_env_shred_dt = I_env_loss / DT
        dI_env_loss_dt_history.append(dI_env_shred_dt)

        if len(I_internal_history) >= 2:
            dI_int_dt = (I_internal_history[-1] - I_internal_history[-2]) / DT
        else:
            dI_int_dt = 0.0
        dI_internal_dt_history.append(dI_int_dt)

        metabolic_ok = dI_int_dt > dI_env_shred_dt
        metabolic_satisfied_history.append(metabolic_ok)

        A_BH_local_history.append(A_BH_local)
        mass_shred_history.append(mass_shred_frame)

        # D. Gradient-driven dynamics
        grad_phi = np.gradient(phi)
        num_particles = len(positions)

        if num_particles > 0:
            total_mass = get_mass_total(phi)
            if total_mass < 0.001:
                total_mass = 0.001
            mass_per_particle = total_mass / num_particles

            applied_force = 0.0

            for i in range(num_particles):
                pos = positions[i]
                grad_at_pos = np.interp(pos, np.arange(N_REGIONS), grad_phi)

                a_grav = -G_dyn * grad_at_pos
                a_ext = applied_force / (mass_per_particle + KAPPA_dyn)

                velocities[i] += (a_grav + a_ext) * DT
                positions[i] += velocities[i] * DT
        else:
            applied_force = 0.0

        # E. Emergent geometry
        physical_x = get_emergent_geometry(phi)

        # F. Visualization
        ax1.clear()
        ax1.set_title(
            f"Frame {frame} │ Particles: {len(positions)} │ "
            f"Shredded: {len(shred_events)} │ A_BH,loc={A_BH_local:.2f}",
            color='#00ffff', fontsize=12, pad=10
        )
        ax1.set_ylabel("Information Density Φ")
        ax1.set_xlabel("Emergent Distance (Spacetime)")

        ax1.plot(physical_x, phi, color='#00ff00', lw=2, label='Φ Field')
        ax1.fill_between(physical_x, phi, 1.0,
                         color='cyan', alpha=0.2, label='Mass (1 - Φ)')

        ax1.scatter(physical_x[::5],
                    np.ones_like(physical_x[::5]) * 0.5,
                    color='white', s=5, alpha=0.4)

        if 0 <= bh_start < len(physical_x) and 0 <= bh_end < len(physical_x):
            ax1.axvspan(physical_x[int(bh_start)], physical_x[int(bh_end)],
                        color='red', alpha=0.3, label='Event Horizon')

        for pos in positions:
            x_phys = np.interp(pos, np.arange(N_REGIONS), physical_x)
            ax1.scatter(x_phys, 0.4, color='yellow', s=150, marker='o',
                        edgecolor='orange', linewidth=2, zorder=10)

        if current_shred:
            ax1.text(0.5, 0.5, "SHREDDED", transform=ax1.transAxes,
                     color='red', fontsize=30,
                     ha='center', va='center',
                     alpha=1.0, fontweight='bold', zorder=20)

        status_text = ("Ω metabolic inequality: satisfied"
                       if metabolic_ok else
                       "Ω metabolic inequality: violated")
        status_color = 'cyan' if metabolic_ok else 'magenta'
        ax1.text(0.02, 0.9, status_text,
                 transform=ax1.transAxes,
                 color=status_color, fontsize=10,
                 ha='left', va='center', alpha=0.8)

        ax1.set_ylim(0.0, 1.1)
        ax1.legend(loc='upper right', facecolor='#000000')

        avg_vel = float(np.mean(velocities)) if velocities else 0.0
        vel_history.append(avg_vel)
        time_history.append(frame)

        ax2.clear()
        ax2.set_ylabel("Avg Velocity")
        ax2.set_xlabel("Time Step")
        ax2.plot(time_history, vel_history,
                 color='#ffff00', lw=2, label='Inertial Velocity')
        ax2.legend(loc='upper left', facecolor='#000000')
        ax2.grid(True, color='gray', linestyle='--', alpha=0.2)

    ani = FuncAnimation(fig, update, frames=TOTAL_FRAMES, interval=30)
    plt.tight_layout()
    
    # Explicitly save or run to ensure data collection in notebook environment
    try:
        ani.save('sim5_v1_dynamics.gif', writer='pillow', fps=30)
        print("Sim 5 v1: saved 'sim5_v1_dynamics.gif'")
    except Exception as e:
        print(f"Sim 5 v1: could not save GIF ({e}). Running manually for data.")
        for f in range(TOTAL_FRAMES):
            update(f)
            
    plt.show()

    # 6. Save diagnostics after animation
    t_arr = np.arange(len(I_internal_history)) * DT

    np.savez(
        "sim5_local_info_BH_depletion.npz",
        t=t_arr,
        A_BH_local=np.array(A_BH_local_history),
        I_internal=np.array(I_internal_history),
        dI_internal_dt=np.array(dI_internal_dt_history),
        dI_env_loss_dt=np.array(dI_env_loss_dt_history),
        mass_shred_per_frame=np.array(mass_shred_history),
        metabolic_ok=np.array(metabolic_satisfied_history),
    )
    print("Saved Sim 5 local Ω-information + BH area histories to sim5_local_info_BH_depletion.npz")

    plt.figure(figsize=(6, 4))
    plt.plot(t_arr, A_BH_local_history)
    plt.xlabel("t (arb. units)")
    plt.ylabel("A_BH,local (arb.)")
    plt.title("Sim 5 v1: Local BH area growth from shredding")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("sim5_local_A_BH_growth.png", dpi=200)

    plt.figure(figsize=(6, 4))
    plt.plot(t_arr, dI_internal_dt_history, label="dI_internal/dt")
    plt.plot(t_arr, dI_env_loss_dt_history, label="|dI_env/dt|_shred")
    plt.xlabel("t (arb. units)")
    plt.ylabel("Rate (arb.)")
    plt.title("Sim 5 v1: Ω metabolic inequality diagnostic")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("sim5_omega_metabolic_check.png", dpi=200)

    plt.close('all')

# -------------------------
# Sim 5 v3.4-style: emergent gravity + BH shredding + GIF
# -------------------------
def run_sim5_v34_dynamics():
    """
    Sim 5 (v3.4-style): emergent gravity & BH shredding visualization.

    - 1D Φ field with emergent geometry via ℓ_P(Φ) ~ exp((1-Φ)/φ_c).
    - Matter packets as regions of low Φ (phi_particle_val).
    - Fixed BH horizon as deep dip in Φ that shreds overlapping packets.
    - Tracks average velocity and (optionally) external force.
    - Saves an animated GIF if pillow is installed.
    """

    plt.style.use('dark_background')

    # 1. Universe config
    N_REGIONS = 200         # Number of Q-regions (nodes)
    DT = 0.1                # Time step magnitude
    TOTAL_FRAMES = 300      # Duration of simulation

    # Omega Theory Constants (local, to avoid clashing with cosmology)
    PHI_VACUUM = 1.0        # Maximum Overlap (Empty Space)
    PHI_CRITICAL = 0.1      # Sensitivity (Space warping factor)
    L_PLANCK_BASE = 1.0     # Base grid unit distance
    KAPPA_loc = 5.0         # NETWORK LATENCY (Informational Inertia)
    G_loc = 0.05            # Emergent gravity constant (tuned for visualization)

    # Initialize the scalar field phi (Information Density)
    phi = np.ones(N_REGIONS) * PHI_VACUUM

    # 2. Matter and Black Hole
    particle_width = 12
    phi_particle_val = 0.4 

    positions = [float(N_REGIONS // 4), float(3 * N_REGIONS // 4)]
    velocities = [0.0, 0.0]

    bh_pos = N_REGIONS - 20
    bh_width = 5
    phi_bh_val = 0.1
    bh_start = bh_pos - bh_width
    bh_end = bh_pos + bh_width

    # 3. Helper functions
    def get_emergent_geometry(phi_field):
        local_l_p = L_PLANCK_BASE * np.exp((PHI_VACUUM - phi_field) / PHI_CRITICAL)
        physical_x = np.cumsum(local_l_p)
        physical_x -= physical_x[0]
        return physical_x

    def get_mass_total(phi_field):
        mass_density = np.maximum(0, 1.0 - phi_field)
        return np.sum(mass_density)

    def is_shredded(start, end):
        return start < bh_end and end > bh_start

    # 4. Simulation setup
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
    fig.suptitle('Omega Protocol v3.4: Emergent Gravity & BH Shredding', color='white', fontsize=16)

    vel_history = []
    force_history = []
    time_history = []
    shred_events = []

    # 5. Main loop
    def update(frame):
        nonlocal positions, velocities

        # A. Reset field & BH
        phi[:] = PHI_VACUUM
        phi[max(0, bh_start):min(N_REGIONS, bh_end)] = phi_bh_val

        # B. Draw particles, check shredding
        active_particles = []
        
        # Temp lists
        next_positions = []
        next_velocities = []
        
        for i in range(len(positions)):
            current_int_pos = int(positions[i])
            start = current_int_pos - particle_width
            end = current_int_pos + particle_width

            if start < 0: start = 0
            if end >= N_REGIONS: end = N_REGIONS - 1

            if is_shredded(start, end):
                shred_events.append((frame, i))
                continue  # Shredded: do not redraw

            if start < end:
                phi[start:end] = phi_particle_val
            
            next_positions.append(positions[i])
            next_velocities.append(velocities[i])

        # Update globals
        positions[:] = next_positions
        velocities[:] = next_velocities

        # C. Gradients for gravity
        grad_phi = np.gradient(phi)

        # D. Physics: update each particle
        num_particles = len(positions)
        applied_force = 0.0  # No external force; gravity only

        if num_particles > 0:
            current_mass = get_mass_total(phi) / num_particles

            for i in range(num_particles):
                pos = positions[i]
                grad_at_pos = np.interp(pos, np.arange(N_REGIONS), grad_phi)

                # Emergent gravity: accelerate towards lower Φ
                a_grav = -G_loc * grad_at_pos
                a_ext = applied_force / (current_mass + KAPPA_loc)

                effective_accel = a_grav + a_ext

                velocities[i] += effective_accel * DT
                positions[i] += velocities[i] * DT

        # E. Update geometry
        physical_x = get_emergent_geometry(phi)

        # F. Visualization
        ax1.clear()
        ax1.set_title(f"Frame {frame}: Emergent Gravity (towards low Φ) & BH Shredding")
        ax1.set_ylabel("Information Density (Φ)")
        ax1.set_xlabel("Emergent Distance (grid stretching)")

        ax1.plot(physical_x, phi, color='#00ff00', lw=2, label='Chain Overlap Density Φ')
        ax1.fill_between(physical_x, phi, 1.0, color='cyan', alpha=0.3, label='Mass (Non-Overlap)')

        ax1.scatter(physical_x[::4], np.ones_like(physical_x[::4])*0.5,
                    color='white', s=10, alpha=0.6, label='Q-Regions')

        ax1.axvspan(physical_x[max(0, bh_start)], physical_x[min(N_REGIONS-1, bh_end)],
                    color='red', alpha=0.2, label='Black Hole Horizon')

        ax1.set_ylim(0.0, 1.1)
        ax1.legend(loc='lower right')

        avg_vel = np.mean(velocities) if velocities else 0.0
        vel_history.append(avg_vel)
        force_history.append(applied_force)
        time_history.append(frame)

        ax2.clear()
        ax2.set_title("Emergent Inertia & Dynamics")
        ax2.set_ylabel("Magnitude")
        ax2.set_xlabel("Time Step")

        ax2.plot(time_history, force_history, color='#ff4444', linestyle='--',
                 alpha=0.8, label='External Force')
        ax2.plot(time_history, vel_history, color='#ffff00', lw=2.5,
                 label='Avg Velocity')

        ax2.legend(loc='upper left')

        ax2.text(0, (max(vel_history) if vel_history else 0) + 0.05,
                 f"Particles: {num_particles}\n"
                 f"Latency (Kappa): {KAPPA_loc}\n"
                 f"Gravity G: {G_loc}\n"
                 f"Force: {applied_force:.1f}",
                 color='white', fontsize=9, verticalalignment='top')

        for event_frame, particle_id in shred_events:
            if event_frame == frame:
                ax2.text(frame, avg_vel, f"Shredded P{particle_id+1}!",
                         color='red', fontsize=10)

    ani = FuncAnimation(fig, update, frames=TOTAL_FRAMES, interval=20)

    # Save to GIF if pillow is available
    try:
        ani.save('omega_protocol_simulation.gif', writer='pillow', fps=30)
        print("Sim 5 v3.4-style: saved 'omega_protocol_simulation.gif'")
    except Exception:
        print("Sim 5 v3.4-style: could not save GIF (pillow missing?), showing animation only.")

    try:
        plt.show()
    except Exception:
        pass

# -------------------------
# Sim 3: Dynamic Planck length & disformal causality band
# -------------------------
def run_sim3_disformal(RUN_DYNAMIC=True):
    """
    Simulation 3: Dynamic Planck Length and Disformal Causality Band

    Static:
    - ℓ_P(φ) = exp((1 - φ)/2)
    - C(φ)   = exp(-2φ)
    - Disformal causality bound: |φ_dot| < exp(-(φ+1)/2) / sqrt(β)

    β-sweep: β in [1.0, 0.5, 0.05]

    Optional dynamics:
    - φ̈ + 3 H(φ) φ̇ + m^2 φ = 0, with H(φ) = H0 e^{φ}.
    - Initial conditions chosen within causality band.
    - Diagnostics: causality ratio, a(t), ringdown-shift proxy.
    """

    # Local helper functions for Sim 3
    def lP_phi(phi):
        return np.exp((1.0 - phi) / 2.0)

    def C_conformal(phi):
        return np.exp(-2.0 * phi)

    def causality_bound(phi, beta=1.0):
        return np.exp(-(phi + 1.0) / 2.0) / np.sqrt(beta)

    # Static relations
    phi_vals = np.linspace(0.0, 5.0, 200)
    beta_sweep = [1.0, 0.5, 0.05]

    lP_vals = lP_phi(phi_vals)
    C_vals = C_conformal(phi_vals)
    bounds = {b: causality_bound(phi_vals, beta=b) for b in beta_sweep}

    # Plot: Dynamic Planck length
    plt.figure(figsize=(6,4))
    plt.semilogy(phi_vals, lP_vals, 'g-', label=r'$\ell_P(\phi)$')
    plt.xlabel(r'$\phi$'); plt.ylabel(r'$\ell_P$')
    plt.title('Sim 3: Dynamic Planck Length')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('sim3_lP_dynamic.png', dpi=150)

    # Plot: Conformal factor
    plt.figure(figsize=(6,4))
    plt.semilogy(phi_vals, C_vals, 'b-', label=r'$C(\phi)$')
    plt.xlabel(r'$\phi$'); plt.ylabel(r'$C(\phi)$')
    plt.title('Sim 3: Conformal Factor')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('sim3_conformal_factor.png', dpi=150)

    # Plot: Disformal causality band
    plt.figure(figsize=(6,4))
    for b in beta_sweep:
        plt.plot(phi_vals, bounds[b], label=rf'$\beta={b}$')
    plt.xlabel(r'$\phi$'); plt.ylabel(r'Bound on $|\dot\phi|$')
    plt.title('Sim 3: Disformal Causality Band')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('sim3_causality_band.png', dpi=150)

    # Static ringdown-shift estimate at φ=0, |∇φ| ~ 0.1
    phi0 = 0.0
    grad_phi0 = 0.1
    lP0 = lP_phi(phi0)
    # Toy estimate: Δf/f ~ |∇φ| * ℓ_P
    ringdown_shift_static = grad_phi0 * lP0
    print(f"Sim 3: Static ringdown shift estimate at φ=0 is ~{100*ringdown_shift_static:.1f}%")

    # Optional dynamics
    if RUN_DYNAMIC:
        H0 = 1.0
        m = 0.1
        beta_dyn = 1.0

        def phi_eom(t, y):
            phival, phidot = y
            Hphi = H0 * np.exp(phival)
            phiddot = -3.0 * Hphi * phidot - m**2 * phival
            return [phidot, phiddot]

        phi0_dyn = 0.0
        phidot0_dyn = 0.5 * causality_bound(phi0_dyn, beta=beta_dyn)
        y0 = [phi0_dyn, phidot0_dyn]
        t_span = (0.0, 10.0)
        t_eval = np.linspace(t_span[0], t_span[1], 400)

        sol = solve_ivp(phi_eom, t_span, y0, t_eval=t_eval, method='RK45')
        if not sol.success:
            print("Sim 3: Dynamic integration failed.")
            return

        phi_num = sol.y[0]
        phidot_num = sol.y[1]
        H_num = H0 * np.exp(phi_num)
        ln_a = cumtrapz(H_num, sol.t, initial=0.0)
        a_num = np.exp(ln_a)
        bound_num = causality_bound(phi_num, beta=beta_dyn)

        ratio = np.abs(phidot_num) / np.maximum(bound_num, 1e-12)
        print("Sim 3: Causality satisfied in dynamic run?",
              bool(np.all(ratio < 1.0)))

        # Ringdown shift proxy: Δf/f ~ |φ̇| / (a ℓ_P(φ))
        lP_num = lP_phi(phi_num)
        ringdown_shift_dyn = 100.0 * np.abs(phidot_num) / np.maximum(a_num * lP_num, 1e-12)

        # Dynamic plots
        fig, axs = plt.subplots(2, 2, figsize=(10, 7))
        axs[0,0].plot(sol.t, phi_num)
        axs[0,0].set_title(r'$\phi(t)$')
        axs[0,0].set_xlabel('t'); axs[0,0].set_ylabel(r'$\phi$')
        axs[0,0].grid(True, alpha=0.3)

        axs[0,1].plot(sol.t, phidot_num, label=r'$\dot\phi$')
        axs[0,1].plot(sol.t, bound_num, 'g--', label='bound')
        axs[0,1].set_title(r'$\dot\phi(t)$ and Causality Bound')
        axs[0,1].set_xlabel('t'); axs[0,1].set_ylabel(r'$\dot\phi$')
        axs[0,1].legend(); axs[0,1].grid(True, alpha=0.3)

        axs[1,0].semilogy(sol.t, a_num)
        axs[1,0].set_title('Emergent Scale Factor a(t)')
        axs[1,0].set_xlabel('t'); axs[1,0].set_ylabel('a(t)')
        axs[1,0].grid(True, alpha=0.3)

        axs[1,1].plot(sol.t, ringdown_shift_dyn)
        axs[1,1].set_title('Ringdown Shift Proxy (%)')
        axs[1,1].set_xlabel('t'); axs[1,1].set_ylabel('%')
        axs[1,1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('sim3_dynamic_eom.png', dpi=150)
        plt.close(fig)

        np.savez(
            "sim3_dynamic_run.npz",
            t=sol.t,
            phi=phi_num,
            phidot=phidot_num,
            a=a_num,
            bound=bound_num,
            ringdown_shift_percent=ringdown_shift_dyn,
            causality_ratio=ratio,
        )
        print("Sim 3: Saved dynamic run outputs to sim3_dynamic_run.npz")


def run_universe_lifecycle_model():
    """
    Run the inception-to-heat-death lifecycle model and persist diagnostics.
    """
    lifecycle_model = UniverseLifecycleModel()
    lifecycle_history = lifecycle_model.simulate()
    checkpoints = lifecycle_model.epoch_report(lifecycle_history)
    lifecycle_model.plot_summary(lifecycle_history, save_path="sim6_universe_lifecycle_summary.png")

    np.savez("sim6_universe_lifecycle.npz", **lifecycle_history, **checkpoints)
    print("Saved sim6_universe_lifecycle.npz and sim6_universe_lifecycle_summary.png")

    print("Universe lifecycle checkpoints (Gyr):")
    for key, value in checkpoints.items():
        print(f"  {key}: {value:.6g}")

# -------------------------
# Main pipeline
# -------------------------
def main():
    # 0) Inception-to-death lifecycle model
    run_universe_lifecycle_model()

    # 1) Run cosmology
    params = {'alpha': alpha_def, 'kappa': kappa_def, 'gamma': gamma_guess_def,
              'Omega_m': OMEGA_M_FID, 'A0': A0_def, 'H0_SI': H0_SI_default}
    
    print("Calibrating gamma for target H0 = {:.2f} km/s/Mpc...".format(H0_fid))
    try:
        params['gamma'] = calibrate_gamma(params)
        print(" -> calibrated gamma = {:.3e}".format(params['gamma']))
    except Exception as e:
        print(f"Calibration failed: {e}. Using guess.")

    
    sol = integrate_cosmo(params)
    out = compute_observables_from_solution(sol, params, zmax=6.0, nz=2000)
    
    # Save cosmology outputs (includes new Omega d_info and phi_bar)
    np.savez('sim6_v16_omega_cosmo.npz', **out)
    print("Saved cosmology output to sim6_v16_omega_cosmo.npz")
    
    # 2) Extract A_BH(t) for Ringdown
    t_gyr = out['t']
    A_rel = out['A_BH'] / (params['A0'] if params['A0']!=0 else 1.0)
    
    # 3) Map to QNM frequency history
    choices = [{'label':'Stellar30', 'M0':30.0, 'a0':0.7, 'tau':0.03},
               {'label':'SMBH1e6', 'M0':1e6, 'a0':0.7, 'tau':100.0}]
    
    ring_t = np.linspace(0.0, 0.6, 16384)  # 0.6 s analysis window
    results_summary = []
    
    for ch in choices:
        M0 = ch['M0']; a0 = ch['a0']
        f_cosmo, M_t = map_A_history_to_qnm_freqs(A_rel, M0_solar=M0, a0=a0)
        
        idx_peak = np.argmax(A_rel)
        win = 40
        i0 = max(0, idx_peak - win//2); i1 = min(len(t_gyr)-1, idx_peak + win//2)
        sub_t = t_gyr[i0:i1+1]; sub_f = f_cosmo[i0:i1+1]
        
        if len(sub_t) < 2:
            f_ring = np.ones_like(ring_t) * f_cosmo[idx_peak]
        else:
            interp = interp1d(sub_t, sub_f, kind='cubic', fill_value='extrapolate')
            sub_span = sub_t[-1] - sub_t[0]
            if sub_span <= 0:
                f_ring = np.ones_like(ring_t) * sub_f[0]
            else:
                frac = np.linspace(0.0, 1.0, len(ring_t))
                t_mapped = sub_t[0] + frac * sub_span
                f_ring = interp(t_mapped)
        
        tau_seconds = 4.0 * G * (M0*M_sun) / (c**3)
        tau = max(1e-3, min(1.0, tau_seconds))
        
        h_ring = synth_damped_sinusoid(f_ring, ring_t, tau, amp=1.0)
        
        analytic_sig = hilbert(h_ring)
        inst_phase = np.unwrap(np.angle(analytic_sig))
        inst_freq = np.gradient(inst_phase, ring_t) / (2*np.pi)
        rel_shift = (inst_freq - inst_freq.mean())/inst_freq.mean()
        
        snr = compute_snr_time_series(h_ring, ring_t[1]-ring_t[0])
        
        prefix = f"sim6_v16_{ch['label']}"
        
        plt.figure(figsize=(10,8))
        plt.subplot(3,1,1)
        plt.plot(ring_t, f_ring)
        plt.title(f"Mapped QNM frequency f(t) for {ch['label']}")
        plt.ylabel("Hz"); plt.grid(True)
        plt.subplot(3,1,2)
        plt.plot(ring_t[ring_t<0.35], h_ring[ring_t<0.35])
        plt.title("Damped ringdown waveform (zoom)")
        plt.grid(True)
        plt.subplot(3,1,3)
        plt.plot(ring_t, inst_freq, label='inst_freq')
        plt.plot(ring_t, f_ring, '--', label='input f(t)')
        plt.ylabel("Hz"); plt.xlabel("Time (s)"); plt.legend(); plt.grid(True)
        plt.tight_layout()
        plt.savefig(prefix + "_waveform.png", dpi=150)
        plt.close()
        
        np.savez(prefix + "_data.npz", ring_t=ring_t, f_ring=f_ring, h_ring=h_ring, 
                 inst_freq=inst_freq, rel_shift=rel_shift, snr=snr)
        print(f"Processed {ch['label']}: SNR={snr:.2f}")
        results_summary.append({'label':ch['label'], 'M0':M0, 'a0':a0, 'snr':snr})

    np.savez("sim6_v16_ringdown_summary.npz", results=results_summary)
    
    # 4) Plot Omega Distance Metric
    plt.figure(figsize=(8,6))
    plt.plot(out['z'], out['chi_Mpc'], 'k-', label=r'Standard $\chi(z)$ (FRW)')
    plt.plot(out['z'], out['d_info_Mpc'], 'r--', label=r'Omega $d_{\mathrm{info}}(z)$ (Correlation)')
    plt.gca().invert_xaxis()
    plt.xlabel('Redshift z')
    plt.ylabel('Distance [Mpc]')
    plt.title('Informational Metric vs Geometric Metric')
    plt.legend()
    plt.grid(True)
    plt.savefig("sim6_v16_omega_distance_metric.png", dpi=150)
    print("Saved sim6_v16_omega_distance_metric.png")

    # 5) Cosmology summary
    plt.figure(figsize=(10,6))
    plt.subplot(2,1,1)
    plt.plot(out['z'], out['H_km_s_Mpc'])
    plt.gca().invert_xaxis(); plt.title("H(z)"); plt.ylabel("km/s/Mpc")
    plt.subplot(2,1,2)
    plt.plot(out['z'], out['A_BH'])
    plt.gca().invert_xaxis(); plt.title("A_BH(z)"); plt.ylabel("Relative Area")
    plt.tight_layout()
    plt.savefig("sim6_v16_cosmo_summary.png", dpi=150)
    print("Saved sim6_v16_cosmo_summary.png")

    # 6) Run local Sim 5 sandboxes
    print("Running Sim 5 v1 (local Ω-dynamics + metabolic inequality)...")
    run_sim5_dynamics()
    print("Running Sim 5 v3.4-style (emergent gravity + GIF)...")
    run_sim5_v34_dynamics()

    # 7) Run Sim 3 (dynamic Planck length & causality band)
    print("Running Sim 3 (dynamic Planck length & disformal causality band)...")
    run_sim3_disformal(RUN_DYNAMIC=True)
    print("Sim 3 complete.")

if __name__ == "__main__":
    main()
