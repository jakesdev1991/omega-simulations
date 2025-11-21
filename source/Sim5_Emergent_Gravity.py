
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def run_sim5_dynamics():
    """
    Sim 5: 1D Φ-field dynamics with emergent geometry, inertia=latency,
    and a fixed black-hole horizon that shreds infalling "matter packets".

    New in this version:
    - Tracks a local BH area proxy A_BH_local(t), incremented by shredding events.
    - Tracks internal Ω-information I_internal(t) from the particle region.
    - Models environment information loss per frame from shredded mass.
    - Checks the Omega Metabolic Inequality:
          dI_internal/dt > |dI_env/dt|_shred
    and saves all histories to disk.
    """

    # --- Visual style for the "Computational Universe" aesthetic ---
    plt.style.use('dark_background')
    plt.rcParams['axes.facecolor'] = '#0a0a0a'
    plt.rcParams['figure.facecolor'] = '#0a0a0a'

    # ==========================================
    # 1. OMEGA PROTOCOL: THE UNIVERSE CONFIG
    # ==========================================

    # Simulation Resolution
    N_REGIONS = 200         # number of Q-regions on the line
    DT = 0.1                # time step (arb. units)
    TOTAL_FRAMES = 300      # animation length

    # Omega Theory Constants (local toy-model values)
    PHI_VACUUM = 1.0
    PHI_CRITICAL = 0.1      # φ_c for local lattice (matches your original Sim 5)
    L_PLANCK_BASE = 1.0     # just a scale factor in lattice units
    KAPPA_dyn = 5.0         # inertia as latency κ
    G_dyn = 0.05            # coupling from ∇Φ to acceleration (not Newton's G!)

    # Information-per-mass and BH area scaling (toy constants)
    INFO_PER_MASS = 1.0     # how much "env information" per unit shredded mass
    deltaA_per_mass = 0.05  # how much BH area increases per unit shredded mass

    # Initialize the scalar field Φ
    phi = np.ones(N_REGIONS) * PHI_VACUUM

    # ==========================================
    # 2. CREATE MATTER AND BLACK HOLE
    # ==========================================
    particle_width = 12
    phi_particle_val = 0.4

    # Particles: Position 50 (Left), Position 150 (Right)
    positions = [float(N_REGIONS // 4), float(3 * N_REGIONS // 4)]
    velocities = [0.0, 0.0]

    # Black Hole: fixed horizon region
    bh_pos = N_REGIONS - 20
    bh_width = 5
    phi_bh_val = 0.1
    bh_start = bh_pos - bh_width
    bh_end = bh_pos + bh_width

    # Local BH area proxy (Ω-style horizon area)
    A_BH_local = 1.0  # initial area in arbitrary units

    # ==========================================
    # 3. HELPER FUNCTIONS
    # ==========================================

    def get_emergent_geometry(phi_field):
        """
        Emergent geometry from local Φ, via ℓ_P(Φ) = L_PLANCK_BASE * exp((1-Φ)/φ_c),
        then cumulative sum of ℓ_P gives the physical coordinate x.
        """
        local_l_p = L_PLANCK_BASE * np.exp((PHI_VACUUM - phi_field) / PHI_CRITICAL)
        physical_x = np.cumsum(local_l_p)
        physical_x -= physical_x[0]
        return physical_x

    def get_mass_total(phi_field):
        """
        Mass density ∝ (1 - Φ): informational asymmetry.
        """
        mass_density = np.maximum(0, 1.0 - phi_field)
        return np.sum(mass_density)

    def is_shredded(start, end):
        """
        Horizon crossing condition: packet overlaps BH region.
        """
        return start < bh_end and end > bh_start

    # ==========================================
    # 4. HISTORY ARRAYS FOR Ω-BOOKKEEPING
    # ==========================================

    vel_history = []
    time_history = []
    shred_events = []

    I_internal_history = []          # Ω internal info (particle region)
    dI_internal_dt_history = []      # time derivative of I_internal
    dI_env_loss_dt_history = []      # |dI_env/dt|_shred from shredded mass
    metabolic_satisfied_history = [] # boolean flags

    A_BH_local_history = []          # local BH area vs time
    mass_shred_history = []          # shredded mass per frame

    # ==========================================
    # 5. MAIN ANIMATION LOOP
    # ==========================================

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=False)
    fig.suptitle('Ω PROTOCOL: EMERGENT GRAVITY (Sim 5)',
                 color='#00ffff', fontsize=18, fontweight='bold')

    def update(frame):
        nonlocal positions, velocities, A_BH_local  # we rebind these

        # --- A. RESET FIELD: vacuum + black hole ---
        phi[:] = PHI_VACUUM
        phi[max(0, bh_start):min(N_REGIONS, bh_end)] = phi_bh_val

        # --- B. PHYSICS UPDATE: particles, gravity, shredding ---
        active_indices = []
        current_shred = False
        mass_shred_frame = 0.0  # total shredded mass this frame

        # Temp lists
        next_positions = []
        next_velocities = []

        for i in range(len(positions)):
            current_int_pos = int(positions[i])
            start = current_int_pos - particle_width
            end = current_int_pos + particle_width
            start = max(0, start)
            end = min(N_REGIONS - 1, end)

            # Check horizon crossing
            if is_shredded(start, end):
                shred_events.append(frame)
                current_shred = True

                # Approximate mass of the portion of the packet that crossed
                overlap_start = max(start, bh_start)
                overlap_end = min(end, bh_end)
                overlap_len = max(0, overlap_end - overlap_start)
                mass_shred = (1.0 - phi_particle_val) * overlap_len

                # Update local BH area (A_BH ∝ sequestered Φ-gradient energy)
                A_BH_local += deltaA_per_mass * mass_shred
                mass_shred_frame += mass_shred
                continue

            # Apply matter density for surviving packets
            if start < end:
                phi[start:end] = phi_particle_val
                next_positions.append(positions[i])
                next_velocities.append(velocities[i])

        # Keep only particles that survived this step
        positions[:] = next_positions
        velocities[:] = next_velocities

        # --- C. Ω-INFORMATION BOOKKEEPING (local) ---

        # Internal region: cells occupied by surviving matter packets
        phi_field = phi  # 1D view
        internal_mask = np.isclose(phi_field, phi_particle_val, atol=1e-3)

        # Internal information measure:
        # I_internal = Σ_(packet cells) (1 - Φ)
        if np.any(internal_mask):
            I_internal = np.sum(1.0 - phi_field[internal_mask])
        else:
            I_internal = 0.0
        I_internal_history.append(I_internal)

        # Shredding-driven environment information loss for this frame
        # dI_env_shred = INFO_PER_MASS * mass_shred_frame
        I_env_loss = INFO_PER_MASS * mass_shred_frame
        dI_env_shred_dt = I_env_loss / DT   # magnitude of env loss rate
        dI_env_loss_dt_history.append(dI_env_shred_dt)

        # Time derivative of internal info (finite difference)
        if len(I_internal_history) >= 2:
            dI_int_dt = (I_internal_history[-1] - I_internal_history[-2]) / DT
        else:
            dI_int_dt = 0.0
        dI_internal_dt_history.append(dI_int_dt)

        # Omega Metabolic Inequality:
        #   dI_internal/dt > |dI_env/dt|_shred
        metabolic_ok = dI_int_dt > dI_env_shred_dt
        metabolic_satisfied_history.append(metabolic_ok)

        # Record BH area and shredded mass history
        A_BH_local_history.append(A_BH_local)
        mass_shred_history.append(mass_shred_frame)

        # --- D. GRADIENT-DRIVEN DYNAMICS (gravity + inertia) ---

        grad_phi = np.gradient(phi)
        num_particles = len(positions)

        if num_particles > 0:
            total_mass = get_mass_total(phi)
            if total_mass < 0.001:
                total_mass = 0.001
            mass_per_particle = total_mass / num_particles

            applied_force = 0.0  # no external forcing in this version

            for i in range(num_particles):
                pos = positions[i]
                grad_at_pos = np.interp(pos, np.arange(N_REGIONS), grad_phi)

                # Gravity (from ∇Φ) + inertia (latency KAPPA_dyn)
                a_grav = -G_dyn * grad_at_pos
                a_ext = applied_force / (mass_per_particle + KAPPA_dyn)

                velocities[i] += (a_grav + a_ext) * DT
                positions[i] += velocities[i] * DT

        # --- E. UPDATE EMERGENT GEOMETRY ---
        physical_x = get_emergent_geometry(phi)

        # --- F. VISUALIZATION ---
        ax1.clear()
        ax1.set_title(
            f"Frame {frame} │ Particles: {len(positions)} │ "
            f"Shredded: {len(shred_events)} │ A_BH,loc={A_BH_local:.2f}",
            color='#00ffff', fontsize=12, pad=10
        )
        ax1.set_ylabel("Information Density Φ")
        ax1.set_xlabel("Emergent Distance (Spacetime)")

        # Draw Φ field and "mass" = 1 - Φ
        ax1.plot(physical_x, phi, color='#00ff00', lw=2, label='Φ Field')
        ax1.fill_between(physical_x, phi, 1.0,
                         color='cyan', alpha=0.2, label='Mass (1 - Φ)')

        # "Grid points" to visualize stretching of space
        ax1.scatter(physical_x[::5],
                    np.ones_like(physical_x[::5]) * 0.5,
                    color='white', s=5, alpha=0.4)

        # Draw horizon region
        if 0 <= bh_start < len(physical_x) and 0 <= bh_end < len(physical_x):
            ax1.axvspan(physical_x[int(bh_start)], physical_x[int(bh_end)],
                        color='red', alpha=0.3, label='Event Horizon')

        # Draw particles as yellow orbs
        for pos in positions:
            x_phys = np.interp(pos, np.arange(N_REGIONS), physical_x)
            ax1.scatter(x_phys, 0.4, color='yellow', s=150, marker='o',
                        edgecolor='orange', linewidth=2, zorder=10)

        # Flash "SHREDDED" if a packet crossed the horizon this frame
        if current_shred:
            ax1.text(0.5, 0.5, "SHREDDED", transform=ax1.transAxes,
                     color='red', fontsize=30,
                     ha='center', va='center',
                     alpha=1.0, fontweight='bold', zorder=20)

        # Show metabolic inequality status
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

        # --- Velocity history plot (inertia verification) ---
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
    
    try:
        ani.save('sim5_emergent_gravity.gif', writer='pillow', fps=30)
        print("Simulation saved as 'sim5_emergent_gravity.gif'")
    except Exception as e:
        print(f"Could not save GIF: {e}. Attempting to run animation loop manually for data collection.")
        # Fallback: run manually if saving fails to ensure data is collected
        for f in range(TOTAL_FRAMES):
            update(f)

    plt.show()

    # ==========================================
    # 6. SAVE DIAGNOSTICS AFTER ANIMATION
    # ==========================================

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

    # Optional: quick diagnostic plots
    plt.figure(figsize=(6, 4))
    plt.plot(t_arr, A_BH_local_history)
    plt.xlabel("t (arb. units)")
    plt.ylabel("A_BH,local (arb.)")
    plt.title("Sim 5: Local BH area growth from shredding")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("sim5_local_A_BH_growth.png", dpi=200)

    plt.figure(figsize=(6, 4))
    plt.plot(t_arr, dI_internal_dt_history, label="dI_internal/dt")
    plt.plot(t_arr, dI_env_loss_dt_history, label="|dI_env/dt|_shred")
    plt.xlabel("t (arb. units)")
    plt.ylabel("Rate (arb.)")
    plt.title("Sim 5: Ω metabolic inequality diagnostic")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("sim5_omega_metabolic_check.png", dpi=200)

    plt.close('all')

if __name__ == "__main__":
    run_sim5_dynamics()
