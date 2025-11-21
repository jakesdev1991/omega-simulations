
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Use a dark style for the "Computational Universe" aesthetic
plt.style.use('dark_background')

# ==========================================
# 1. OMEGA PROTOCOL: THE UNIVERSE CONFIG
# ==========================================

# Simulation Resolution
N_REGIONS = 200         # Number of Q-regions (nodes)
DT = 0.1                # Time step magnitude
TOTAL_FRAMES = 300      # Duration of simulation

# Omega Theory Constants
PHI_VACUUM = 1.0        # Maximum Overlap (Empty Space)
PHI_CRITICAL = 0.1      # Sensitivity (Space warping factor)
L_PLANCK_BASE = 1.0     # Base grid unit distance
KAPPA = 5.0             # NETWORK LATENCY (The "Informational Inertia" constant)
G = 0.05                # Gravitational constant for emergent gravity (tuned for visible effect)

# Initialize the scalar field phi (Information Density)
phi = np.ones(N_REGIONS) * PHI_VACUUM

# ==========================================
# 2. CREATE MATTER AND BLACK HOLE
# ==========================================
# Matter particles: regions of low overlap (low phi)
particle_width = 12
phi_particle_val = 0.4 

# Multiple particles
positions = [float(N_REGIONS // 4), float(3 * N_REGIONS // 4)]
velocities = [0.0, 0.0]

# Black Hole: fixed deep dip in phi, acting as correlation shredder
bh_pos = N_REGIONS - 20
bh_width = 5
phi_bh_val = 0.1
bh_start = bh_pos - bh_width
bh_end = bh_pos + bh_width

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

def get_emergent_geometry(phi_field):
    local_l_p = L_PLANCK_BASE * np.exp((PHI_VACUUM - phi_field) / PHI_CRITICAL)
    physical_x = np.cumsum(local_l_p)
    physical_x -= physical_x[0]
    return physical_x

def get_mass_total(phi_field):
    mass_density = np.maximum(0, 1.0 - phi_field)
    return np.sum(mass_density)

# New: Check if particle overlaps with BH for shredding
def is_shredded(start, end):
    return start < bh_end and end > bh_start

# ==========================================
# 4. SIMULATION SETUP
# ==========================================

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
fig.suptitle('Omega Protocol v3.4: Emergent Gravity & BH Shredding', color='white', fontsize=16)

# History for plotting (now average velocity for simplicity)
vel_history = []
force_history = []
time_history = []
shred_events = []

# ==========================================
# 5. THE MAIN LOOP (Time Evolution)
# ==========================================

def update(frame):
    global phi, positions, velocities, shred_events
    
    # --- A. RESET FIELD AND DRAW FIXED BH ---
    phi[:] = PHI_VACUUM  # Reset to vacuum
    # Draw BH (fixed)
    phi[max(0, bh_start):min(N_REGIONS, bh_end)] = phi_bh_val
    
    # --- B. DRAW PARTICLES (if not shredded) ---
    active_particles = []
    # Note: We iterate by index, but we need to be careful as we might modify the lists
    # The original code had a bug where it deleted while iterating or used a complex logic.
    # Here we rebuild the lists of survivors.
    
    # Temporary storage for next state
    next_positions = []
    next_velocities = []
    
    # We iterate over the CURRENT state
    for i in range(len(positions)):
        current_int_pos = int(positions[i])
        start = current_int_pos - particle_width
        end = current_int_pos + particle_width
        
        # Handle boundaries
        if start < 0: start = 0
        if end >= N_REGIONS: end = N_REGIONS - 1
        
        # Check for shredding
        if is_shredded(start, end):
            shred_events.append((frame, i))
            continue  # Shred: don't draw, remove particle
        
        # Draw particle
        if start < end:
            phi[start:end] = phi_particle_val
        
        next_positions.append(positions[i])
        next_velocities.append(velocities[i])
    
    # Update global state with survivors
    positions[:] = next_positions
    velocities[:] = next_velocities
    
    # --- C. COMPUTE GRADIENTS FOR GRAVITY ---
    grad_phi = np.gradient(phi)
    
    # --- D. PHYSICS: UPDATE EACH PARTICLE ---
    num_particles = len(positions)
    applied_force = 0.0
    
    if num_particles > 0:
        current_mass = get_mass_total(phi) / num_particles  # Approximate per particle
        
        for i in range(num_particles):
            pos = positions[i]
            
            # Interpolate grad at float pos
            grad_at_pos = np.interp(pos, np.arange(N_REGIONS), grad_phi)
            
            # Emergent Gravity: attract to low phi (a_grav = -G * grad_phi)
            a_grav = -G * grad_at_pos
            
            # External acceleration (if any)
            a_ext = applied_force / (current_mass + KAPPA)
            
            # Total acceleration
            effective_acceleration = a_grav + a_ext
            
            # Update
            velocities[i] += effective_acceleration * DT
            positions[i] += velocities[i] * DT
    
    # --- E. UPDATE GEOMETRY ---
    physical_x = get_emergent_geometry(phi)
    
    # --- F. VISUALIZATION ---
    ax1.clear()
    ax1.set_title(f"Frame {frame}: Emergent Gravity (towards low phi) & BH Shredding")
    ax1.set_ylabel("Information Density (phi)")
    ax1.set_xlabel("Emergent Distance (grid stretching)")
    
    ax1.plot(physical_x, phi, color='#00ff00', lw=2, label='Chain Overlap Density phi')
    ax1.fill_between(physical_x, phi, 1.0, color='cyan', alpha=0.3, label='Mass (Non-Overlap)')
    
    ax1.scatter(physical_x[::4], np.ones_like(physical_x[::4])*0.5, 
                color='white', s=10, alpha=0.6, label='Q-Regions')
    
    # Mark BH
    ax1.axvspan(physical_x[max(0, bh_start)], physical_x[min(N_REGIONS-1, bh_end)], color='red', alpha=0.2, label='Black Hole Horizon')
    
    ax1.set_ylim(0.0, 1.1)
    ax1.legend(loc='lower right')
    
    # Plot 2: Dynamics (average velocity)
    avg_vel = np.mean(velocities) if velocities else 0.0
    vel_history.append(avg_vel)
    force_history.append(applied_force)
    time_history.append(frame)
    
    ax2.clear()
    ax2.set_title(f"Emergent Inertia & Dynamics")
    ax2.set_ylabel("Magnitude")
    ax2.set_xlabel("Time Step")
    
    ax2.plot(time_history, force_history, color='#ff4444', linestyle='--', alpha=0.8, label='External Force')
    ax2.plot(time_history, vel_history, color='#ffff00', lw=2.5, label='Avg Velocity')
    
    ax2.legend(loc='upper left')
    
    # Annotation
    ax2.text(0, max(vel_history or [0]) + 0.05, 
             f"Particles: {num_particles}\n"
             f"Latency (Kappa): {KAPPA}\n"
             f"Gravity G: {G}\n"
             f"Force: {applied_force:.1f}", 
             color='white', fontsize=9, verticalalignment='top')
    
    # Shred events
    for event_frame, particle_id in shred_events:
        if event_frame == frame:
            ax2.text(frame, avg_vel, f"Shredded P{particle_id+1}!", color='red', fontsize=10)

# Run Animation
ani = FuncAnimation(fig, update, frames=TOTAL_FRAMES, interval=20)

# Save to GIF for easy viewing
try:
    ani.save('omega_protocol_simulation.gif', writer='pillow', fps=30)
    print("Simulation saved as 'omega_protocol_simulation.gif'")
except:
    print("Could not save GIF (missing pillow?), but animation object created.")

# To display in Jupyter/Colab:
plt.show()
