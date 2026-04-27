import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.integrate import solve_ivp


def l_planck(phi):
    return np.exp((1.0 - phi) / 2.0)


def c_conformal(phi):
    return np.exp(-2.0 * phi)


def causality_bound(phi, beta=1.0):
    return np.exp(-(phi + 1.0) / 2.0) / np.sqrt(beta)


def run_static_diagnostics(phi_grid, beta_sweep):
    lp_vals = l_planck(phi_grid)
    conf_vals = c_conformal(phi_grid)

    plt.figure()
    plt.semilogy(phi_grid, lp_vals, "g-o", label="lP(phi)")
    plt.xlabel("phi")
    plt.ylabel("lP")
    plt.title("Dynamic Planck Length")
    plt.legend()
    plt.grid(True)
    plt.savefig("fig4_1a.pdf")

    plt.figure()
    for beta in beta_sweep:
        plt.plot(phi_grid, causality_bound(phi_grid, beta=beta), "o-", label=f"beta={beta}")
    plt.xlabel("phi")
    plt.ylabel("Bound")
    plt.title("Disformal Causality Band")
    plt.legend()
    plt.grid(True)
    plt.savefig("fig4_1b.pdf")

    return lp_vals, conf_vals


def run_dynamic_eom(h0=1.0, m=0.1, beta=1.0):
    def phi_eom(_, y):
        phi_val, phi_dot = y
        h_phi = h0 * np.exp(phi_val)
        phi_ddot = -3.0 * h_phi * phi_dot - m**2 * phi_val
        return [phi_dot, phi_ddot]

    t_span = (0.0, 10.0)
    t_eval = np.linspace(*t_span, 400)
    y0 = [0.0, 0.5 * causality_bound(0.0, beta=beta)]

    sol = solve_ivp(phi_eom, t_span, y0, t_eval=t_eval)
    phi_num, phi_dot_num = sol.y

    h_num = h0 * np.exp(phi_num)
    ln_a = cumtrapz(h_num, sol.t, initial=0.0)
    a_num = np.exp(ln_a)
    bound_num = causality_bound(phi_num, beta=beta)

    ratio = np.abs(phi_dot_num) / np.maximum(bound_num, 1e-12)
    print("Causality satisfied?", bool(np.all(ratio < 1.0)))

    ringdown_shift = 100.0 * np.abs(phi_dot_num) / np.maximum(a_num * l_planck(phi_num), 1e-12)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs[0, 0].plot(sol.t, phi_num)
    axs[0, 0].set_title("phi(t)")

    axs[0, 1].plot(sol.t, phi_dot_num, label="phi_dot")
    axs[0, 1].plot(sol.t, bound_num, "g--", label="bound")
    axs[0, 1].legend()

    axs[1, 0].semilogy(sol.t, a_num)
    axs[1, 0].set_title("a(t)")

    axs[1, 1].plot(sol.t, ringdown_shift)
    axs[1, 1].set_title("Ringdown Shift (%)")

    plt.tight_layout()
    plt.savefig("fig41_eom.pdf")
    plt.show()


if __name__ == "__main__":
    phi = np.linspace(0, 5, 11)
    beta_values = [1.0, 0.5, 0.05]

    run_static_diagnostics(phi, beta_values)
    run_dynamic_eom()
