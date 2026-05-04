#!/usr/bin/env python3
"""Universe lifecycle simulation from inception to heat death.

This module introduces a compact but robust cosmology model that tracks:
- Expansion dynamics with radiation, matter, and dark-energy components.
- Horizon and comoving distance scales.
- Temperature, entropy, and an information proxy over cosmic time.
- Epoch transitions from radiation era to dark-energy dominance.

The implementation is intentionally lightweight and numerically stable so it can be
used directly by notebooks/scripts in this repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class LifecycleParams:
    """Physical parameters for the lifecycle model."""

    h0_km_s_mpc: float = 67.4
    omega_r: float = 9.2e-5
    omega_m: float = 0.315
    omega_lambda: float = 0.685
    t_start_gyr: float = 1e-6
    t_end_gyr: float = 3.0e3
    n_steps: int = 12000
    t0_cmb_k: float = 2.7255


class UniverseLifecycleModel:
    """Simulate universe evolution from early expansion to asymptotic heat death."""

    C_KM_S = 299792.458
    MPC_IN_KM = 3.085677581e19
    SEC_PER_GYR = 3.15576e16
    C_MPC_PER_GYR = C_KM_S * SEC_PER_GYR / MPC_IN_KM

    def __init__(self, params: LifecycleParams | None = None):
        self.params = params or LifecycleParams()
        self._validate_params()

    def _validate_params(self) -> None:
        p = self.params
        if p.n_steps < 1000:
            raise ValueError("n_steps must be >= 1000 for stable epoch detection")
        if min(p.omega_r, p.omega_m, p.omega_lambda) < 0:
            raise ValueError("Density parameters must be non-negative")

    @property
    def h0_gyr_inv(self) -> float:
        return (self.params.h0_km_s_mpc / self.MPC_IN_KM) * self.SEC_PER_GYR

    def _hubble(self, a: np.ndarray) -> np.ndarray:
        p = self.params
        e2 = p.omega_r / a**4 + p.omega_m / a**3 + p.omega_lambda
        return self.h0_gyr_inv * np.sqrt(np.maximum(e2, 1e-30))

    @staticmethod
    def _cumtrapz(y: np.ndarray, x: np.ndarray) -> np.ndarray:
        out = np.zeros_like(y)
        out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))
        return out

    def simulate(self) -> Dict[str, np.ndarray]:
        """Return a full lifecycle history as NumPy arrays."""
        p = self.params
        t = np.geomspace(p.t_start_gyr, p.t_end_gyr, p.n_steps)
        a_init = np.sqrt(2.0 * self.h0_gyr_inv * np.sqrt(p.omega_r) * p.t_start_gyr)

        def ode(_, y):
            a_val = max(y[0], 1e-20)
            return [a_val * self._hubble(np.array([a_val]))[0]]

        sol = solve_ivp(
            ode,
            (t[0], t[-1]),
            [a_init],
            t_eval=t,
            method="Radau",
            atol=1e-11,
            rtol=1e-9,
            max_step=0.25,
        )
        if not sol.success:
            raise RuntimeError(f"Lifecycle integration failed: {sol.message}")
        a = np.maximum(sol.y[0], 1e-20)

        h = self._hubble(a)
        z = 1.0 / np.maximum(a, 1e-20) - 1.0
        t_hubble = 1.0 / np.maximum(h, 1e-20)

        integrand = self.C_MPC_PER_GYR / np.maximum(a * h, 1e-20)
        horizon_mpc = self._cumtrapz(integrand, t)

        temp_k = p.t0_cmb_k / np.maximum(a, 1e-20)
        entropy_proxy = a**3 * temp_k**3
        info_proxy = 1.0 / (1.0 + np.log1p(np.maximum(entropy_proxy, 1e-30)))

        rho_r = p.omega_r / a**4
        rho_m = p.omega_m / a**3
        rho_l = np.full_like(a, p.omega_lambda)

        return {
            "t_gyr": t,
            "a": a,
            "z": z,
            "H_gyr_inv": h,
            "t_hubble_gyr": t_hubble,
            "horizon_mpc": horizon_mpc,
            "temp_k": temp_k,
            "entropy_proxy": entropy_proxy,
            "info_proxy": info_proxy,
            "rho_r": rho_r,
            "rho_m": rho_m,
            "rho_lambda": rho_l,
        }

    def epoch_report(self, history: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Compute key timeline markers for inception-to-death narrative."""
        t = history["t_gyr"]
        rho_r = history["rho_r"]
        rho_m = history["rho_m"]
        rho_l = history["rho_lambda"]
        a = history["a"]

        def first_crossing_index(x: np.ndarray, y: np.ndarray) -> int:
            diff = x - y
            signs = np.sign(diff)
            crossing = np.where(signs[:-1] * signs[1:] <= 0)[0]
            if len(crossing) > 0:
                return int(crossing[0] + 1)
            return int(np.argmin(np.abs(np.log(np.maximum(x, 1e-300)) - np.log(np.maximum(y, 1e-300)))))

        idx_rm = first_crossing_index(rho_r, rho_m)
        idx_ml = first_crossing_index(rho_m, rho_l)
        idx_hd = int(np.argmin(np.abs(history["temp_k"] - 1e-3)))

        return {
            "radiation_matter_eq_gyr": float(t[idx_rm]),
            "matter_lambda_eq_gyr": float(t[idx_ml]),
            "heat_death_threshold_gyr": float(t[idx_hd]),
            "final_scale_factor": float(a[-1]),
        }

    def plot_summary(self, history: Dict[str, np.ndarray], save_path: str = "universe_lifecycle_summary.png") -> None:
        """Generate a polished 2x2 summary plot for lifecycle diagnostics."""
        t = history["t_gyr"]
        plt.style.use("dark_background")
        fig, ax = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)
        fig.patch.set_facecolor("#0b1020")

        ax[0, 0].loglog(t, history["a"], color="#4cc9f0", lw=2.2)
        ax[0, 0].set_title("Scale Factor Evolution", color="white")
        ax[0, 0].set_xlabel("Time [Gyr]")
        ax[0, 0].set_ylabel("a(t)")

        ax[0, 1].loglog(t, history["temp_k"], color="#fca311", lw=2.2)
        ax[0, 1].set_title("CMB Temperature Cooling", color="white")
        ax[0, 1].set_xlabel("Time [Gyr]")
        ax[0, 1].set_ylabel("Temperature [K]")

        ax[1, 0].loglog(t, history["rho_r"], label="Radiation", color="#ff6b6b", lw=1.8)
        ax[1, 0].loglog(t, history["rho_m"], label="Matter", color="#ffd166", lw=1.8)
        ax[1, 0].loglog(t, history["rho_lambda"], label="Dark energy", color="#06d6a0", lw=1.8)
        ax[1, 0].set_title("Energy Component Densities", color="white")
        ax[1, 0].set_xlabel("Time [Gyr]")
        ax[1, 0].set_ylabel("Relative Density")
        ax[1, 0].legend(frameon=False, fontsize=9)

        ax[1, 1].semilogx(t, history["info_proxy"], color="#b5179e", lw=2.2)
        ax[1, 1].set_title("Information Proxy", color="white")
        ax[1, 1].set_xlabel("Time [Gyr]")
        ax[1, 1].set_ylabel("I_proxy")

        for axis in ax.flat:
            axis.grid(True, which="both", alpha=0.25, linestyle="--")
            axis.set_facecolor("#111827")

        fig.suptitle("Universe Lifecycle: Inception → Heat Death", fontsize=14, color="#e5e7eb")
        plt.savefig(save_path, dpi=220, bbox_inches="tight")
        plt.close(fig)


def run_universe_lifecycle(save_plot: bool = True) -> Dict[str, np.ndarray]:
    """Convenience runner for scripts and notebooks."""
    model = UniverseLifecycleModel()
    hist = model.simulate()
    report = model.epoch_report(hist)

    print("Universe lifecycle checkpoints (Gyr):")
    for key, value in report.items():
        print(f"  {key}: {value:.6g}")

    if save_plot:
        model.plot_summary(hist)
        print("Saved plot: universe_lifecycle_summary.png")

    return hist


if __name__ == "__main__":
    run_universe_lifecycle(save_plot=True)
