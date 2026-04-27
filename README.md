# Omega Protocol Simulations

**Author:** jakesdev1991

This repository contains a suite of physics simulations and theoretical papers related to the **Omega Protocol**: a unified theory positing that reality is a network of informational correlations, where geometry, gravity, and cosmology emerge from information processing.

## Repository Structure

- **`notebooks/`**: Jupyter Notebooks for each simulation. This is the best place to explore the models interactively.
- **`source/`**: Raw Python scripts extracted and cleaned from the original documentation.
- **`papers/`**: LaTeX source and compiled PDF of the theoretical foundation (*ToE-V.Omega.4.0*).
- **`originals/`**: Original reference documents (.docx).

## Simulations

The following simulations are included:

1.  **Informational Geometry Test** (`Informational_Geometry_Test.ipynb`)
    *   *Goal*: Demonstrate how spatial distance emerges from informational correlation.
    *   *Method*: Uses Multidimensional Scaling (MDS) to reconstruct a 1D geometry from a correlation matrix.

2.  **Horizon Shredding Model (Cosmology)** (`Horizon_Shredding_Model.ipynb`)
    *   *Goal*: Model cosmic expansion as an information loss process ("Chain-Break").
    *   *Method*: Simulates the "Depletion Law" where information shredding at horizons drives expansion ($H \propto A_{BH}$).

3.  **Dynamic Scale Simulation** (`Dynamic_Scale_Simulation.ipynb`)
    *   *Goal*: Explore the dynamic Planck length and disformal causality.
    *   *Method*: Solves the equation of motion for a scalar field $\phi$ with a dynamic metric scaling.

4.  **Emergence Omega Sim** (`Emergence_Omega_Sim.ipynb`)
    *   *Goal*: Visualizing the "Omega Protocol" universe.
    *   *Method*: A 1D particle simulation showing emergent gravity and the shredding of matter by a black hole.

5.  **Sim 5: Emergent Gravity** (`Sim5_Emergent_Gravity.ipynb`)
    *   *Goal*: Advanced dynamics with metabolic inequalities.
    *   *Features*: Tracks "Internal Information" vs "Environment Loss" to test the Omega Metabolic Inequality ($dI_{int}/dt > dI_{env}/dt$).

6.  **Sim 6: Omega Suite** (`sim6_v16_omega.ipynb`)
    *   *Goal*: The comprehensive suite.
    *   *Features*: Combines Cosmology, Ringdown frequency shift analysis (gravitational waves), and local emergent gravity sandboxes.

7.  **Universe Lifecycle Model** (`source/universe_lifecycle_model.py`)
    *   *Goal*: Provide a robust, end-to-end timeline from universe inception through late-time heat-death behavior.
    *   *Features*: Integrates expansion history across radiation/matter/Λ eras, tracks horizon growth, temperature cooling, and entropy/information proxies, and emits checkpoint reports for key transitions.

## The Paper

The theoretical basis for these simulations can be found in:
*   [papers/ToE-V.Omega.4.0.pdf](papers/ToE-V.Omega.4.0.pdf)

## Getting Started

### Prerequisites

You will need Python 3 and the following libraries:

```bash
pip install -r requirements.txt
```

### Running the Simulations

1.  Start Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
2.  Navigate to the `notebooks/` directory.
3.  Open any `.ipynb` file and run the cells.

## License

[MIT](LICENSE) (or specify your preferred license)
