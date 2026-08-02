# Omega Simulations

Computational physics simulations and Jupyter notebooks for the Omega Theory framework. Includes gravitational dynamics, cosmological evolution, and emergent geometry simulations.

## Overview

This repository contains simulation code and notebooks exploring:

- **Emergent Geometry**: Spacetime from quantum entanglement
- **Cosmological Evolution**: Scale factor dynamics, dark energy
- **Dynamic Scale**: Renormalization group flow in spacetime
- **Evolution**: System evolution with depletion mechanics
- **Emergent Gravity**: Gravity as entropic force

## Structure

```
omega-simulations/
├── notebooks/           # Jupyter notebooks
├── cpp/                 # C++ simulation cores
├── source/              # Source documents
├── papers/              # Research papers
├── originals/           # Original data
├── create_notebooks.py  # Notebook generator
├── extract_docx.py      # Document extractor
└── requirements.txt     # Python dependencies
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate notebooks
python create_notebooks.py

# Run Jupyter
jupyter lab notebooks/
```

## Simulations

| Notebook | Description |
|----------|-------------|
| `Sim1_Emergent_Geometry.ipynb` | Quantum entanglement → spacetime |
| `Sim2_Cosmology.ipynb` | FLRW cosmology with dynamic scale |
| `Sim3_Dynamic_Scale.ipynb` | RG flow in spacetime |
| `Sim4_Evolution.ipynb` | System evolution with depletion |
| `Sim5_Emergent_Gravity.ipynb` | Entropic gravity derivation |
| `sim6_v14_depletion.ipynb` | v14 depletion mechanics |

## Development

```bash
# Lint notebooks
nbqa ruff notebooks/

# Execute notebooks
jupyter nbconvert --execute --to notebook --inplace notebooks/*.ipynb
```

## License

See [LICENSE](LICENSE) for details.