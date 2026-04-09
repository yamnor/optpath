# optpath

**optpath** is a Python package for computing minimum energy paths (MEP) using the zero-temperature string method (improved string method, 2007).

External quantum chemistry engines (Gaussian, Q-Chem) and a Python-native engine (PySCF) are supported for energy and gradient calculations. Initial path generation via Z-matrix interpolation is built in.

---

## Features

- Zero-temperature string method with equal arc-length reparameterization
- Engines: **Gaussian**, **Q-Chem** (including TDDFT/TDA excited-state MEP), **PySCF**
- Fixed-point-charge **QM/MM** support
- Initial path generation by **Z-matrix (internal coordinate) interpolation**
- Serial and **parallel image** execution modes
- Checkpoint/resume for long calculations
- YAML-based configuration and CLI

---

## Requirements

- Python 3.10+
- [ASE](https://wiki.fysik.dtu.dk/ase/) >= 3.22
- NumPy, SciPy, Pydantic, PyYAML
- [chemcoord](https://chemcoord.readthedocs.io/) >= 2.1, pandas >= 2.0, < 3.0

External engines (Gaussian, Q-Chem) must be installed and accessible in `PATH` separately.

---

## Installation

```bash
git clone https://github.com/yamnor/optpath.git
cd optpath
pip install -e ".[dev]"
```

PySCF support (optional):

```bash
pip install -e ".[dev,pyscf]"
```

---

## Quick Start

### 1. Generate an initial path by Z-matrix interpolation

```bash
optpath interp reactant.xyz product.xyz path.xyz --nimages 8
```

### 2. Create a config file (`optpath.yaml`)

```yaml
path:
  initial_xyz: path.xyz
  nimages: 8

optimizer:
  step_size: 0.05
  max_steps: 100
  grad_tol: 1.0e-3
  disp_tol: 1.0e-3
  energy_tol: 1.0e-6

engine:
  type: gaussian
  method: b3lyp
  basis: 6-31G*
  charge: 0
  multiplicity: 1
  template: qm.grad

execution:
  mode: serial

output:
  run_dir: run
```

### 3. Run

```bash
optpath run optpath.yaml
```

### 4. Inspect results

```bash
optpath inspect run
```

---

## CLI Commands

| Command | Description |
|---|---|
| `optpath interp <reactant> <product> <output>` | Generate initial path by Z-matrix interpolation |
| `optpath run <config>` | Run MEP optimization |
| `optpath singlepoint <config>` | Single-point evaluation (for testing setup) |
| `optpath resume <config>` | Resume from the latest checkpoint |
| `optpath inspect <run_dir>` | Show convergence status |

---

## Supported Engines

| Engine | Ground state | Excited state | Notes |
|---|---|---|---|
| Gaussian | HF, DFT, MP2, ... | — | Template-based input |
| Q-Chem | HF, DFT, ... | TDDFT, TDA, TDDFT-1D | Template-based input |
| PySCF | HF, DFT | TDA, TDDFT | No template needed |

---

## Examples

See [examples/](examples/) for ready-to-run configurations:

- [`examples/gaussian/`](examples/gaussian/) — SN2 reaction, RHF/3-21G
- [`examples/qchem/`](examples/qchem/) — Ethylene, TDDFT/6-31G*

---

## Documentation

- [User Guide](development/docs/user-guide.md)

---

## License

MIT License. See [LICENSE](LICENSE).

> **Note**: Gaussian and Q-Chem are commercial software and require separate licenses. optpath only provides interface code and does not distribute these programs.
