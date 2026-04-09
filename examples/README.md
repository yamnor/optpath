# Examples

Minimal working examples for each engine and state type.

## Overview

| Directory | Engine | System | Method | State |
|---|---|---|---|---|
| `gaussian/ground/` | Gaussian | SN2 (Cl⁻ + CH₃NH₃⁺) | RHF/3-21G | Ground |
| `gaussian/excited/` | Gaussian | Ethylene (C₂H₄) | TD-B3LYP/6-31G* | Excited (S₁) |
| `qchem/ground/` | Q-Chem | SN2 (Cl⁻ + CH₃NH₃⁺) | B3LYP/6-31G* | Ground |
| `qchem/excited/` | Q-Chem | Ethylene (C₂H₄) | TDDFT/6-31G* | Excited (S₂) |
| `pyscf/ground/` | PySCF | SN2 (Cl⁻ + CH₃NH₃⁺) | B3LYP/6-31G* | Ground |
| `pyscf/excited/` | PySCF | Ethylene (C₂H₄) | TDA/6-31G* | Excited (S₁) |
| `pyscf/qmmm/` | PySCF | Formaldehyde (H₂C=O) in point-charge environment | B3LYP/6-31G* | Ground QM/MM |

## Prerequisites

```bash
conda activate optpath
pip install -e ".[dev]"        # Gaussian / Q-Chem / PySCF (common)
pip install -e ".[dev,pyscf]"  # required for PySCF examples
```

External engines (Gaussian, Q-Chem) must be installed separately and accessible in `PATH`.

---

## Gaussian

Input is generated from a template file (`qm.grad`) by substituting `__geom__` with Cartesian coordinates.

### Ground state — RHF/3-21G, SN2 reaction

```bash
optpath run examples/gaussian/ground/optpath.yaml
optpath inspect examples/gaussian/ground/run-gaussian-ground
```

> `g16` must be in `PATH`. Override the command with `engine.command` if needed.

### Excited state — TD-B3LYP/6-31G*, ethylene

```bash
optpath run examples/gaussian/excited/optpath.yaml
optpath inspect examples/gaussian/excited/run-gaussian-excited
```

> `Root=` and `NStates=` in the template must match `engine.root` / `engine.nroots` in the config (they are not auto-substituted). The TDDFT energy is parsed from the `Total Energy, E(TD-HF/TD-DFT)` line.

---

## Q-Chem

The `$molecule` geometry block is substituted via `__geom__`. Charge and multiplicity must be set directly in the template.

### Ground state — B3LYP/6-31G*, SN2 reaction

```bash
optpath run examples/qchem/ground/optpath.yaml
optpath inspect examples/qchem/ground/run-qchem-ground
```

### Excited state — TDDFT/6-31G*, ethylene

```bash
optpath run examples/qchem/excited/optpath.yaml
optpath inspect examples/qchem/excited/run-qchem-excited
```

> `qchem` must be in `PATH`. `CIS_STATE_DERIV` in the template must match `engine.root`.

---

## PySCF

No template file is needed. Requires `pyscf` (`pip install -e ".[pyscf]"`).

### Ground state — B3LYP/6-31G*, SN2 reaction

```bash
optpath run examples/pyscf/ground/optpath.yaml
optpath inspect examples/pyscf/ground/run-pyscf-ground
```

### Excited state — TDA/6-31G*, ethylene

```bash
optpath run examples/pyscf/excited/optpath.yaml
optpath inspect examples/pyscf/excited/run-pyscf-excited
```

> Use `engine.method: pyscf_td` for TDA (or `tddft` for TDDFT). `engine.nroots` and `engine.root` (1-based) are required.

### QM/MM — B3LYP/6-31G*, formaldehyde in point-charge environment

**System**: H₂C=O (4 QM atoms) surrounded by fixed MM point charges representing a nearby water molecule. The MEP explores C=O bond elongation (1.20 → 1.35 Å) under the electrostatic influence of the MM environment.

```bash
optpath run examples/pyscf/qmmm/optpath.yaml
optpath inspect examples/pyscf/qmmm/run-pyscf-qmmm
```

**Key files**:

| File | Description |
|---|---|
| `path.xyz` | Initial path (8 images, Z-matrix interpolated) |
| `mm_charges.xyzq` | Fixed MM point charges (`x y z charge` per line) |
| `optpath.yaml` | Config with `qmmm` section and `coordinate_mode: qm_region_only` |

**Key config settings**:

```yaml
path:
  coordinate_mode: qm_region_only  # only QM atoms are moved during MEP

qmmm:
  enabled: true
  qm_atoms: [0, 1, 2, 3]           # 0-based atom indices of the QM region
  mm_charges_file: mm_charges.xyzq
  update_region_only: true
```

> In v1, MM atom positions are fixed. Only the QM region is updated during the string method iterations. The MM charges enter the QM Hamiltonian as external point charges via PySCF's `qmmm.mm_charge`.
