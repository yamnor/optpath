"""QM-MM van der Waals (Lennard-Jones) repulsion correction.

Parameters are taken from the Universal Force Field (UFF):
  Rappe et al., J. Am. Chem. Soc. 1992, 114, 10024-10035.

Only the repulsive r⁻¹² term is applied by default (vdw_only_repulsion=True),
which is sufficient to prevent QM-MM over-polarization without introducing
spurious long-range attractive forces.

Units throughout: Angstrom, eV.
"""

from __future__ import annotations

import numpy as np

# UFF vdW parameters: element -> (sigma [Å], epsilon [kcal/mol])
# sigma = x_i (vdW radius) from UFF Table 1, epsilon = D_i (well depth)
# Values from Rappe et al. 1992, Table 1.
# fmt: off
_UFF_PARAMS: dict[str, tuple[float, float]] = {
    "H":  (2.571, 0.044),
    "He": (2.104, 0.056),
    "Li": (2.183, 0.025),
    "Be": (2.445, 0.085),
    "B":  (3.638, 0.180),
    "C":  (3.431, 0.105),
    "N":  (3.261, 0.069),
    "O":  (3.118, 0.060),
    "F":  (2.997, 0.050),
    "Ne": (2.889, 0.042),
    "Na": (2.658, 0.030),
    "Mg": (2.691, 0.111),
    "Al": (4.008, 0.505),
    "Si": (3.826, 0.402),
    "P":  (3.694, 0.305),
    "S":  (3.594, 0.274),
    "Cl": (3.516, 0.227),
    "Ar": (3.446, 0.185),
    "K":  (3.396, 0.035),
    "Ca": (3.028, 0.238),
    "Sc": (2.935, 0.019),
    "Ti": (2.828, 0.017),
    "V":  (2.800, 0.016),
    "Cr": (2.693, 0.015),
    "Mn": (2.638, 0.013),
    "Fe": (2.594, 0.013),
    "Co": (2.559, 0.014),
    "Ni": (2.525, 0.015),
    "Cu": (3.114, 0.005),
    "Zn": (2.462, 0.124),
    "Ga": (3.905, 0.415),
    "Ge": (3.813, 0.379),
    "As": (3.768, 0.309),
    "Se": (3.746, 0.291),
    "Br": (3.732, 0.251),
    "Kr": (3.689, 0.220),
    "Rb": (3.665, 0.040),
    "Sr": (3.244, 0.235),
    "Y":  (2.980, 0.072),
    "Zr": (2.783, 0.069),
    "Nb": (2.820, 0.059),
    "Mo": (2.719, 0.056),
    "Tc": (2.671, 0.048),
    "Ru": (2.640, 0.056),
    "Rh": (2.609, 0.053),
    "Pd": (2.583, 0.048),
    "Ag": (2.805, 0.036),
    "Cd": (2.537, 0.228),
    "In": (3.976, 0.599),
    "Sn": (3.913, 0.567),
    "Sb": (3.937, 0.449),
    "Te": (3.982, 0.398),
    "I":  (4.009, 0.339),
    "Xe": (3.923, 0.332),
    "Cs": (4.024, 0.045),
    "Ba": (3.299, 0.364),
    "La": (3.137, 0.017),
    "Ce": (3.168, 0.013),
    "Pr": (3.213, 0.010),
    "Nd": (3.185, 0.010),
    "Sm": (3.137, 0.010),
    "Eu": (3.098, 0.010),
    "Gd": (3.065, 0.009),
    "Tb": (3.034, 0.009),
    "Dy": (3.010, 0.009),
    "Ho": (2.997, 0.010),
    "Er": (2.977, 0.010),
    "Tm": (2.963, 0.011),
    "Yb": (2.999, 0.028),
    "Lu": (3.074, 0.010),
    "Hf": (2.798, 0.072),
    "Ta": (2.824, 0.082),
    "W":  (2.734, 0.067),
    "Re": (2.672, 0.066),
    "Os": (2.631, 0.037),
    "Ir": (2.600, 0.073),
    "Pt": (2.454, 0.080),
    "Au": (2.934, 0.039),
    "Hg": (2.409, 0.385),
    "Tl": (3.873, 0.680),
    "Pb": (3.828, 0.663),
    "Bi": (3.893, 0.518),
    "Po": (4.195, 0.325),
    "At": (4.231, 0.284),
    "Rn": (4.245, 0.248),
}
# fmt: on

_KCAL_TO_EV = 0.04336411531  # 1 kcal/mol = 0.04336... eV


def get_uff_params(symbol: str) -> tuple[float, float]:
    """Return (sigma [Å], epsilon [eV]) for the given element symbol.

    Falls back to carbon parameters for unknown elements and issues a warning.
    """
    key = symbol.capitalize()
    if key not in _UFF_PARAMS:
        import warnings
        warnings.warn(
            f"No UFF parameters for element '{symbol}'; using carbon defaults.",
            stacklevel=2,
        )
        key = "C"
    sigma, eps_kcal = _UFF_PARAMS[key]
    return sigma, eps_kcal * _KCAL_TO_EV


def lj_forces(
    qm_positions: np.ndarray,
    qm_symbols: list[str],
    mm_positions: np.ndarray,
    mm_symbols: list[str],
    only_repulsion: bool = True,
) -> np.ndarray:
    """Compute LJ forces on QM atoms due to MM atoms.

    Uses Lorentz-Berthelot combining rules:
        sigma_ij = (sigma_i + sigma_j) / 2
        epsilon_ij = sqrt(epsilon_i * epsilon_j)

    Parameters
    ----------
    qm_positions : (N_qm, 3) array, Angstrom
    qm_symbols   : list of N_qm element symbols
    mm_positions : (N_mm, 3) array, Angstrom
    mm_symbols   : list of N_mm element symbols
    only_repulsion : if True, apply only the r⁻¹² repulsive term

    Returns
    -------
    forces : (N_qm, 3) array, eV/Angstrom  (forces = -dV/dr)
    """
    n_qm = len(qm_symbols)
    forces = np.zeros((n_qm, 3), dtype=float)

    qm_params = [get_uff_params(s) for s in qm_symbols]
    mm_params = [get_uff_params(s) for s in mm_symbols]

    for i, (pos_i, (sig_i, eps_i)) in enumerate(zip(qm_positions, qm_params)):
        for pos_j, (sig_j, eps_j) in zip(mm_positions, mm_params):
            r_vec = pos_i - pos_j          # vector from MM to QM atom
            r2 = float(np.dot(r_vec, r_vec))
            if r2 < 1e-6:
                continue
            r = np.sqrt(r2)

            sig_ij = (sig_i + sig_j) / 2.0
            eps_ij = np.sqrt(eps_i * eps_j)

            sr6 = (sig_ij / r) ** 6
            sr12 = sr6 * sr6

            if only_repulsion:
                # dV/dr = 4ε * (-12 σ¹²/r¹³)  →  force = +48ε σ¹²/r¹³ * r_hat
                dv_dr = -48.0 * eps_ij * sr12 / r2
            else:
                # Full LJ: dV/dr = 4ε * (-12 σ¹²/r¹³ + 6 σ⁶/r⁷)
                dv_dr = 4.0 * eps_ij * (-12.0 * sr12 + 6.0 * sr6) / r2

            forces[i] -= dv_dr * r_vec   # F = -dV/dr * r_hat * r = -dV/dr * r_vec

    return forces
