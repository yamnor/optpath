"""QM region helpers."""

from __future__ import annotations

import numpy as np


def qm_dof_mask(natoms: int, qm_atoms: list[int]) -> np.ndarray:
    mask = np.zeros((natoms, 3), dtype=bool)
    if qm_atoms:
        mask[qm_atoms, :] = True
    return mask

