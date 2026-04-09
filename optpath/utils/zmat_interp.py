"""Internal-coordinate interpolation using Z-matrix (chemcoord)."""

from __future__ import annotations

import chemcoord as cc
import numpy as np
import pandas as pd
from ase import Atoms

from optpath.utils.alignment import kabsch_align_positions


def _ase_to_cc(atoms: Atoms) -> cc.Cartesian:
    """Convert ASE Atoms to chemcoord Cartesian."""
    symbols = atoms.get_chemical_symbols()
    pos = atoms.get_positions()
    return cc.Cartesian(
        pd.DataFrame({"atom": symbols, "x": pos[:, 0], "y": pos[:, 1], "z": pos[:, 2]})
    )


def _cc_to_ase(cart: cc.Cartesian, reference: Atoms, align_to: np.ndarray | None = None) -> Atoms:
    """Convert chemcoord Cartesian back to ASE Atoms, preserving cell/pbc from reference.

    chemcoord may reorder rows in the internal frame when building the Z-matrix tree.
    We sort by the original integer index (0..N-1) to restore the original atom order.
    """
    ordered = cart._frame.sort_index()
    pos = ordered[["x", "y", "z"]].values.astype(float)
    if align_to is not None:
        pos = kabsch_align_positions(pos, align_to)
    result = reference.copy()
    result.set_positions(pos)
    return result


def _interp_angle(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Shortest-arc interpolation for dihedral angles (degrees)."""
    diff = (b - a + 180.0) % 360.0 - 180.0
    return a + t * diff


def interpolate_zmat(
    atoms1: Atoms,
    atoms2: Atoms,
    nimages: int,
    interpolate_dihedrals: bool = True,
) -> list[Atoms]:
    """Return a list of *nimages* structures interpolated in Z-matrix space.

    atoms1 and atoms2 are the endpoints (included in the returned list).
    The Z-matrix topology is derived from atoms1 and reused for atoms2.

    Parameters
    ----------
    atoms1, atoms2:
        Endpoint structures (must have same atoms in the same order).
    nimages:
        Total number of structures in the path, including both endpoints.
    interpolate_dihedrals:
        If True, wrap dihedral differences into [-180, 180] before interpolating
        (avoids going the long way round on angle changes near ±180°).

    Raises
    ------
    ValueError:
        If atoms1 and atoms2 have different atom types or counts, or nimages < 2.
    """
    if nimages < 2:
        raise ValueError("nimages must be >= 2")

    syms1 = atoms1.get_chemical_symbols()
    syms2 = atoms2.get_chemical_symbols()
    if syms1 != syms2:
        raise ValueError(
            f"atoms1 and atoms2 must have the same atoms in the same order; "
            f"got {syms1} vs {syms2}"
        )

    cart1 = _ase_to_cc(atoms1)
    cart2 = _ase_to_cc(atoms2)

    zmat1 = cart1.get_zmat()
    c_table = zmat1.loc[:, ["b", "a", "d"]]
    zmat2 = cart2.get_zmat(c_table)

    v1 = zmat1.loc[:, ["bond", "angle", "dihedral"]].values.astype(float)
    v2 = zmat2.loc[:, ["bond", "angle", "dihedral"]].values.astype(float)

    images: list[Atoms] = []
    for i, t in enumerate(np.linspace(0.0, 1.0, nimages)):
        if i == 0:
            images.append(atoms1.copy())
            continue
        if i == nimages - 1:
            images.append(atoms2.copy())
            continue

        bond_interp = (1.0 - t) * v1[:, 0] + t * v2[:, 0]
        angle_interp = (1.0 - t) * v1[:, 1] + t * v2[:, 1]

        if interpolate_dihedrals:
            dihedral_interp = _interp_angle(v1[:, 2], v2[:, 2], t)
        else:
            dihedral_interp = (1.0 - t) * v1[:, 2] + t * v2[:, 2]

        new_df = zmat1._frame.copy()
        new_df["bond"] = bond_interp
        new_df["angle"] = angle_interp
        new_df["dihedral"] = dihedral_interp
        zmat_mid = cc.Zmat(new_df, zmat1.metadata)
        cart_mid = zmat_mid.get_cartesian()
        images.append(_cc_to_ase(cart_mid, atoms1, align_to=images[-1].get_positions()))

    return images


def get_zmatrix_string(atoms: Atoms) -> str:
    """Return a human-readable Z-matrix string for the given structure."""
    cart = _ase_to_cc(atoms)
    zmat = cart.get_zmat()
    lines: list[str] = []
    for _, row in zmat._frame.iterrows():
        atom = row["atom"]
        b, bond = row["b"], row["bond"]
        a, angle = row["a"], row["angle"]
        d, dihedral = row["d"], row["dihedral"]
        lines.append(
            f"{atom:2s}  {str(b):8s} {bond:12.6f}  {str(a):8s} {angle:12.6f}  "
            f"{str(d):8s} {dihedral:12.6f}"
        )
    return "\n".join(lines)
