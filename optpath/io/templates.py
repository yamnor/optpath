"""Template rendering helpers."""

from __future__ import annotations

from pathlib import Path

from ase import Atoms


def render_geometry_block(atoms: Atoms) -> str:
    lines: list[str] = []
    for symbol, position in zip(atoms.get_chemical_symbols(), atoms.get_positions(), strict=True):
        lines.append(f"{symbol:2s} {position[0]:16.10f} {position[1]:16.10f} {position[2]:16.10f}")
    return "\n".join(lines)


def render_charges_block(mm_charges: list[dict]) -> str:
    """Format MM point charges as 'x y z charge' lines (Angstrom, used by Gaussian and Q-Chem)."""
    return "\n".join(
        f"{c['x']:14.8f} {c['y']:14.8f} {c['z']:14.8f} {c['charge']:10.6f}"
        for c in mm_charges
    )


def render_template(
    path: str | Path,
    atoms: Atoms,
    mm_charges: list[dict] | None = None,
    placeholder: str = "__geom__",
    charges_placeholder: str = "__charges__",
) -> str:
    template = Path(path).read_text(encoding="utf-8")
    result = template.replace(placeholder, render_geometry_block(atoms))
    if charges_placeholder in result:
        result = result.replace(charges_placeholder, render_charges_block(mm_charges) if mm_charges else "")
    return result

