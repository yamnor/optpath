"""XYZ helpers."""

from __future__ import annotations

from pathlib import Path

from ase import Atoms
from ase.io import read, write


def read_xyz_images(path: str | Path) -> list[Atoms]:
    images = read(str(path), index=":")
    if isinstance(images, Atoms):
        return [images]
    return [image.copy() for image in images]


def write_xyz_images(path: str | Path, images: list[Atoms]) -> None:
    write(str(path), images, format="xyz")

