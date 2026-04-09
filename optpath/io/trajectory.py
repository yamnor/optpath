"""Trajectory output."""

from __future__ import annotations

from pathlib import Path

from optpath.io.xyz import write_xyz_images
from optpath.utils.filesystem import ensure_dir


def write_band_trajectory(path: Path, images) -> None:
    ensure_dir(path.parent)
    write_xyz_images(path, list(images))

