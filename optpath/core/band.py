"""Image band container."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from ase import Atoms

from optpath.core.reparam import arc_lengths, reparameterize as reparameterize_points
from optpath.io.xyz import read_xyz_images


@dataclass
class ImageBand:
    images: list[Atoms]
    fixed_images: list[bool]
    dof_mask: np.ndarray
    qm_atom_indices: list[int] | None = None
    iteration: int = 0
    _last_arc_lengths: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))

    @classmethod
    def from_xyz(
        cls,
        path: str | Path,
        nimages: int,
        freeze_endpoints: bool = True,
        coordinate_mode: str = "all_atoms",
        qm_atom_indices: list[int] | None = None,
    ) -> "ImageBand":
        images = read_xyz_images(path)
        if not images:
            raise ValueError(f"no images found in {path}")
        natoms = len(images[0])
        mask = np.ones((natoms, 3), dtype=bool)
        if coordinate_mode == "qm_region_only":
            mask[:] = False
            if not qm_atom_indices:
                raise ValueError("qm_atom_indices required for qm_region_only")
            mask[qm_atom_indices, :] = True
        fixed = [False] * len(images)
        if freeze_endpoints and images:
            fixed[0] = True
            fixed[-1] = True
        band = cls(images=[image.copy() for image in images], fixed_images=fixed, dof_mask=mask, qm_atom_indices=qm_atom_indices)
        if len(images) != nimages:
            band.resize(nimages)
        return band

    def copy(self) -> "ImageBand":
        return ImageBand(
            images=[image.copy() for image in self.images],
            fixed_images=list(self.fixed_images),
            dof_mask=self.dof_mask.copy(),
            qm_atom_indices=list(self.qm_atom_indices) if self.qm_atom_indices else None,
            iteration=self.iteration,
        )

    @property
    def nimages(self) -> int:
        return len(self.images)

    @property
    def n_active_dof(self) -> int:
        return int(self.dof_mask.sum())

    def resize(self, nimages: int) -> None:
        points = np.vstack([self.get_active_coordinate_vector(i) for i in range(self.nimages)])
        resized = reparameterize_points(points, method="linear")
        target_s = np.linspace(0.0, 1.0, nimages)
        current_s = arc_lengths(resized)
        interp = np.empty((nimages, resized.shape[1]), dtype=float)
        for dim in range(resized.shape[1]):
            interp[:, dim] = np.interp(target_s, current_s, resized[:, dim])
        template = self.images[0]
        self.images = [template.copy() for _ in range(nimages)]
        for idx in range(nimages):
            self.set_active_coordinate_vector(idx, interp[idx])
        self.fixed_images = [False] * nimages
        self.fixed_images[0] = True
        self.fixed_images[-1] = True

    def get_free_image_indices(self) -> list[int]:
        return [idx for idx, fixed in enumerate(self.fixed_images) if not fixed]

    def get_active_coordinate_vector(self, image_index: int) -> np.ndarray:
        positions = self.images[image_index].get_positions()
        return positions[self.dof_mask].astype(float)

    def set_active_coordinate_vector(self, image_index: int, vector: np.ndarray) -> None:
        positions = self.images[image_index].get_positions()
        positions[self.dof_mask] = vector
        self.images[image_index].set_positions(positions)

    def compute_arc_lengths(self) -> np.ndarray:
        points = np.vstack([self.get_active_coordinate_vector(i) for i in range(self.nimages)])
        self._last_arc_lengths = arc_lengths(points)
        return self._last_arc_lengths.copy()

    def reparameterize(self, method: str = "cubic") -> None:
        if self.nimages <= 2:
            return
        endpoints = (
            self.images[0].get_positions().copy(),
            self.images[-1].get_positions().copy(),
        )
        points = np.vstack([self.get_active_coordinate_vector(i) for i in range(self.nimages)])
        updated = reparameterize_points(points, method=method)
        for idx in range(self.nimages):
            self.set_active_coordinate_vector(idx, updated[idx])
        self.images[0].set_positions(endpoints[0])
        self.images[-1].set_positions(endpoints[1])
        self.compute_arc_lengths()

