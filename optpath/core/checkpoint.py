"""Checkpoint helpers."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from optpath.core.band import ImageBand
from optpath.core.convergence import StepMetrics
from optpath.engines.results import ImageResult, TrackedState
from optpath.utils.filesystem import ensure_dir, read_json, write_json


def save_checkpoint(
    checkpoint_dir: Path,
    name: str,
    band: ImageBand,
    results: list[ImageResult],
    metrics: StepMetrics | None,
    tracked_states: list[TrackedState] | None,
    config_snapshot: str,
    diagnostics: bool = False,
) -> tuple[Path, Path]:
    ensure_dir(checkpoint_dir)
    meta_path = checkpoint_dir / f"{name}.meta.json"
    arrays_path = checkpoint_dir / f"{name}.arrays.npz"
    meta = {
        "name": name,
        "diagnostics": diagnostics,
        "iteration": band.iteration,
        "config_snapshot": config_snapshot,
        "results": [
            {
                "image_index": result.image_index,
                "energy": result.energy,
                "selected_root": result.selected_root,
                "state_label": result.state_label,
                "converged": result.converged,
                "success": result.success,
                "error_message": result.error_message,
                "metadata": result.metadata,
                "warnings": result.warnings,
            }
            for result in results
        ],
        "tracked_states": [asdict(state) for state in (tracked_states or [])],
        "metrics": None
        if metrics is None
        else {
            "step_index": metrics.step_index,
            "rms_grad_perp": metrics.rms_grad_perp,
            "displacement": metrics.displacement,
            "energy_delta": metrics.energy_delta,
            "converged": metrics.converged,
        },
    }
    write_json(meta_path, meta)
    np.savez(
        arrays_path,
        positions=np.stack([image.get_positions() for image in band.images], axis=0),
        dof_mask=band.dof_mask,
        energies=np.array([np.nan if result.energy is None else result.energy for result in results], dtype=float),
        forces=np.stack(
            [
                np.zeros((len(band.images[0]), 3), dtype=float)
                if result.forces is None
                else result.forces
                for result in results
            ],
            axis=0,
        ),
    )
    return meta_path, arrays_path


def load_checkpoint(meta_path: Path, arrays_path: Path, band: ImageBand) -> dict[str, Any]:
    meta = read_json(meta_path)
    arrays = np.load(arrays_path)
    positions = arrays["positions"]
    saved_mask = arrays["dof_mask"]
    if saved_mask.shape != band.dof_mask.shape or not np.array_equal(saved_mask, band.dof_mask):
        raise ValueError("checkpoint dof_mask does not match current configuration")
    for idx, image in enumerate(band.images):
        image.set_positions(positions[idx])
    return {"meta": meta, "arrays": arrays}
