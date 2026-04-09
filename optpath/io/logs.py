"""Logging helpers."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

from optpath.core.convergence import StepMetrics
from optpath.engines.results import ImageResult, TrackedState
from optpath.utils.filesystem import ensure_dir


SUMMARY_HEADER = [
    "step",
    "total_images",
    "active_images",
    "max_rms_grad_perp_eV_per_A",
    "avg_rms_grad_perp_eV_per_A",
    "max_displacement_A",
    "max_abs_delta_energy_eV",
    "selected_roots",
    "warning_count",
]

TABLE_HEADER = [
    "step",
    "image_index",
    "total_energy_eV",
    "relative_energy_eV",
    "rms_gradient_eV_per_A",
    "rms_perpendicular_gradient_eV_per_A",
    "displacement_A",
    "selected_root",
    "state_label",
    "converged",
    "success",
]


def write_summary(path: Path, metrics: StepMetrics, total_images: int, active_images: int, tracked: list[TrackedState]) -> None:
    ensure_dir(path.parent)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if not exists:
            writer.writerow(SUMMARY_HEADER)
        writer.writerow(
            [
                metrics.step_index,
                total_images,
                active_images,
                f"{metrics.max_rms_grad_perp:.10f}",
                f"{metrics.avg_rms_grad_perp:.10f}",
                f"{metrics.max_displacement:.10f}",
                "NA" if metrics.max_abs_delta_energy is None else f"{metrics.max_abs_delta_energy:.10f}",
                " ".join(
                    f"{state.image_index}:{state.selected_root if state.selected_root is not None else 'NA'}"
                    for state in tracked
                ),
                sum(len(state.warnings) for state in tracked),
            ]
        )


def write_table(path: Path, step_index: int, results: list[ImageResult], rms_perp: dict[int, float], displacement: dict[int, float]) -> None:
    ensure_dir(path.parent)
    exists = path.exists()
    finite_energies = [result.energy for result in results if result.energy is not None and math.isfinite(result.energy)]
    reference = finite_energies[0] if finite_energies else None
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if not exists:
            writer.writerow(TABLE_HEADER)
        for result in results:
            rel = None if reference is None or result.energy is None else result.energy - reference
            grad = None
            if result.gradient is not None:
                grad = float((result.gradient**2).mean() ** 0.5)
            writer.writerow(
                [
                    step_index,
                    result.image_index,
                    _fmt_float(result.energy),
                    _fmt_float(rel),
                    _fmt_float(grad),
                    _fmt_float(rms_perp.get(result.image_index)),
                    _fmt_float(displacement.get(result.image_index)),
                    result.selected_root,
                    result.state_label,
                    result.converged,
                    result.success,
                ]
            )


def read_summary(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_table(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _fmt_float(value: float | None) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "NA"
    return f"{value:.10f}"

