"""Shared engine result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from ase import Atoms


@dataclass(slots=True)
class RunContext:
    execution_mode: str
    max_concurrent_images: int
    threads_per_image: int
    total_slots: int | None
    scratch_root: Path | None
    run_dir: Path
    step_index: int


@dataclass(slots=True)
class ImageJob:
    image_index: int
    atoms: Atoms
    state_spec: dict[str, Any]
    workdir: Path
    scratch_dir: Path | None
    previous_metadata: dict[str, Any] = field(default_factory=dict)
    qmmm: dict[str, Any] = field(default_factory=dict)
    active_mask: np.ndarray | None = None


@dataclass(slots=True)
class ImageResult:
    image_index: int
    energy: float | None
    forces: np.ndarray | None
    gradient: np.ndarray | None
    selected_root: int | None
    available_roots: list[dict[str, Any]]
    state_label: str | None
    converged: bool
    success: bool
    error_message: str | None
    metadata: dict[str, Any] = field(default_factory=dict)
    stdout_path: Path | None = None
    stderr_path: Path | None = None
    workdir: Path | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TrackedState:
    image_index: int
    selected_root: int | None
    state_label: str | None
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class Engine(Protocol):
    def evaluate(self, jobs: list[ImageJob], context: RunContext) -> list[ImageResult]:
        """Evaluate a batch of image jobs."""

