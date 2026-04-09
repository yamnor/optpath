"""Gaussian engine."""

from __future__ import annotations

import os
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from optpath.config.schema import EngineConfig
from optpath.engines.results import ImageJob, ImageResult, RunContext
from optpath.io.templates import render_template
from optpath.utils.filesystem import ensure_dir
from optpath.utils.units import hartree_per_bohr_to_ev_per_angstrom, hartree_to_ev

_SCF_RE = re.compile(r"SCF Done:\s+\S+\s+=\s+(-?\d+(?:\.\d+)?)")
_TDDFT_ENERGY_RE = re.compile(r"Total Energy, E\(TD-HF/TD-DFT\)\s*=\s*(-?\d+\.\d+)")
_EXCITED_STATE_RE = re.compile(r"Excited State\s+(\d+):\s+\S+\s+([-+]?\d+\.\d+)\s+eV")


class GaussianEngine:
    def __init__(self, config: EngineConfig) -> None:
        self.config = config

    def evaluate(self, jobs: list[ImageJob], context: RunContext) -> list[ImageResult]:
        if context.execution_mode == "parallel_images":
            return _parallel_map(_gaussian_worker, jobs, self.config.model_dump(), context)
        return [_gaussian_worker(job, self.config.model_dump(), context) for job in jobs]


def _parallel_map(worker, jobs: list[ImageJob], config: dict, context: RunContext) -> list[ImageResult]:
    results: dict[int, ImageResult] = {}
    with ProcessPoolExecutor(max_workers=context.max_concurrent_images) as executor:
        future_map = {executor.submit(worker, job, config, context): job.image_index for job in jobs}
        for future in as_completed(future_map):
            image_index = future_map[future]
            try:
                result = future.result()
            except Exception as exc:  # pragma: no cover - defensive
                result = ImageResult(
                    image_index=image_index,
                    energy=None,
                    forces=None,
                    gradient=None,
                    selected_root=None,
                    available_roots=[],
                    state_label=None,
                    converged=False,
                    success=False,
                    error_message=str(exc),
                )
            results[image_index] = result
    return [results[job.image_index] for job in jobs]


def _gaussian_worker(job: ImageJob, config: dict, context: RunContext) -> ImageResult:
    workdir = ensure_dir(job.workdir)
    input_name = config.get("extra", {}).get("input_filename", "gau.inp")
    output_name = config.get("extra", {}).get("output_filename", "gau.out")
    input_path = workdir / input_name
    output_path = workdir / output_name
    try:
        tpl = config.get("template_gradient") or config.get("template")
        if not tpl:
            raise ValueError("engine.template (or engine.template_gradient) is required for the Gaussian engine")
        template_path = Path(tpl)
        mm_charges = job.qmmm.get("mm_charges") if job.qmmm.get("enabled") else None
        rendered = render_template(template_path, job.atoms, mm_charges=mm_charges)
        input_path.write_text(rendered, encoding="utf-8")
        command = config.get("command") or f"g16 < {input_name} > {output_name}"
        env = os.environ.copy()
        completed = subprocess.run(
            command,
            cwd=workdir,
            shell=True,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.stdout:
            (workdir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
        if completed.stderr:
            (workdir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
        result = parse_gaussian_output(output_path, image_index=job.image_index, selected_root=job.state_spec.get("root"))
        if completed.returncode != 0:
            result.success = False
            result.error_message = result.error_message or f"Gaussian exited with status {completed.returncode}"
        result.stdout_path = workdir / "stdout.log"
        result.stderr_path = workdir / "stderr.log"
        result.workdir = workdir
        return result
    except Exception as exc:
        return ImageResult(
            image_index=job.image_index,
            energy=None,
            forces=None,
            gradient=None,
            selected_root=None,
            available_roots=[],
            state_label=None,
            converged=False,
            success=False,
            error_message=str(exc),
            workdir=workdir,
        )


def parse_gaussian_output(path: Path, image_index: int, selected_root: int | None = None) -> ImageResult:
    text = path.read_text(encoding="utf-8")
    # Ground-state SCF energy (used as fallback)
    energy = None
    for match in _SCF_RE.finditer(text):
        energy = hartree_to_ev(float(match.group(1)))
    # TDDFT excited-state energy overrides SCF energy when present
    tddft_match = _TDDFT_ENERGY_RE.search(text)
    is_excited = tddft_match is not None
    if is_excited:
        energy = hartree_to_ev(float(tddft_match.group(1)))
    # Available excited states
    available_roots = [
        {"root": int(root), "excitation_energy_ev": float(ev)}
        for root, ev in _EXCITED_STATE_RE.findall(text)
    ]
    state_label = "ground" if not is_excited else (f"root_{selected_root}" if selected_root else "excited")
    converged = "Normal termination of Gaussian" in text and "Convergence failure" not in text
    forces = _parse_gaussian_forces(text)
    gradient = None if forces is None else -forces
    success = energy is not None and forces is not None
    return ImageResult(
        image_index=image_index,
        energy=energy,
        forces=forces,
        gradient=gradient,
        selected_root=selected_root if is_excited else None,
        available_roots=available_roots,
        state_label=state_label,
        converged=converged,
        success=success,
        error_message=None if success else "failed to parse Gaussian output",
    )


def _parse_gaussian_forces(text: str) -> np.ndarray | None:
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if "Forces (Hartrees/Bohr)" in line:
            start = idx + 3
            values = []
            for row in lines[start:]:
                if not row.strip() or row.strip().startswith("---"):
                    if values:
                        break
                    continue
                parts = row.split()
                if len(parts) < 5 or not parts[0].isdigit():
                    if values:
                        break
                    continue
                values.append([hartree_per_bohr_to_ev_per_angstrom(float(x)) for x in parts[-3:]])
            if values:
                return np.array(values, dtype=float)
    return None

