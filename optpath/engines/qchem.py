"""Q-Chem engine."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from optpath.config.schema import EngineConfig
from optpath.engines.results import ImageJob, ImageResult, RunContext
from optpath.io.templates import render_template
from optpath.utils.filesystem import ensure_dir
from optpath.utils.units import hartree_per_bohr_to_ev_per_angstrom, hartree_to_ev

_TOTAL_ENERGY_FINAL_BASIS_RE = re.compile(
    r"Total energy in the final basis set\s*=\s*([-+]?\d+\.\d+(?:[Ee][-+]?\d+)?)"
)
# Q-Chem 6.x often prints "Total energy =  -556.64645845" without "in the final basis set".
_TOTAL_ENERGY_EQ_RE = re.compile(r"^\s*Total energy\s*=\s*([-+]?\d+\.\d+(?:[Ee][-+]?\d+)?)\s*$", re.MULTILINE)
_EXCITED_STATE_RE = re.compile(r"Excited state\s+(\d+):.*?=\s*([-+]?\d+\.\d+)")


def _promote_qchem_grad_file(workdir: Path, grad_name: str, dest: Path) -> None:
    """If Q-Chem wrote GRAD under workdir/<scratch_id>/, copy it next to qm.out (dest)."""
    if dest.exists():
        return
    candidates: list[Path] = []
    for sub in workdir.iterdir():
        if sub.is_dir():
            p = sub / grad_name
            if p.is_file():
                candidates.append(p)
    if not candidates:
        return
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    shutil.copy2(candidates[0], dest)


def _qchem_total_energy_ev(text: str) -> float | None:
    m = _TOTAL_ENERGY_FINAL_BASIS_RE.search(text)
    if m:
        return hartree_to_ev(float(m.group(1)))
    matches = list(_TOTAL_ENERGY_EQ_RE.finditer(text))
    if matches:
        return hartree_to_ev(float(matches[-1].group(1)))
    return None


def _parse_qchem_cartesian_gradient(text: str) -> np.ndarray | None:
    """Parse Hartree/Bohr Cartesian gradient from standard output when GRAD is absent."""
    markers = ("Gradient of SCF Energy", "Gradient of excited state energy")
    start = -1
    for m in markers:
        pos = text.find(m)
        if pos >= 0:
            start = pos + len(m)
            break
    if start < 0:
        return None
    lines = text[start:].splitlines()
    rows: list[list[float]] = []

    def _is_header(parts: list[str]) -> bool:
        return bool(parts) and all(re.fullmatch(r"-?\d+", p) is not None for p in parts)

    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.strip()
        if not line:
            i += 1
            continue
        parts = line.split()
        if _is_header(parts):
            n = len(parts)
            i += 1
            block: list[list[float]] = []
            for _ in range(3):
                if i >= len(lines):
                    return None
                row_parts = lines[i].split()
                i += 1
                if len(row_parts) < n + 1:
                    return None
                try:
                    block.append(
                        [hartree_per_bohr_to_ev_per_angstrom(float(row_parts[j])) for j in range(1, n + 1)]
                    )
                except ValueError:
                    return None
            for j in range(n):
                rows.append([block[0][j], block[1][j], block[2][j]])
            continue
        if "Max gradient" in line or "RMS gradient" in line or "Gradient time:" in line:
            break
        i += 1

    if not rows:
        return None
    return np.array(rows, dtype=float)


class QChemEngine:
    def __init__(self, config: EngineConfig) -> None:
        self.config = config

    def evaluate(self, jobs: list[ImageJob], context: RunContext) -> list[ImageResult]:
        if context.execution_mode == "parallel_images":
            return _parallel_map(_qchem_worker, jobs, self.config.model_dump(), context)
        return [_qchem_worker(job, self.config.model_dump(), context) for job in jobs]


def _parallel_map(worker, jobs: list[ImageJob], config: dict, context: RunContext) -> list[ImageResult]:
    results: dict[int, ImageResult] = {}
    with ProcessPoolExecutor(max_workers=context.max_concurrent_images) as executor:
        future_map = {executor.submit(worker, job, config, context): job.image_index for job in jobs}
        for future in as_completed(future_map):
            image_index = future_map[future]
            try:
                result = future.result()
            except Exception as exc:  # pragma: no cover
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


def _qchem_worker(job: ImageJob, config: dict, context: RunContext) -> ImageResult:
    workdir = ensure_dir(job.workdir)
    input_name = config.get("extra", {}).get("input_filename", "qchem.inp")
    output_name = config.get("extra", {}).get("output_filename", "qchem.out")
    grad_name = config.get("extra", {}).get("grad_filename", "GRAD")
    input_path = workdir / input_name
    output_path = workdir / output_name
    grad_path = workdir / grad_name
    try:
        tpl = config.get("template_gradient") or config.get("template")
        if not tpl:
            raise ValueError("engine.template (or engine.template_gradient) is required for the Q-Chem engine")
        template_path = Path(tpl)
        mm_charges = job.qmmm.get("mm_charges") if job.qmmm.get("enabled") else None
        rendered = render_template(template_path, job.atoms, mm_charges=mm_charges)
        input_path.write_text(rendered, encoding="utf-8")
        command = config.get("command") or f"qchem -nt {context.threads_per_image} {input_name} {output_name}"
        env = os.environ.copy()
        extra = config.get("extra") or {}
        # Q-Chem writes GRAD (and other scratch) under $QCSCRATCH; point it at this job dir so
        # files stay with qm.out. See engine.extra.qchem_scratch in example YAML.
        if extra.get("qchem_scratch") == "workdir":
            env["QCSCRATCH"] = str(workdir.resolve())
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
        _promote_qchem_grad_file(workdir, grad_name, grad_path)
        result = parse_qchem_output(output_path, grad_path, image_index=job.image_index, selected_root=job.state_spec.get("root"))
        if completed.returncode != 0:
            result.success = False
            result.error_message = result.error_message or f"Q-Chem exited with status {completed.returncode}"
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
            selected_root=job.state_spec.get("root"),
            available_roots=[],
            state_label=None,
            converged=False,
            success=False,
            error_message=str(exc),
            workdir=workdir,
        )


def parse_qchem_output(output_path: Path, grad_path: Path, image_index: int, selected_root: int | None) -> ImageResult:
    text = output_path.read_text(encoding="utf-8")
    energy = _qchem_total_energy_ev(text)
    roots = [
        {"root": int(root), "excitation_energy_ev": float(value)}
        for root, value in _EXCITED_STATE_RE.findall(text)
    ]
    grad_energy, gradient = parse_qchem_grad_file(grad_path)
    if gradient is None:
        gradient = _parse_qchem_cartesian_gradient(text)
    if grad_energy is not None:
        energy = grad_energy
    forces = None if gradient is None else -gradient
    converged = "Thank you very much for using Q-Chem" in text and "failed" not in text.lower()
    success = energy is not None and forces is not None
    state_label = "ground" if selected_root is None else f"root_{selected_root}"
    return ImageResult(
        image_index=image_index,
        energy=energy,
        forces=forces,
        gradient=gradient,
        selected_root=selected_root,
        available_roots=roots,
        state_label=state_label,
        converged=converged,
        success=success,
        error_message=None if success else "failed to parse Q-Chem output",
    )


def parse_qchem_grad_file(path: Path) -> tuple[float | None, np.ndarray | None]:
    if not path.exists():
        return None, None
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 4:
        return None, None
    try:
        energy = hartree_to_ev(float(lines[1].split()[0]))
    except ValueError:
        energy = None
    rows = []
    for line in lines[3:]:
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            rows.append([hartree_per_bohr_to_ev_per_angstrom(float(value)) for value in parts[:3]])
        except ValueError:
            continue
    gradient = np.array(rows, dtype=float) if rows else None
    return energy, gradient
