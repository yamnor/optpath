"""Integration tests for QM/MM functionality."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from optpath.config.schema import RunConfig
from optpath.core.string_optimizer import StringOptimizer
from optpath.engines.results import ImageResult
from optpath.io.templates import render_charges_block, render_template
from optpath.io.xyz import write_xyz_images


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class ZeroForceEngine:
    """Engine that always returns zero forces (energy = 0)."""

    def evaluate(self, jobs, context):
        results = []
        for job in jobs:
            forces = np.zeros((len(job.atoms), 3), dtype=float)
            results.append(
                ImageResult(
                    image_index=job.image_index,
                    energy=0.0,
                    forces=forces,
                    gradient=-forces,
                    selected_root=None,
                    available_roots=[],
                    state_label="ground",
                    converged=True,
                    success=True,
                    error_message=None,
                )
            )
        return results


def _write_path(path: Path) -> None:
    write_xyz_images(
        path,
        [
            Atoms("C", positions=[[0.0, 0.0, 0.0]]),
            Atoms("C", positions=[[0.5, 0.0, 0.0]]),
            Atoms("C", positions=[[1.0, 0.0, 0.0]]),
        ],
    )


def _qmmm_config(tmp_path: Path, mm_file: Path, *, vdw_repulsion: bool) -> RunConfig:
    xyz = tmp_path / "path.xyz"
    _write_path(xyz)
    return RunConfig.model_validate(
        {
            "path": {"initial_xyz": str(xyz), "nimages": 3, "coordinate_mode": "qm_region_only"},
            "optimizer": {"step_size": 0.05, "max_steps": 1, "grad_tol": 1e-3, "disp_tol": 1e-3, "energy_tol": 1e-6},
            "engine": {"type": "pyscf", "method": "hf"},
            "execution": {"mode": "serial", "max_concurrent_images": 1, "threads_per_image": 1},
            "qmmm": {
                "enabled": True,
                "qm_atoms": [0],
                "mm_charges_file": str(mm_file),
                "update_region_only": True,
                "vdw_repulsion": vdw_repulsion,
            },
            "output": {"run_dir": str(tmp_path / "run")},
        }
    )


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------

def test_render_charges_block_format() -> None:
    charges = [{"x": 1.0, "y": 2.0, "z": 3.0, "charge": -0.5}]
    block = render_charges_block(charges)
    parts = block.split()
    assert len(parts) == 4
    assert float(parts[0]) == pytest.approx(1.0)
    assert float(parts[3]) == pytest.approx(-0.5)


def test_render_template_injects_charges(tmp_path: Path) -> None:
    tpl = tmp_path / "gau.grad"
    tpl.write_text("#P HF Force Charge\n\ntitle\n\n0 1\n__geom__\n\n__charges__\n\n")
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    charges = [{"x": 5.0, "y": 0.0, "z": 0.0, "charge": -1.0}]
    rendered = render_template(tpl, atoms, mm_charges=charges)
    assert "5.00000000" in rendered
    assert "-1.000000" in rendered


def test_render_template_no_charges_placeholder(tmp_path: Path) -> None:
    """Templates without __charges__ are not modified when charges are given."""
    tpl = tmp_path / "gau.grad"
    tpl.write_text("0 1\n__geom__\n\n")
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    charges = [{"x": 5.0, "y": 0.0, "z": 0.0, "charge": -1.0}]
    rendered = render_template(tpl, atoms, mm_charges=charges)
    assert "__charges__" not in rendered
    assert "5.00000000" not in rendered


def test_render_template_empty_charges_clears_placeholder(tmp_path: Path) -> None:
    tpl = tmp_path / "gau.grad"
    tpl.write_text("0 1\n__geom__\n\n__charges__\n")
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    rendered = render_template(tpl, atoms, mm_charges=None)
    assert "__charges__" not in rendered


# ---------------------------------------------------------------------------
# QM/MM optimizer integration
# ---------------------------------------------------------------------------

def test_vdw_repulsion_moves_middle_image(tmp_path: Path, monkeypatch) -> None:
    """With zero engine forces, vdW should produce non-zero displacement in the step."""
    from optpath.core import string_optimizer

    mm_file = tmp_path / "mm.xyzq"
    mm_file.write_text("O  3.0  0.0  0.0  -0.834\n")

    config = _qmmm_config(tmp_path, mm_file, vdw_repulsion=True)
    monkeypatch.setattr(string_optimizer, "build_engine", lambda _: ZeroForceEngine())
    opt = StringOptimizer(config)

    # displacement is measured before reparameterization, so it captures the vdW step
    _, metrics, _ = opt.step(0)
    assert metrics.max_displacement > 0, "vdW forces should produce non-zero displacement"


def test_no_vdw_zero_displacement(tmp_path: Path, monkeypatch) -> None:
    """With vdw_repulsion=False and zero engine forces, positions should not change."""
    from optpath.core import string_optimizer

    mm_file = tmp_path / "mm.xyzq"
    mm_file.write_text("O  3.0  0.0  0.0  -0.834\n")

    config = _qmmm_config(tmp_path, mm_file, vdw_repulsion=False)
    monkeypatch.setattr(string_optimizer, "build_engine", lambda _: ZeroForceEngine())
    opt = StringOptimizer(config)

    pos_before = opt.band.images[1].get_positions().copy()
    opt.step(0)
    pos_after = opt.band.images[1].get_positions()

    assert np.allclose(pos_before, pos_after, atol=1e-12), "Positions should not change without vdW"


def test_vdw_repulsion_stronger_nearer(tmp_path: Path, monkeypatch) -> None:
    """Displacement should be larger when MM atom is closer."""
    from optpath.core import string_optimizer

    def run_with_mm_at(x_mm: float, subdir: str) -> float:
        sub = tmp_path / subdir
        sub.mkdir()
        mm_file = sub / "mm.xyzq"
        mm_file.write_text(f"O  {x_mm}  0.0  0.0  -0.834\n")
        config = _qmmm_config(sub, mm_file, vdw_repulsion=True)
        monkeypatch.setattr(string_optimizer, "build_engine", lambda _: ZeroForceEngine())
        opt = StringOptimizer(config)
        _, metrics, _ = opt.step(0)
        return metrics.max_displacement

    disp_near = run_with_mm_at(1.5, "near")
    disp_far = run_with_mm_at(4.0, "far")
    assert disp_near > disp_far


# ---------------------------------------------------------------------------
# Template validation errors
# ---------------------------------------------------------------------------

def test_gaussian_worker_raises_without_template(tmp_path: Path) -> None:
    """Gaussian worker should raise ValueError when no template is configured."""
    from optpath.engines.gaussian import _gaussian_worker
    from optpath.engines.results import ImageJob, RunContext

    job = ImageJob(
        image_index=0,
        atoms=Atoms("H", positions=[[0.0, 0.0, 0.0]]),
        state_spec={"kind": "ground", "method": "hf", "root": None, "nroots": None, "multiplicity": 1, "state_follow": False},
        workdir=tmp_path / "work",
        scratch_dir=None,
        previous_metadata={},
        qmmm={"enabled": False},
        active_mask=np.ones((1, 3), dtype=bool),
    )
    context = RunContext(
        execution_mode="serial",
        max_concurrent_images=1,
        threads_per_image=1,
        total_slots=None,
        scratch_root=None,
        run_dir=tmp_path,
        step_index=0,
    )
    result = _gaussian_worker(job, {"method": "hf"}, context)
    assert not result.success
    assert "template" in (result.error_message or "").lower()


def test_qchem_worker_raises_without_template(tmp_path: Path) -> None:
    """Q-Chem worker should raise ValueError when no template is configured."""
    from optpath.engines.qchem import _qchem_worker
    from optpath.engines.results import ImageJob, RunContext

    job = ImageJob(
        image_index=0,
        atoms=Atoms("H", positions=[[0.0, 0.0, 0.0]]),
        state_spec={"kind": "ground", "method": "hf", "root": None, "nroots": None, "multiplicity": 1, "state_follow": False},
        workdir=tmp_path / "work",
        scratch_dir=None,
        previous_metadata={},
        qmmm={"enabled": False},
        active_mask=np.ones((1, 3), dtype=bool),
    )
    context = RunContext(
        execution_mode="serial",
        max_concurrent_images=1,
        threads_per_image=1,
        total_slots=None,
        scratch_root=None,
        run_dir=tmp_path,
        step_index=0,
    )
    result = _qchem_worker(job, {"method": "hf"}, context)
    assert not result.success
    assert "template" in (result.error_message or "").lower()
