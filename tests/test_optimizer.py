from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from optpath.config.schema import RunConfig
from optpath.core.string_optimizer import StringOptimizer
from optpath.engines.results import ImageResult, TrackedState
from optpath.io.xyz import write_xyz_images


class DummyEngine:
    def evaluate(self, jobs, context):
        results = []
        for job in jobs:
            forces = np.zeros((len(job.atoms), 3), dtype=float)
            gradient = -forces
            results.append(
                ImageResult(
                    image_index=job.image_index,
                    energy=0.0,
                    forces=forces,
                    gradient=gradient,
                    selected_root=None,
                    available_roots=[],
                    state_label="ground",
                    converged=True,
                    success=True,
                    error_message=None,
                )
            )
        return results


class FailingEngine(DummyEngine):
    def evaluate(self, jobs, context):
        results = super().evaluate(jobs, context)
        results[0].success = False
        results[0].converged = False
        results[0].error_message = "failed"
        return results


class NotConvergedEngine(DummyEngine):
    """Returns success=True but converged=False (e.g. SCF not converged)."""

    def evaluate(self, jobs, context):
        results = super().evaluate(jobs, context)
        for r in results:
            r.converged = False
        return results


def build_config(tmp_path: Path) -> RunConfig:
    xyz = tmp_path / "path.xyz"
    write_xyz_images(
        xyz,
        [
            Atoms("H", positions=[[0.0, 0.0, 0.0]]),
            Atoms("H", positions=[[0.5, 0.0, 0.0]]),
            Atoms("H", positions=[[1.0, 0.0, 0.0]]),
        ],
    )
    return RunConfig.model_validate(
        {
            "path": {"initial_xyz": str(xyz), "nimages": 3},
            "optimizer": {"step_size": 0.1, "max_steps": 2, "grad_tol": 1e-3, "disp_tol": 1e-3, "energy_tol": 1e-6},
            "engine": {"type": "gaussian", "method": "hf"},
            "execution": {"mode": "serial", "max_concurrent_images": 1, "threads_per_image": 1},
            "output": {"run_dir": str(tmp_path / "run")},
        }
    )


def test_optimizer_with_dummy_engine(tmp_path: Path, monkeypatch) -> None:
    from optpath.core import string_optimizer

    monkeypatch.setattr(string_optimizer, "build_engine", lambda config: DummyEngine())
    optimizer = StringOptimizer(build_config(tmp_path))
    result = optimizer.run()
    assert result.converged
    assert (tmp_path / "run" / "summary.log").exists()
    assert (tmp_path / "run" / "table.csv").exists()


def test_failed_image_aborts(tmp_path: Path, monkeypatch) -> None:
    from optpath.core import string_optimizer
    from optpath.utils.errors import OptimizationAbort

    monkeypatch.setattr(string_optimizer, "build_engine", lambda config: FailingEngine())
    optimizer = StringOptimizer(build_config(tmp_path))
    with pytest.raises(OptimizationAbort):
        optimizer.run()
    diagnostics = list((tmp_path / "run" / "checkpoints").glob("*.diagnostics.meta.json"))
    assert diagnostics


def test_scf_not_converged_aborts_with_diagnostics(tmp_path: Path, monkeypatch) -> None:
    from optpath.core import string_optimizer
    from optpath.core.checkpoint import load_checkpoint
    from optpath.utils.errors import OptimizationAbort

    monkeypatch.setattr(string_optimizer, "build_engine", lambda config: NotConvergedEngine())
    optimizer = StringOptimizer(build_config(tmp_path))
    with pytest.raises(OptimizationAbort, match="SCF did not converge"):
        optimizer.run()

    checkpoint_dir = tmp_path / "run" / "checkpoints"
    diagnostics = sorted(checkpoint_dir.glob("*.diagnostics.meta.json"))
    assert diagnostics

    latest_meta = diagnostics[-1]
    latest_arrays = latest_meta.with_suffix("").with_suffix(".arrays.npz")
    restored = StringOptimizer(build_config(tmp_path))
    checkpoint_data = load_checkpoint(latest_meta, latest_arrays, restored.band)
    free_results = [
        result
        for result in checkpoint_data["meta"]["results"]
        if result["image_index"] in restored.band.get_free_image_indices()
    ]
    assert all(result["warnings"] for result in free_results)


def test_ground_state_track_states_preserve_warnings(tmp_path: Path, monkeypatch) -> None:
    from optpath.core import string_optimizer

    monkeypatch.setattr(string_optimizer, "build_engine", lambda config: DummyEngine())
    optimizer = StringOptimizer(build_config(tmp_path))
    tracked = optimizer.track_states(
        [
            ImageResult(
                image_index=1,
                energy=0.0,
                forces=np.zeros((1, 3), dtype=float),
                gradient=np.zeros((1, 3), dtype=float),
                selected_root=None,
                available_roots=[],
                state_label="ground",
                converged=False,
                success=True,
                error_message=None,
                warnings=["SCF not converged"],
            )
        ]
    )
    assert tracked == [TrackedState(image_index=1, selected_root=None, state_label="ground", warnings=["SCF not converged"])]


def test_restore_checkpoint_sets_start_step(tmp_path: Path, monkeypatch) -> None:
    from optpath.core import string_optimizer
    from optpath.core.checkpoint import load_checkpoint

    monkeypatch.setattr(string_optimizer, "build_engine", lambda config: DummyEngine())
    optimizer = StringOptimizer(build_config(tmp_path))
    optimizer.run()

    restored = StringOptimizer(build_config(tmp_path))
    checkpoint_dir = tmp_path / "run" / "checkpoints"
    latest_meta = sorted(checkpoint_dir.glob("step_*.meta.json"))[-1]
    latest_arrays = latest_meta.with_suffix("").with_suffix(".arrays.npz")
    checkpoint_data = load_checkpoint(latest_meta, latest_arrays, restored.band)
    start_step = restored.restore_checkpoint(checkpoint_data)

    assert start_step == restored.band.iteration + 1
    assert restored.previous_results is not None
    assert restored.last_successful_step == restored.band.iteration
