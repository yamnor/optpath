from pathlib import Path

import pytest

from optpath.config.schema import RunConfig


def test_parallel_slots_validation(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        RunConfig.model_validate(
            {
                "path": {
                    "initial_xyz": str(tmp_path / "path.xyz"),
                    "nimages": 3,
                },
                "optimizer": {
                    "step_size": 0.1,
                    "max_steps": 10,
                    "grad_tol": 1e-3,
                    "disp_tol": 1e-3,
                    "energy_tol": 1e-5,
                },
                "engine": {"type": "gaussian", "method": "hf"},
                "execution": {
                    "mode": "parallel_images",
                    "max_concurrent_images": 4,
                    "threads_per_image": 8,
                    "total_slots": 16,
                },
                "output": {"run_dir": str(tmp_path / "run")},
            }
        )

