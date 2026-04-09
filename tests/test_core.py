from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from optpath.core.band import ImageBand
from optpath.core.checkpoint import load_checkpoint, save_checkpoint
from optpath.core.convergence import build_metrics
from optpath.core.tangent import bisection_tangents, perpendicular_gradient, rms_norm
from optpath.engines.results import ImageResult


def make_band() -> ImageBand:
    images = [
        Atoms("H", positions=[[0.0, 0.0, 0.0]]),
        Atoms("H", positions=[[1.0, 0.0, 0.0]]),
        Atoms("H", positions=[[2.0, 0.0, 0.0]]),
    ]
    return ImageBand(images=images, fixed_images=[True, False, True], dof_mask=np.ones((1, 3), dtype=bool))


def test_reparameterize_keeps_endpoints() -> None:
    band = make_band()
    band.images[1].set_positions([[0.2, 0.7, 0.0]])
    start = band.images[0].get_positions().copy()
    end = band.images[-1].get_positions().copy()
    band.reparameterize("cubic")
    assert np.allclose(band.images[0].get_positions(), start)
    assert np.allclose(band.images[-1].get_positions(), end)


def test_bisection_tangent_linear_path() -> None:
    points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    tangents = bisection_tangents(points)
    assert np.allclose(tangents[1], np.array([1.0, 0.0]))


def test_perpendicular_gradient_and_rms() -> None:
    tangent = np.array([1.0, 0.0, 0.0])
    grad = np.array([2.0, 3.0, 4.0])
    perp = perpendicular_gradient(grad, tangent)
    assert np.allclose(perp, np.array([0.0, 3.0, 4.0]))
    assert np.isclose(rms_norm(perp), 5.0 / np.sqrt(3.0))


def test_first_step_energy_delta_skip() -> None:
    metrics = build_metrics(
        step_index=0,
        rms_grad_perp=[1e-5],
        displacement=[1e-5],
        energy_delta=[None],
        grad_tol=1e-3,
        disp_tol=1e-3,
        energy_tol=1e-6,
    )
    assert metrics.max_abs_delta_energy is None
    assert metrics.converged


def test_checkpoint_dof_mask_mismatch_raises(tmp_path: Path) -> None:
    band = make_band()
    results = [
        ImageResult(
            image_index=0,
            energy=0.0,
            forces=np.zeros((1, 3)),
            gradient=np.zeros((1, 3)),
            selected_root=None,
            available_roots=[],
            state_label="ground",
            converged=True,
            success=True,
            error_message=None,
        )
    ]
    save_checkpoint(tmp_path, "step_0000", band, results, None, None, "config.yaml")
    altered = ImageBand(
        images=[Atoms("H", positions=[[0.0, 0.0, 0.0]]), Atoms("H", positions=[[1.0, 0.0, 0.0]]), Atoms("H", positions=[[2.0, 0.0, 0.0]])],
        fixed_images=[True, False, True],
        dof_mask=np.array([[True, False, True]], dtype=bool),
    )
    with pytest.raises(ValueError):
        load_checkpoint(tmp_path / "step_0000.meta.json", tmp_path / "step_0000.arrays.npz", altered)
