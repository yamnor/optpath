"""Tests for QM-MM vdW repulsion."""

from __future__ import annotations

import numpy as np
import pytest

from optpath.qmmm.point_charges import load_xyzq
from optpath.qmmm.vdw import get_uff_params, lj_forces


def test_uff_params_known_element() -> None:
    sigma, eps = get_uff_params("O")
    assert sigma > 0
    assert eps > 0


def test_uff_params_case_insensitive() -> None:
    s1, e1 = get_uff_params("O")
    s2, e2 = get_uff_params("o")
    assert s1 == s2 and e1 == e2


def test_uff_params_unknown_element_warns() -> None:
    with pytest.warns(UserWarning, match="No UFF parameters"):
        sigma, eps = get_uff_params("Xx")
    # falls back to carbon
    sigma_c, eps_c = get_uff_params("C")
    assert sigma == sigma_c and eps == eps_c


def test_lj_forces_repulsion_direction() -> None:
    """Force on QM atom should point away from nearby MM atom."""
    qm_pos = np.array([[0.0, 0.0, 0.0]])
    mm_pos = np.array([[2.0, 0.0, 0.0]])   # MM atom along +x
    f = lj_forces(qm_pos, ["C"], mm_pos, ["C"], only_repulsion=True)
    # force on QM atom should have negative x component (pushed away from MM)
    assert f[0, 0] < 0.0
    assert abs(f[0, 1]) < 1e-10
    assert abs(f[0, 2]) < 1e-10


def test_lj_forces_decay_with_distance() -> None:
    """Repulsion should be stronger at shorter distance."""
    qm_pos = np.array([[0.0, 0.0, 0.0]])
    mm_near = np.array([[2.0, 0.0, 0.0]])
    mm_far  = np.array([[5.0, 0.0, 0.0]])
    f_near = lj_forces(qm_pos, ["C"], mm_near, ["O"])
    f_far  = lj_forces(qm_pos, ["C"], mm_far,  ["O"])
    assert abs(f_near[0, 0]) > abs(f_far[0, 0])


def test_lj_forces_zero_at_large_distance() -> None:
    """Repulsion should be negligible at large separation."""
    qm_pos = np.array([[0.0, 0.0, 0.0]])
    mm_pos = np.array([[100.0, 0.0, 0.0]])
    f = lj_forces(qm_pos, ["C"], mm_pos, ["C"])
    assert np.allclose(f, 0.0, atol=1e-10)


def test_load_xyzq_with_symbol(tmp_path) -> None:
    f = tmp_path / "charges.xyzq"
    f.write_text("O  1.0  2.0  3.0  -0.834\nH  1.5  2.5  3.0   0.417\n")
    charges = load_xyzq(f)
    assert len(charges) == 2
    assert charges[0]["symbol"] == "O"
    assert charges[0]["x"] == pytest.approx(1.0)
    assert charges[0]["charge"] == pytest.approx(-0.834)
    assert charges[1]["symbol"] == "H"


def test_load_xyzq_legacy_no_symbol(tmp_path) -> None:
    """Legacy format (x y z charge) should still parse with symbol='X'."""
    f = tmp_path / "charges.xyzq"
    f.write_text("1.0  2.0  3.0  -0.834\n")
    charges = load_xyzq(f)
    assert len(charges) == 1
    assert charges[0]["symbol"] == "X"
    assert charges[0]["charge"] == pytest.approx(-0.834)


def test_load_xyzq_skips_comments_and_blank(tmp_path) -> None:
    f = tmp_path / "charges.xyzq"
    f.write_text("# comment\n\nO  0.0  0.0  0.0  -0.834\n")
    charges = load_xyzq(f)
    assert len(charges) == 1
