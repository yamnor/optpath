"""Tests for Z-matrix interpolation."""

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms


from optpath.utils.zmat_interp import (
    get_zmatrix_string,
    interpolate_zmat,
)


def water() -> Atoms:
    return Atoms(
        "OHH",
        positions=[
            [0.000,  0.000, 0.000],
            [0.757,  0.586, 0.000],
            [-0.757, 0.586, 0.000],
        ],
    )


def water_distorted() -> Atoms:
    """Water with slightly different bond lengths and angle."""
    return Atoms(
        "OHH",
        positions=[
            [0.000,  0.000, 0.000],
            [0.850,  0.600, 0.000],
            [-0.850, 0.600, 0.000],
        ],
    )


def test_endpoints_preserved() -> None:
    a1 = water()
    a2 = water_distorted()
    images = interpolate_zmat(a1, a2, nimages=5)
    assert len(images) == 5
    assert np.allclose(images[0].get_positions(), a1.get_positions(), atol=1e-6)
    assert np.allclose(images[-1].get_positions(), a2.get_positions(), atol=1e-6)


def test_midpoint_is_between_endpoints() -> None:
    a1 = water()
    a2 = water_distorted()
    images = interpolate_zmat(a1, a2, nimages=3)
    mid = images[1].get_positions()
    # midpoint should be between the two endpoints
    # Internal-coordinate interpolation followed by Kabsch alignment can induce a
    # tiny rigid-body shift even for atoms that do not move in the endpoints.
    # Allow a small floor for near-zero endpoint displacements.
    d_mid = np.linalg.norm(mid - a1.get_positions(), axis=1)
    d_end = np.linalg.norm(a2.get_positions() - a1.get_positions(), axis=1)
    assert np.all(d_mid <= np.maximum(d_end, 1e-2))


def test_nimages_2_returns_endpoints_only() -> None:
    images = interpolate_zmat(water(), water_distorted(), nimages=2)
    assert len(images) == 2


def test_nimages_less_than_2_raises() -> None:
    with pytest.raises(ValueError, match="nimages"):
        interpolate_zmat(water(), water_distorted(), nimages=1)


def test_mismatched_atoms_raises() -> None:
    a1 = Atoms("OHH", positions=[[0, 0, 0], [1, 0, 0], [-1, 0, 0]])
    a2 = Atoms("NHH", positions=[[0, 0, 0], [1, 0, 0], [-1, 0, 0]])
    with pytest.raises(ValueError, match="same atoms"):
        interpolate_zmat(a1, a2, nimages=3)


def test_dihedral_wrap() -> None:
    """Dihedral interpolation should take the short arc around ±180°."""
    from optpath.utils.zmat_interp import _interp_angle

    # 170° -> -170° : shortest arc is 20°, midpoint = ±180°
    mid = _interp_angle(np.array([170.0]), np.array([-170.0]), 0.5)
    assert abs(abs(mid[0]) - 180.0) < 1e-9

    # 170° -> -170° : at t=0.25 should be 175°
    q = _interp_angle(np.array([170.0]), np.array([-170.0]), 0.25)
    assert np.isclose(q[0], 175.0)


def test_get_zmatrix_string() -> None:
    result = get_zmatrix_string(water())
    assert "O" in result
    assert "H" in result
    lines = [l for l in result.splitlines() if l.strip()]
    assert len(lines) == 3  # 3 atoms


def test_interp_cli(tmp_path: Path) -> None:
    from optpath.io.xyz import write_xyz_images, read_xyz_images
    from optpath.cli import build_parser

    r_path = tmp_path / "reactant.xyz"
    p_path = tmp_path / "product.xyz"
    o_path = tmp_path / "path.xyz"

    write_xyz_images(r_path, [water()])
    write_xyz_images(p_path, [water_distorted()])

    parser = build_parser()
    args = parser.parse_args(["interp", str(r_path), str(p_path), str(o_path), "--nimages", "5"])
    ret = args.func(args)
    assert ret == 0
    images = read_xyz_images(o_path)
    assert len(images) == 5
