import numpy as np

from optpath.utils.alignment import kabsch_align_positions


def test_kabsch_align_positions_removes_rigid_rotation_and_translation() -> None:
    reference = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.2, 0.1, 0.0],
            [-0.4, 0.8, 0.3],
            [0.2, -0.3, 1.1],
        ],
        dtype=float,
    )
    rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    translation = np.array([2.5, -1.1, 0.7], dtype=float)
    mobile = reference @ rotation.T + translation

    aligned = kabsch_align_positions(mobile, reference)

    assert np.allclose(aligned, reference, atol=1e-10)


def test_kabsch_align_positions_single_atom_matches_reference_centroid() -> None:
    mobile = np.array([[3.0, -2.0, 1.0]], dtype=float)
    reference = np.array([[0.5, 0.5, 0.5]], dtype=float)

    aligned = kabsch_align_positions(mobile, reference)

    assert np.allclose(aligned, reference, atol=1e-12)
