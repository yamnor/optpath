"""Rigid-body alignment helpers."""

from __future__ import annotations

import numpy as np


def kabsch_align_positions(mobile: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Return *mobile* rigidly aligned onto *reference* using the Kabsch algorithm."""
    mobile = np.asarray(mobile, dtype=float)
    reference = np.asarray(reference, dtype=float)

    if mobile.shape != reference.shape:
        raise ValueError(f"mobile/reference shapes must match; got {mobile.shape} vs {reference.shape}")
    if mobile.ndim != 2 or mobile.shape[1] != 3:
        raise ValueError(f"expected positions shaped (N, 3); got {mobile.shape}")
    if mobile.shape[0] == 0:
        return mobile.copy()

    mobile_centroid = mobile.mean(axis=0)
    reference_centroid = reference.mean(axis=0)
    mobile_centered = mobile - mobile_centroid
    reference_centered = reference - reference_centroid

    if mobile.shape[0] == 1:
        return mobile_centered + reference_centroid

    covariance = mobile_centered.T @ reference_centered
    u, _, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        vt[-1, :] *= -1.0
        rotation = u @ vt
    return mobile_centered @ rotation + reference_centroid
