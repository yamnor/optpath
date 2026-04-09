"""Reparameterization functions."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline


def arc_lengths(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.array([], dtype=float)
    s = np.zeros(len(points), dtype=float)
    if len(points) == 1:
        return s
    diffs = np.diff(points, axis=0)
    seg = np.linalg.norm(diffs, axis=1)
    s[1:] = np.cumsum(seg)
    if s[-1] > 0.0:
        s /= s[-1]
    return s


def reparameterize(points: np.ndarray, method: str) -> np.ndarray:
    if len(points) <= 2:
        return points.copy()
    s = arc_lengths(points)
    targets = np.linspace(0.0, 1.0, len(points))
    result = np.empty_like(points)
    for dim in range(points.shape[1]):
        values = points[:, dim]
        if method == "cubic" and len(points) >= 3 and np.unique(s).size == len(s):
            spline = CubicSpline(s, values, bc_type="natural")
            result[:, dim] = spline(targets)
        else:
            result[:, dim] = np.interp(targets, s, values)
    return result

