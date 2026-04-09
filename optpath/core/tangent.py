"""Tangent and perpendicular gradient helpers."""

from __future__ import annotations

import math

import numpy as np


def _unit(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0.0:
        return np.zeros_like(vec)
    return vec / norm


def bisection_tangents(points: np.ndarray) -> np.ndarray:
    nimages = len(points)
    tangents = np.zeros_like(points)
    if nimages == 1:
        return tangents
    tangents[0] = _unit(points[1] - points[0])
    tangents[-1] = _unit(points[-1] - points[-2])
    for idx in range(1, nimages - 1):
        fw = _unit(points[idx + 1] - points[idx])
        bw = _unit(points[idx] - points[idx - 1])
        tangents[idx] = _unit(fw + bw)
    return tangents


def perpendicular_gradient(gradient: np.ndarray, tangent: np.ndarray) -> np.ndarray:
    tangent = _unit(tangent)
    return gradient - np.dot(gradient, tangent) * tangent


def rms_norm(vector: np.ndarray) -> float:
    if vector.size == 0:
        return 0.0
    return float(np.linalg.norm(vector) / math.sqrt(vector.size))

