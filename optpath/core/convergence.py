"""Convergence helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class StepMetrics:
    step_index: int
    rms_grad_perp: list[float]
    displacement: list[float]
    energy_delta: list[float | None]
    converged: bool

    @property
    def max_rms_grad_perp(self) -> float:
        return max(self.rms_grad_perp, default=0.0)

    @property
    def avg_rms_grad_perp(self) -> float:
        return float(np.mean(self.rms_grad_perp)) if self.rms_grad_perp else 0.0

    @property
    def max_displacement(self) -> float:
        return max(self.displacement, default=0.0)

    @property
    def max_abs_delta_energy(self) -> float | None:
        finite = [abs(value) for value in self.energy_delta if value is not None]
        return max(finite, default=None)


def build_metrics(
    step_index: int,
    rms_grad_perp: list[float],
    displacement: list[float],
    energy_delta: list[float | None],
    grad_tol: float,
    disp_tol: float,
    energy_tol: float,
) -> StepMetrics:
    max_grad = max(rms_grad_perp, default=0.0)
    max_disp = max(displacement, default=0.0)
    max_energy = max((abs(x) for x in energy_delta if x is not None), default=None)
    energy_ok = True if max_energy is None else max_energy < energy_tol
    converged = max_grad < grad_tol and max_disp < disp_tol and energy_ok
    return StepMetrics(
        step_index=step_index,
        rms_grad_perp=rms_grad_perp,
        displacement=displacement,
        energy_delta=energy_delta,
        converged=converged,
    )

