"""String optimizer implementation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from optpath.config.schema import RunConfig
from optpath.core.band import ImageBand
from optpath.core.checkpoint import save_checkpoint
from optpath.core.convergence import StepMetrics, build_metrics
from optpath.core.state_tracking import StateTracker
from optpath.core.tangent import bisection_tangents, perpendicular_gradient, rms_norm
from optpath.engines.base import build_engine
from optpath.engines.results import ImageJob, ImageResult, RunContext, TrackedState
from optpath.io.logs import write_summary, write_table
from optpath.io.trajectory import write_band_trajectory
from optpath.qmmm.point_charges import load_xyzq
from optpath.qmmm.vdw import lj_forces
from optpath.utils.constants import EXCITED_STATE_METHODS
from optpath.utils.errors import OptimizationAbort
from optpath.utils.filesystem import ensure_dir


@dataclass
class OptimizationResult:
    band: ImageBand
    results: list[ImageResult]
    metrics: StepMetrics | None
    converged: bool
    last_successful_step: int


class StringOptimizer:
    def __init__(self, config: RunConfig) -> None:
        self.config = config
        qm_atoms = config.qmmm.qm_atoms if config.path.coordinate_mode == "qm_region_only" else None
        self.band = ImageBand.from_xyz(
            config.path.initial_xyz,
            nimages=config.path.nimages,
            freeze_endpoints=config.path.freeze_endpoints,
            coordinate_mode=config.path.coordinate_mode,
            qm_atom_indices=qm_atoms,
        )
        self.engine = build_engine(config)
        self.state_tracker = StateTracker(enabled=config.engine.method in EXCITED_STATE_METHODS)
        self.run_dir = ensure_dir(config.output.run_dir)
        self.checkpoint_dir = ensure_dir(self.run_dir / "checkpoints")
        self.trajectory_dir = ensure_dir(self.run_dir / "trajectories")
        self.iteration_dir = self.run_dir
        self.summary_path = self.run_dir / "summary.log"
        self.table_path = self.run_dir / "table.csv"
        self.config_snapshot_path = self.run_dir / "config.snapshot.yaml"
        self.previous_results: list[ImageResult] | None = None
        self.last_successful_step = -1
        self.qmmm_payload = self._build_qmmm_payload()

    def _build_qmmm_payload(self) -> dict:
        if not self.config.qmmm.enabled:
            return {"enabled": False}
        charges = []
        if self.config.qmmm.mm_charges_file is not None:
            charges = load_xyzq(self.config.qmmm.mm_charges_file)
        return {
            "enabled": True,
            "qm_atoms": self.config.qmmm.qm_atoms,
            "mm_charges": charges,
            "update_region_only": self.config.qmmm.update_region_only,
            "vdw_repulsion": self.config.qmmm.vdw_repulsion,
        }

    def _state_spec(self) -> dict:
        return {
            "kind": "excited" if self.config.engine.method in EXCITED_STATE_METHODS else "ground",
            "method": self.config.engine.method,
            "root": self.config.engine.root,
            "nroots": self.config.engine.nroots,
            "multiplicity": self.config.engine.multiplicity,
            "state_follow": self.config.engine.state_follow,
        }

    def _context(self, step_index: int) -> RunContext:
        return RunContext(
            execution_mode=self.config.execution.mode,
            max_concurrent_images=self.config.execution.max_concurrent_images,
            threads_per_image=self.config.execution.threads_per_image,
            total_slots=self.config.execution.total_slots,
            scratch_root=self.config.execution.scratch_root,
            run_dir=self.run_dir,
            step_index=step_index,
        )

    def _jobs(self, step_index: int) -> list[ImageJob]:
        jobs: list[ImageJob] = []
        state_spec = self._state_spec()
        previous_map = {result.image_index: result for result in (self.previous_results or [])}
        for image_index in self.band.get_free_image_indices():
            workdir = ensure_dir(self.run_dir / f"iter_{step_index:04d}" / f"image_{image_index:02d}")
            scratch_dir = (
                None
                if self.config.execution.scratch_root is None
                else self.config.execution.scratch_root / f"step_{step_index:04d}" / f"image_{image_index:02d}"
            )
            previous_metadata = previous_map.get(image_index).metadata if image_index in previous_map else {}
            jobs.append(
                ImageJob(
                    image_index=image_index,
                    atoms=self.band.images[image_index].copy(),
                    state_spec=state_spec.copy(),
                    workdir=workdir,
                    scratch_dir=scratch_dir,
                    previous_metadata=dict(previous_metadata),
                    qmmm=self.qmmm_payload,
                    active_mask=self.band.dof_mask.copy(),
                )
            )
        return jobs

    def evaluate_band(self, step_index: int) -> list[ImageResult]:
        jobs = self._jobs(step_index)
        if not jobs:
            return []
        return self.engine.evaluate(jobs, self._context(step_index))

    def track_states(self, results: list[ImageResult]) -> list[TrackedState]:
        if self._state_spec()["kind"] == "ground":
            return [
                TrackedState(
                    image_index=result.image_index,
                    selected_root=result.selected_root,
                    state_label=result.state_label,
                    warnings=list(result.warnings),
                )
                for result in results
            ]
        return self.state_tracker.update(self.previous_results, results)

    def _active_points(self) -> np.ndarray:
        return np.vstack([self.band.get_active_coordinate_vector(i) for i in range(self.band.nimages)])

    def _perp_metrics(self, gradients: dict[int, np.ndarray]) -> dict[int, float]:
        tangents = bisection_tangents(self._active_points())
        values: dict[int, float] = {}
        for image_index, gradient in gradients.items():
            perp = perpendicular_gradient(gradient, tangents[image_index])
            values[image_index] = rms_norm(perp)
        return values

    def _energy_delta(self, results: list[ImageResult]) -> list[float | None]:
        if self.previous_results is None:
            return [None for _ in results]
        previous_map = {result.image_index: result.energy for result in self.previous_results}
        values: list[float | None] = []
        for result in results:
            current = result.energy
            previous = previous_map.get(result.image_index)
            values.append(None if current is None or previous is None else current - previous)
        return values

    def _augment_results(self, results: list[ImageResult]) -> list[ImageResult]:
        existing = {result.image_index: result for result in results}
        full_results: list[ImageResult] = []
        for idx in range(self.band.nimages):
            if idx in existing:
                full_results.append(existing[idx])
            else:
                full_results.append(
                    ImageResult(
                        image_index=idx,
                        energy=None,
                        forces=None,
                        gradient=None,
                        selected_root=None,
                        available_roots=[],
                        state_label=None,
                        converged=True,
                        success=True,
                        error_message=None,
                        metadata={"fixed_endpoint": True},
                    )
                )
        return full_results

    def save_outputs(
        self,
        step_index: int,
        results: list[ImageResult],
        metrics: StepMetrics,
        tracked_states: list[TrackedState],
        perp: dict[int, float],
        displacement: dict[int, float],
    ) -> None:
        write_summary(self.summary_path, metrics, self.band.nimages, len(self.band.get_free_image_indices()), tracked_states)
        write_table(self.table_path, step_index, results, perp, displacement)
        if step_index % self.config.output.write_xyz_every == 0:
            write_band_trajectory(self.trajectory_dir / f"band_{step_index:04d}.xyz", self.band.images)
        if step_index % self.config.output.checkpoint_every == 0:
            save_checkpoint(
                self.checkpoint_dir,
                f"step_{step_index:04d}",
                self.band,
                results,
                metrics,
                tracked_states,
                self.config_snapshot_path.name,
            )

    def step(self, step_index: int) -> tuple[list[ImageResult], StepMetrics, list[TrackedState]]:
        results = self._augment_results(self.evaluate_band(step_index))
        failed = [r for r in results if not r.success]
        if failed:
            save_checkpoint(
                self.checkpoint_dir,
                f"step_{step_index:04d}.diagnostics",
                self.band,
                results,
                None,
                None,
                self.config_snapshot_path.name,
                diagnostics=True,
            )
            indices = [r.image_index for r in failed]
            raise OptimizationAbort(f"step {step_index} aborted: image(s) {indices} failed — {failed[0].error_message}")
        for result in results:
            if not result.converged:
                result.warnings.append(f"SCF not converged at step {step_index}, image {result.image_index}")
        tracked = self.track_states(results)
        unconverged = [
            result
            for result in results
            if not result.converged and result.image_index in self.band.get_free_image_indices()
        ]
        if unconverged:
            save_checkpoint(
                self.checkpoint_dir,
                f"step_{step_index:04d}.diagnostics",
                self.band,
                results,
                None,
                tracked,
                self.config_snapshot_path.name,
                diagnostics=True,
            )
            indices = [r.image_index for r in unconverged]
            raise OptimizationAbort(f"step {step_index} aborted: image(s) {indices} SCF did not converge")
        free_indices = self.band.get_free_image_indices()
        x_before = {idx: self.band.get_active_coordinate_vector(idx).copy() for idx in free_indices}
        # Pre-compute vdW correction if QM/MM is enabled with vdw_repulsion
        vdw_payload = self.qmmm_payload
        use_vdw = (
            vdw_payload.get("enabled")
            and vdw_payload.get("vdw_repulsion")
            and vdw_payload.get("mm_charges")
        )
        mm_positions = None
        mm_symbols = None
        if use_vdw:
            mm_charges = vdw_payload["mm_charges"]
            mm_positions = np.array([[c["x"], c["y"], c["z"]] for c in mm_charges])
            mm_symbols = [c["symbol"] for c in mm_charges]

        for result in results:
            if result.image_index not in x_before:
                continue
            if result.forces is None:
                continue
            current = self.band.get_active_coordinate_vector(result.image_index)
            active_forces = result.forces[self.band.dof_mask]
            if use_vdw:
                image = self.band.images[result.image_index]
                qm_pos = image.get_positions()[self.band.dof_mask.reshape(-1, 3)[:, 0]]
                qm_sym = [image.get_chemical_symbols()[i] for i in range(len(image)) if self.band.dof_mask[i, 0]]
                vdw_f = lj_forces(qm_pos, qm_sym, mm_positions, mm_symbols)
                active_forces = active_forces + vdw_f.ravel()
            self.band.set_active_coordinate_vector(
                result.image_index,
                current + self.config.optimizer.step_size * active_forces,
            )
        displacement = {}
        for idx in free_indices:
            x_after = self.band.get_active_coordinate_vector(idx)
            disp = np.linalg.norm(x_after - x_before[idx]) / np.sqrt(x_after.size)
            displacement[idx] = float(disp)
        self.band.reparameterize(self.config.path.reparameterization)
        gradients = {
            result.image_index: result.gradient[self.band.dof_mask]
            for result in results
            if result.gradient is not None and result.image_index in free_indices
        }
        perp = self._perp_metrics(gradients)
        energy_delta = self._energy_delta([result for result in results if result.image_index in free_indices])
        metrics = build_metrics(
            step_index=step_index,
            rms_grad_perp=[perp[idx] for idx in free_indices],
            displacement=[displacement[idx] for idx in free_indices],
            energy_delta=energy_delta,
            grad_tol=self.config.optimizer.grad_tol,
            disp_tol=self.config.optimizer.disp_tol,
            energy_tol=self.config.optimizer.energy_tol,
        )
        self.band.iteration = step_index
        self.save_outputs(step_index, results, metrics, tracked, perp, displacement)
        self.previous_results = results
        self.last_successful_step = step_index
        return results, metrics, tracked

    def run(self) -> OptimizationResult:
        return self.run_from_step(start_step=0)

    def restore_checkpoint(self, checkpoint_data: dict) -> int:
        meta = checkpoint_data["meta"]
        self.band.iteration = int(meta.get("iteration", -1))
        self.last_successful_step = self.band.iteration
        restored_results: list[ImageResult] = []
        for result in meta.get("results", []):
            restored_results.append(
                ImageResult(
                    image_index=int(result["image_index"]),
                    energy=result.get("energy"),
                    forces=None,
                    gradient=None,
                    selected_root=result.get("selected_root"),
                    available_roots=result.get("metadata", {}).get("available_roots", []),
                    state_label=result.get("state_label"),
                    converged=bool(result.get("converged", True)),
                    success=bool(result.get("success", True)),
                    error_message=result.get("error_message"),
                    metadata=dict(result.get("metadata", {})),
                    warnings=list(result.get("warnings", [])),
                )
            )
        self.previous_results = restored_results or None
        return self.band.iteration + 1

    def run_from_step(self, start_step: int) -> OptimizationResult:
        self.config_snapshot_path.write_text(
            yaml.safe_dump(self.config.model_dump(mode="json"), sort_keys=False),
            encoding="utf-8",
        )
        final_results: list[ImageResult] = []
        final_metrics: StepMetrics | None = None
        converged = False
        for step_index in range(start_step, self.config.optimizer.max_steps):
            final_results, final_metrics, _ = self.step(step_index)
            if final_metrics.converged:
                converged = True
                break
        return OptimizationResult(
            band=self.band.copy(),
            results=final_results,
            metrics=final_metrics,
            converged=converged,
            last_successful_step=self.last_successful_step,
        )
