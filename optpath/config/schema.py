"""Pydantic models for optpath configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from optpath.utils.constants import EXCITED_STATE_METHODS


class PathConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    initial_xyz: Path
    nimages: int = Field(ge=2)
    freeze_endpoints: bool = True
    reparameterization: Literal["cubic", "linear"] = "cubic"
    coordinate_mode: Literal["all_atoms", "qm_region_only"] = "all_atoms"


class OptimizerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["zero_temperature_string"] = "zero_temperature_string"
    step_size: float = Field(gt=0.0)
    max_steps: int = Field(gt=0)
    grad_tol: float = Field(gt=0.0)
    disp_tol: float = Field(gt=0.0)
    energy_tol: float = Field(gt=0.0)


class EngineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["gaussian", "qchem", "pyscf"]
    method: str
    basis: str | None = None
    charge: int = 0
    multiplicity: int = 1
    nroots: int | None = Field(default=None, ge=1)
    root: int | None = Field(default=None, ge=1)
    state_follow: bool = False
    template: str | None = None
    command: str | None = None
    template_singlepoint: str | None = None
    template_gradient: str | None = None
    extra: dict[str, str] = Field(default_factory=dict)


class ExecutionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["serial", "parallel_images"] = "serial"
    max_concurrent_images: int = Field(default=1, ge=1)
    threads_per_image: int = Field(default=1, ge=1)
    total_slots: int | None = Field(default=None, ge=1)
    scratch_root: Path | None = None


class QMMMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    qm_atoms: list[int] = Field(default_factory=list)
    mm_charges_file: Path | None = None
    update_region_only: bool = False
    vdw_repulsion: bool = True   # add LJ repulsion between QM and MM atoms


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_dir: Path
    checkpoint_every: int = Field(default=1, ge=1)
    write_xyz_every: int = Field(default=1, ge=1)
    write_table_every: int = Field(default=1, ge=1)


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: PathConfig
    optimizer: OptimizerConfig
    engine: EngineConfig
    execution: ExecutionConfig
    qmmm: QMMMConfig = Field(default_factory=QMMMConfig)
    output: OutputConfig

    @model_validator(mode="after")
    def validate_cross_fields(self) -> "RunConfig":
        if self.execution.mode == "parallel_images":
            if self.execution.total_slots is None:
                raise ValueError("execution.total_slots is required in parallel_images mode")
            if self.execution.max_concurrent_images * self.execution.threads_per_image > self.execution.total_slots:
                raise ValueError("max_concurrent_images * threads_per_image must be <= total_slots")
        if self.path.coordinate_mode == "qm_region_only":
            if not self.qmmm.enabled:
                raise ValueError("qmmm.enabled must be true when coordinate_mode=qm_region_only")
            if not self.qmmm.qm_atoms:
                raise ValueError("qmmm.qm_atoms is required when coordinate_mode=qm_region_only")
        if self.engine.method in EXCITED_STATE_METHODS:
            if self.engine.root is None or self.engine.nroots is None:
                raise ValueError("excited-state methods require root and nroots")
        return self

