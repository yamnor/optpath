"""Configuration loading helpers."""

from __future__ import annotations

from pathlib import Path

import yaml

from optpath.config.schema import RunConfig


def load_config(path: str | Path) -> RunConfig:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    base_dir = config_path.parent.resolve()
    _resolve_paths(data, base_dir)
    config = RunConfig.model_validate(data)
    return config


def _resolve_paths(data: dict, base_dir: Path) -> None:
    path_section = data.get("path", {})
    for key in ("initial_xyz",):
        if key in path_section and path_section[key] is not None:
            path_section[key] = str((base_dir / path_section[key]).resolve())
    output = data.get("output", {})
    if "run_dir" in output and output["run_dir"] is not None:
        output["run_dir"] = str((base_dir / output["run_dir"]).resolve())
    qmmm = data.get("qmmm", {})
    if "mm_charges_file" in qmmm and qmmm["mm_charges_file"] is not None:
        qmmm["mm_charges_file"] = str((base_dir / qmmm["mm_charges_file"]).resolve())
    execution = data.get("execution", {})
    if "scratch_root" in execution and execution["scratch_root"] is not None:
        execution["scratch_root"] = str(Path(execution["scratch_root"]))
    engine = data.get("engine", {})
    for key in ("template", "template_singlepoint", "template_gradient"):
        if key in engine and engine[key] is not None:
            engine[key] = str((base_dir / engine[key]).resolve())
