"""Engine factory and shared helpers."""

from __future__ import annotations

from pathlib import Path

from optpath.config.schema import EngineConfig, RunConfig
from optpath.engines.results import Engine


def build_engine(config: RunConfig) -> Engine:
    engine_type = config.engine.type
    if engine_type == "gaussian":
        from optpath.engines.gaussian import GaussianEngine

        return GaussianEngine(config.engine)
    if engine_type == "qchem":
        from optpath.engines.qchem import QChemEngine

        return QChemEngine(config.engine)
    if engine_type == "pyscf":
        from optpath.engines.pyscf import PySCFEngine

        return PySCFEngine(config.engine)
    raise ValueError(f"unsupported engine type: {engine_type}")


def resolve_template(base_dir: Path, template: str | None, fallback: str | None = None) -> Path | None:
    chosen = template or fallback
    if chosen is None:
        return None
    path = Path(chosen)
    if path.is_absolute():
        return path
    return base_dir / path

