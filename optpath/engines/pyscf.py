"""PySCF engine."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from optpath.config.schema import EngineConfig
from optpath.engines.results import ImageJob, ImageResult, RunContext
from optpath.utils.parallel import thread_env
from optpath.utils.units import hartree_per_bohr_to_ev_per_angstrom, hartree_to_ev


class PySCFEngine:
    def __init__(self, config: EngineConfig) -> None:
        self.config = config

    def evaluate(self, jobs: list[ImageJob], context: RunContext) -> list[ImageResult]:
        if context.execution_mode == "parallel_images":
            return _parallel_map(_pyscf_worker, jobs, self.config.model_dump(), context)
        return [_pyscf_worker(job, self.config.model_dump(), context) for job in jobs]


def _parallel_map(worker, jobs: list[ImageJob], config: dict, context: RunContext) -> list[ImageResult]:
    results: dict[int, ImageResult] = {}
    with ProcessPoolExecutor(max_workers=context.max_concurrent_images) as executor:
        future_map = {executor.submit(worker, job, config, context): job.image_index for job in jobs}
        for future in as_completed(future_map):
            image_index = future_map[future]
            try:
                result = future.result()
            except Exception as exc:  # pragma: no cover
                result = ImageResult(
                    image_index=image_index,
                    energy=None,
                    forces=None,
                    gradient=None,
                    selected_root=None,
                    available_roots=[],
                    state_label=None,
                    converged=False,
                    success=False,
                    error_message=str(exc),
                )
            results[image_index] = result
    return [results[job.image_index] for job in jobs]


def _pyscf_worker(job: ImageJob, config: dict, context: RunContext) -> ImageResult:
    try:
        from pyscf import dft, gto, qmmm, scf, tdscf
    except ModuleNotFoundError:
        return ImageResult(
            image_index=job.image_index,
            energy=None,
            forces=None,
            gradient=None,
            selected_root=job.state_spec.get("root"),
            available_roots=[],
            state_label=None,
            converged=False,
            success=False,
            error_message="pyscf is not installed",
        )
    symbols = job.atoms.get_chemical_symbols()
    positions = job.atoms.get_positions()
    atom_spec = "; ".join(
        f"{symbol} {x:.12f} {y:.12f} {z:.12f}"
        for symbol, (x, y, z) in zip(symbols, positions, strict=True)
    )
    with thread_env(context.threads_per_image):
        mol = gto.M(atom=atom_spec, basis=config.get("basis") or "sto-3g", charge=config.get("charge", 0), spin=max(config.get("multiplicity", 1) - 1, 0))
        method = config.get("method", "hf")
        if method == "hf":
            mf = scf.RHF(mol)
        else:
            mf = dft.RKS(mol)
            mf.xc = config.get("extra", {}).get("xc", "b3lyp")
        if job.qmmm.get("enabled") and job.qmmm.get("mm_charges"):
            coords = [(entry["x"], entry["y"], entry["z"]) for entry in job.qmmm["mm_charges"]]
            charges = [entry["charge"] for entry in job.qmmm["mm_charges"]]
            mf = qmmm.mm_charge(mf, coords, charges)
        energy = mf.kernel()
        converged = bool(mf.converged)
        if method in {"tda", "tddft", "pyscf_td"}:
            nroots = job.state_spec.get("nroots") or 1
            # Config root is 1-based; PySCF uses 0-based state indices.
            root_1based = job.state_spec.get("root") or 1
            root_0based = root_1based - 1
            td = tdscf.TDA(mf) if method == "tda" else tdscf.TDDFT(mf)
            td.nstates = nroots
            td.kernel()
            grad = td.nuc_grad_method().kernel(state=root_0based)
            roots = [
                {"root": idx + 1, "excitation_energy_ev": float(hartree_to_ev(float(value)))}
                for idx, value in enumerate(td.e)
            ]
            state_label = f"root_{root_1based}"
            selected_root = root_1based
        else:
            grad = mf.nuc_grad_method().kernel()
            roots = []
            state_label = "ground"
            selected_root = None
        gradient = hartree_per_bohr_to_ev_per_angstrom(np.array(grad, dtype=float))
        forces = -gradient
    return ImageResult(
        image_index=job.image_index,
        energy=float(hartree_to_ev(float(energy))),
        forces=forces,
        gradient=gradient,
        selected_root=selected_root,
        available_roots=roots,
        state_label=state_label,
        converged=converged,
        success=True,
        error_message=None,
    )
