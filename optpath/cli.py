"""Command line interface."""

from __future__ import annotations

import argparse
from pathlib import Path

from optpath.config.loader import load_config
from optpath.core.checkpoint import load_checkpoint
from optpath.core.string_optimizer import StringOptimizer
from optpath.io.logs import read_summary, read_table
from optpath.io.xyz import read_xyz_images, write_xyz_images


def cmd_run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    optimizer = StringOptimizer(config)
    result = optimizer.run_from_step(start_step=0)
    print(f"converged={result.converged} last_successful_step={result.last_successful_step}")
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    optimizer = StringOptimizer(config)
    checkpoint_dir = optimizer.run_dir / "checkpoints"
    meta_candidates = sorted(
        path for path in checkpoint_dir.glob("step_*.meta.json") if not path.name.endswith(".diagnostics.meta.json")
    )
    if not meta_candidates:
        raise SystemExit("no checkpoints found")
    latest_meta = meta_candidates[-1]
    latest_arrays = latest_meta.with_suffix("").with_suffix(".arrays.npz")
    checkpoint_data = load_checkpoint(latest_meta, latest_arrays, optimizer.band)
    start_step = optimizer.restore_checkpoint(checkpoint_data)
    result = optimizer.run_from_step(start_step=start_step)
    print(f"converged={result.converged} last_successful_step={result.last_successful_step}")
    return 0


def cmd_singlepoint(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    optimizer = StringOptimizer(config)
    results = optimizer.evaluate_band(step_index=0)
    for result in results:
        print(
            f"image={result.image_index} success={result.success} converged={result.converged} "
            f"energy={result.energy}"
        )
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    summary_path = run_dir / "summary.log"
    table_path = run_dir / "table.csv"
    rows = read_summary(summary_path)
    latest = rows[-1] if rows else {}
    print(f"run_dir={run_dir}")
    print(f"last_successful_step={latest.get('step', 'NA')}")
    if latest:
        energy_delta = latest.get("max_abs_delta_energy_eV", "NA")
        print(f"max_rms_grad_perp={latest.get('max_rms_grad_perp_eV_per_A', 'NA')} eV/Angstrom")
        print(f"max_displacement={latest.get('max_displacement_A', 'NA')} Angstrom")
        print(f"max_abs_delta_energy={energy_delta} eV")
        print(f"warning_count={latest.get('warning_count', 'NA')}")
    checkpoints = sorted((run_dir / "checkpoints").glob("*.meta.json"))
    diagnostics = sorted((run_dir / "checkpoints").glob("*.diagnostics.meta.json"))
    print(f"checkpoints={len(checkpoints)} diagnostics={len(diagnostics)}")
    if table_path.exists():
        rows = read_table(table_path)
        if rows:
            latest_step = rows[-1]["step"]
            latest_rows = [row for row in rows if row["step"] == latest_step]
            print("latest_step_rows:")
            for row in latest_rows:
                print(
                    f"  image={row['image_index']} energy={row['total_energy_eV']} "
                    f"relative={row['relative_energy_eV']} selected_root={row['selected_root']}"
                )
    return 0


def cmd_interp(args: argparse.Namespace) -> int:
    from optpath.utils.zmat_interp import get_zmatrix_string, interpolate_zmat

    reactant_images = read_xyz_images(args.reactant)
    product_images = read_xyz_images(args.product)

    if len(reactant_images) == 0:
        raise SystemExit(f"no structures found in {args.reactant}")
    if len(product_images) == 0:
        raise SystemExit(f"no structures found in {args.product}")

    atoms1 = reactant_images[0]
    atoms2 = product_images[-1]

    if args.show_zmat:
        print("=== Z-matrix (derived from reactant) ===")
        print(get_zmatrix_string(atoms1))
        return 0

    images = interpolate_zmat(
        atoms1,
        atoms2,
        nimages=args.nimages,
        interpolate_dihedrals=not args.no_dihedral_wrap,
    )

    output = Path(args.output)
    write_xyz_images(output, images)
    print(f"wrote {len(images)} images -> {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="optpath")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("config")
    run_parser.set_defaults(func=cmd_run)

    resume_parser = subparsers.add_parser("resume")
    resume_parser.add_argument("config")
    resume_parser.set_defaults(func=cmd_resume)

    single_parser = subparsers.add_parser("singlepoint")
    single_parser.add_argument("config")
    single_parser.set_defaults(func=cmd_singlepoint)

    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("run_dir")
    inspect_parser.set_defaults(func=cmd_inspect)

    interp_parser = subparsers.add_parser(
        "interp",
        help="Interpolate two structures in Z-matrix (internal coordinate) space",
    )
    interp_parser.add_argument("reactant", help="Reactant XYZ file (first frame used)")
    interp_parser.add_argument("product", help="Product XYZ file (last frame used)")
    interp_parser.add_argument("output", help="Output multi-frame XYZ file")
    interp_parser.add_argument(
        "--nimages", type=int, default=8,
        help="Total number of images including endpoints (default: 8)",
    )
    interp_parser.add_argument(
        "--no-dihedral-wrap", action="store_true",
        help="Disable shortest-arc wrapping for dihedral angles",
    )
    interp_parser.add_argument(
        "--show-zmat", action="store_true",
        help="Print the Z-matrix definition (from reactant) and exit",
    )
    interp_parser.set_defaults(func=cmd_interp)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
