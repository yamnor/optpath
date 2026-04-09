from pathlib import Path

from optpath.cli import main


def test_inspect_requires_existing_run(tmp_path: Path, capsys) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "summary.log").write_text(
        "step,total_images,active_images,max_rms_grad_perp_eV_per_A,avg_rms_grad_perp_eV_per_A,max_displacement_A,max_abs_delta_energy_eV,selected_roots,warning_count\n"
        "0,3,1,0.1,0.1,0.0,NA,1:NA,0\n",
        encoding="utf-8",
    )
    (run_dir / "table.csv").write_text(
        "step,image_index,total_energy_eV,relative_energy_eV,rms_gradient_eV_per_A,rms_perpendicular_gradient_eV_per_A,displacement_A,selected_root,state_label,converged,success\n"
        "0,1,0.0,0.0,0.0,0.0,0.0,,ground,True,True\n",
        encoding="utf-8",
    )
    (run_dir / "checkpoints").mkdir()
    assert main(["inspect", str(run_dir)]) == 0
    out = capsys.readouterr().out
    assert "last_successful_step=0" in out

