from pathlib import Path

import numpy as np

from optpath.engines.gaussian import parse_gaussian_output
from optpath.engines.qchem import _promote_qchem_grad_file, parse_qchem_grad_file, parse_qchem_output


def test_parse_gaussian_output() -> None:
    path = Path("tests/fixtures/gaussian/sample.out")
    result = parse_gaussian_output(path, image_index=1)
    assert result.success
    assert result.converged
    assert result.energy is not None
    assert result.forces is not None
    assert result.forces.shape == (2, 3)


def test_parse_qchem_output() -> None:
    output = Path("tests/fixtures/qchem/sample.out")
    grad = Path("tests/fixtures/qchem/GRAD")
    result = parse_qchem_output(output, grad, image_index=1, selected_root=2)
    assert result.success
    assert result.selected_root == 2
    assert result.available_roots
    assert result.gradient is not None


def test_parse_qchem_grad_file() -> None:
    energy, gradient = parse_qchem_grad_file(Path("tests/fixtures/qchem/GRAD"))
    assert energy is not None
    assert gradient is not None
    assert gradient.shape == (2, 3)


def test_promote_qchem_grad_file_from_scratch_subdir(tmp_path: Path) -> None:
    sub = tmp_path / "qchem2178.0"
    sub.mkdir()
    (sub / "GRAD").write_text("$comment\n-1.0\n$end\n", encoding="utf-8")
    dest = tmp_path / "GRAD"
    _promote_qchem_grad_file(tmp_path, "GRAD", dest)
    assert dest.is_file()
    assert "-1.0" in dest.read_text()


def test_promote_qchem_grad_file_keeps_existing_dest(tmp_path: Path) -> None:
    dest = tmp_path / "GRAD"
    dest.write_text("keep\n", encoding="utf-8")
    sub = tmp_path / "other"
    sub.mkdir()
    (sub / "GRAD").write_text("other\n", encoding="utf-8")
    _promote_qchem_grad_file(tmp_path, "GRAD", dest)
    assert dest.read_text() == "keep\n"


def test_parse_qchem_output_qchem6_style_without_grad_file(tmp_path: Path) -> None:
    """Q-Chem 6.x prints Total energy = ... and may not write GRAD in the run directory."""
    out = tmp_path / "qm.out"
    out.write_text(
        """
 Total energy =  -556.64645845
 Calculating analytic gradient of the SCF energy
 Gradient of SCF Energy
            1           2
    1   0.0100000000  -0.0200000000
    2   0.0300000000   0.0400000000
    3   0.0500000000   0.0600000000
 Max gradient component =       1.000E-02
 Thank you very much for using Q-Chem.  Have a nice day.
""",
        encoding="utf-8",
    )
    missing_grad = tmp_path / "GRAD"
    result = parse_qchem_output(out, missing_grad, image_index=0, selected_root=None)
    assert result.success
    assert result.energy is not None
    assert result.forces is not None
    assert result.forces.shape == (2, 3)

