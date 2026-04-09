from pathlib import Path

import numpy as np

from optpath.engines.gaussian import parse_gaussian_output
from optpath.engines.qchem import parse_qchem_grad_file, parse_qchem_output


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

