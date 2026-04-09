"""Unit conversion helpers."""

from __future__ import annotations

HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANGSTROM = 0.529177210903
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM
EV_TO_HARTREE = 1.0 / HARTREE_TO_EV
HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM = HARTREE_TO_EV / BOHR_TO_ANGSTROM
EV_PER_ANGSTROM_TO_HARTREE_PER_BOHR = 1.0 / HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM


def hartree_to_ev(value: float) -> float:
    return value * HARTREE_TO_EV


def ev_to_hartree(value: float) -> float:
    return value * EV_TO_HARTREE


def bohr_to_angstrom(value: float) -> float:
    return value * BOHR_TO_ANGSTROM


def angstrom_to_bohr(value: float) -> float:
    return value * ANGSTROM_TO_BOHR


def hartree_per_bohr_to_ev_per_angstrom(value: float) -> float:
    return value * HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM


def ev_per_angstrom_to_hartree_per_bohr(value: float) -> float:
    return value * EV_PER_ANGSTROM_TO_HARTREE_PER_BOHR

