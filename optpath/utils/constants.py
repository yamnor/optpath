"""Shared constants for optpath."""

from __future__ import annotations

# Method names that require excited-state treatment
# (state tracking, root/nroots specification, TDDFT gradient).
# Used in config validation, state tracker, and engine dispatch.
EXCITED_STATE_METHODS: frozenset[str] = frozenset({"tda", "tddft", "tddft_1d", "pyscf_td"})
