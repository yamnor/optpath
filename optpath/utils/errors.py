"""Custom exceptions."""

from __future__ import annotations


class OptPathError(RuntimeError):
    """Base package error."""


class ConfigurationError(OptPathError):
    """Invalid configuration."""


class EngineUnavailableError(OptPathError):
    """Optional engine dependency missing."""


class EngineExecutionError(OptPathError):
    """External engine failed."""


class OptimizationAbort(OptPathError):
    """Optimization aborted due to engine failure or convergence issue."""

