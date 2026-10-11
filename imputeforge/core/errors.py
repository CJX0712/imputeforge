"""Typed error hierarchy. Codes E100-E500.

Author: 晨星
"""


class ImputeForgeError(Exception):
    """Base error for all ImputeForge failures."""

    code = "E000"


class ConfigError(ImputeForgeError):
    """E100: invalid configuration or environment override."""

    code = "E100"


class DataError(ImputeForgeError):
    """E200: malformed data, bad mask, or degenerate column."""

    code = "E200"


class ConvergenceError(ImputeForgeError):
    """E300: iterative algorithm failed to converge or violated monotonicity."""

    code = "E300"


class ShapeError(ImputeForgeError):
    """E400: shape / dtype contract violation."""

    code = "E400"


class BackendUnavailable(ImputeForgeError):
    """E500: optional Tier-0 backend (sklearn) not importable."""

    code = "E500"
