"""Configuration with IMPUTE_FORGE_* env overrides and schema validation.

Author: 晨星
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from typing import get_type_hints

from imputeforge.core.errors import ConfigError

ENV_PREFIX = "IMPUTE_FORGE_"


@dataclass
class Config:
    """Runtime configuration. All knobs are validated on construction."""

    n_rows: int = 500
    n_cols: int = 8
    missing_rate: float = 0.25
    probe_rate: float = 0.25  # hidden probe rate for per-column model selection
    probe_repeats: int = 2  # independent probe masks pooled when scoring candidates
    weight_power: float = 4.0  # inverse-error sharpening exponent
    fuse_max_iter: int = 3  # MICE refinement sweeps
    seeds: tuple[int, ...] = (7, 42, 123)
    mice_max_iter: int = 4
    forest_estimators: int = 12
    em_max_iter: int = 200
    em_tol: float = 1e-7
    soft_lambda_grid: tuple[float, ...] = (0.05, 0.1, 0.2)
    train_frac: float = 0.7
    gate_reduction: float = 0.05  # pre-registered performance gate

    @classmethod
    def from_env(cls) -> Config:
        """Build a Config honoring IMPUTE_FORGE_<FIELD> environment overrides."""
        hints = get_type_hints(cls)
        kwargs: dict[str, object] = {}
        for f in fields(cls):
            key = ENV_PREFIX + f.name.upper()
            if key not in os.environ:
                continue
            raw = os.environ[key]
            typ = hints[f.name]
            try:
                if typ is int:
                    kwargs[f.name] = int(raw)
                elif typ is float:
                    kwargs[f.name] = float(raw)
                else:
                    kwargs[f.name] = raw
            except ValueError as exc:  # pragma: no cover - defensive
                raise ConfigError(f"invalid env override {key}={raw!r}") from exc
        return cls(**kwargs)  # type: ignore[arg-type]

    def __post_init__(self) -> None:
        if not (0.0 < self.missing_rate < 1.0):
            raise ConfigError("missing_rate must be in (0, 1)")
        if not (0.0 < self.probe_rate < 1.0):
            raise ConfigError("probe_rate must be in (0, 1)")
        if self.n_rows < 50 or self.n_cols < 2:
            raise ConfigError("n_rows>=50 and n_cols>=2 required")
        if self.weight_power <= 0:
            raise ConfigError("weight_power must be positive")
        if len(self.seeds) < 1:
            raise ConfigError("at least one seed required")
