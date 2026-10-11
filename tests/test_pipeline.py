"""Tests for metrics, config, determinism, CLI and the offline fallback."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import imputeforge.imputers.base as ibase
from imputeforge.core.config import Config
from imputeforge.core.errors import DataError
from imputeforge.eval.metrics import column_nrmse
from imputeforge.pipeline.pipeline import determinism_check

ROOT = Path(__file__).resolve().parents[1]


def test_perfect_imputation_scores_zero():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 4))
    mask = np.zeros(X.shape, dtype=bool)
    mask[:10, 0] = True
    score, per_col = column_nrmse(X, X.copy(), mask)
    assert score == 0.0 and len(per_col) >= 1


def test_shape_mismatch_raises():
    X = np.zeros((10, 3))
    with pytest.raises(DataError):
        column_nrmse(X, np.zeros((10, 4)), np.zeros((10, 3), dtype=bool))


def test_config_env_override():
    os.environ["IMPUTE_FORGE_N_ROWS"] = "321"
    try:
        cfg = Config.from_env()
        assert cfg.n_rows == 321
    finally:
        del os.environ["IMPUTE_FORGE_N_ROWS"]


def test_config_validation():
    with pytest.raises(Exception):
        Config(missing_rate=1.5)


def test_determinism_same_seed_twice():
    cfg = Config(n_rows=200, forest_estimators=6, fuse_max_iter=2, mice_max_iter=2)
    res = determinism_check(cfg, seed=7)
    assert res["n_metrics"] >= 5
    assert res["bit_identical"], res


def test_offline_fallback_without_sklearn(monkeypatch):
    monkeypatch.setattr(ibase, "_SKLEARN_OK", False)
    from imputeforge.imputers import build_pool

    pool = build_pool(seed=0)
    names = [p.name for p in pool]
    assert "knn_np" in names
    assert "mice_forest" not in names
    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 4))
    X[rng.random(X.shape) < 0.3] = np.nan
    for imp in pool:
        out = imp.fit(X).transform(X)
        assert not np.isnan(out).any()

    from imputeforge.fuse.imputefuse import ImputeFuse

    f = ImputeFuse(seed=0).fit(X)
    assert not np.isnan(f.transform(X)).any()


def test_cli_list_and_quick_benchmark(tmp_path):
    out = tmp_path / "b.json"
    r = subprocess.run([sys.executable, "cli.py", "list"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0 and "imputefuse" in r.stdout

    r = subprocess.run(
        [sys.executable, "cli.py", "benchmark", "--quick", "--full-grid", "--out", str(out)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr[-2000:]
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "imputefuse" in data["summary"]["aggregate"]
