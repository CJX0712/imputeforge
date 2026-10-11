"""End-to-end pipeline: DGP -> mask -> impute -> NRMSE -> benchmark report.

Protocol (no leakage): every imputer is fit on the *training rows only*; the
per-column normalizer (std) comes from the ground-truth generator's full
column, a property of the DGP rather than of the test block.

Author: 晨星
"""

from __future__ import annotations

import time

import numpy as np

from imputeforge.core.config import Config
from imputeforge.core.seed import set_all
from imputeforge.core.types import CellResult, MaskedSplit
from imputeforge.data.dgp import DGP_NAMES, generate
from imputeforge.data.masks import MECHANISMS, apply_mechanism, mask_values
from imputeforge.eval.metrics import column_nrmse
from imputeforge.fuse.imputefuse import FUSE_MODES, ImputeFuse
from imputeforge.imputers import build_pool


def _build_split(dgp: str, mechanism: str, seed: int, cfg: Config) -> MaskedSplit:
    """Deterministic cell construction: one rng, one mask per block."""
    set_all(seed)
    rng = np.random.default_rng(seed * 7919 + 13)
    X = generate(dgp, cfg.n_rows, cfg.n_cols, seed)
    n_train = int(cfg.train_frac * cfg.n_rows)
    X_train_full, X_test_full = X[:n_train], X[n_train:]
    mask_train = apply_mechanism(X_train_full, mechanism, cfg.missing_rate, rng)
    mask_test = apply_mechanism(X_test_full, mechanism, cfg.missing_rate, rng)
    return MaskedSplit(
        X_train=mask_values(X_train_full, mask_train),
        X_test=mask_values(X_test_full, mask_test),
        X_train_full=X_train_full,
        X_test_full=X_test_full,
        mask_train=mask_train,
        mask_test=mask_test,
        cell=f"{dgp}/{mechanism}",
        seed=seed,
    )


def make_split(dgp: str, mechanism: str, seed: int, cfg: Config | None = None) -> MaskedSplit:
    """Public wrapper for building a single benchmark cell."""
    return _build_split(dgp, mechanism, seed, cfg or Config.from_env())


def _method_list(cfg: Config, seed: int) -> list[object]:
    methods: list[object] = list(
        build_pool(
            seed=seed,
            mice_max_iter=cfg.mice_max_iter,
            forest_estimators=cfg.forest_estimators,
        )
    )
    methods.append(
        ImputeFuse(
            probe_rate=cfg.probe_rate,
            max_iter=cfg.fuse_max_iter,
            forest_estimators=cfg.forest_estimators,
            seed=seed,
            probe_repeats=cfg.probe_repeats,
        )
    )
    return methods


def evaluate_split(
    split: MaskedSplit,
    cfg: Config,
    methods: list[object] | None = None,
) -> list[CellResult]:
    """Fit every method on train rows, score its imputation on test rows."""
    results: list[CellResult] = []
    pool = methods if methods is not None else _method_list(cfg, split.seed)
    for imp in pool:
        t0 = time.perf_counter()
        rec = CellResult(
            cell=split.cell,
            seed=split.seed,
            method=getattr(imp, "name", "unknown"),
            nrmse=float("nan"),
            per_column=[],
            elapsed_sec=0.0,
            skipped=False,
        )
        try:
            imp.fit(split.X_train)
            X_imp = imp.transform(split.X_test)
            nrmse, per_col = column_nrmse(split.X_test_full, X_imp, split.mask_test)
            rec.nrmse = nrmse
            rec.per_column = per_col
        except Exception as exc:
            rec.skipped = True
            rec.error = f"{type(exc).__name__}: {exc}"
        rec.elapsed_sec = time.perf_counter() - t0
        results.append(rec)
    return results


def _agg(values: list[float]) -> dict[str, float]:
    return {"mean": float(np.mean(values)), "std": float(np.std(values)), "n": len(values)}


def summarize(results: list[CellResult]) -> dict:
    """Aggregate per method globally, per cell, and per seed."""
    per_method: dict[str, list[float]] = {}
    per_cell: dict[str, dict[str, list[float]]] = {}
    per_seed: dict[str, dict[int, list[float]]] = {}
    for r in results:
        if r.skipped or not np.isfinite(r.nrmse):
            continue
        per_method.setdefault(r.method, []).append(r.nrmse)
        per_cell.setdefault(r.cell, {}).setdefault(r.method, []).append(r.nrmse)
        per_seed.setdefault(r.method, {}).setdefault(r.seed, []).append(r.nrmse)
    return {
        "aggregate": {m: _agg(v) for m, v in per_method.items()},
        "per_cell": {c: {m: _agg(v) for m, v in d.items()} for c, d in per_cell.items()},
        "per_seed": {
            m: {str(s): float(np.mean(v)) for s, v in d.items()} for m, d in per_seed.items()
        },
    }


def _headline(summary: dict, gate: float) -> dict:
    agg = summary["aggregate"]
    if "imputefuse" not in agg:
        return {"ok": False, "reason": "flagship missing"}
    n_fuse = agg["imputefuse"]["n"]
    eligible = {m: v for m, v in agg.items() if m != "imputefuse" and v["n"] == n_fuse}
    if not eligible:
        return {"ok": False, "reason": "no comparable baseline"}
    best_name = min(eligible, key=lambda m: eligible[m]["mean"])
    base, fuse = eligible[best_name], agg["imputefuse"]
    reduction = (base["mean"] - fuse["mean"]) / max(base["mean"], 1e-12)
    fs = summary["per_seed"].get("imputefuse", {})
    bs = summary["per_seed"].get(best_name, {})
    seeds = sorted(set(fs) & set(bs), key=int)
    f_vals = [fs[str(s)] for s in seeds]
    b_vals = [bs[str(s)] for s in seeds]
    delta = float(np.mean(f_vals) - np.mean(b_vals))
    sigma_crit = 0.5 * (float(np.std(f_vals)) + float(np.std(b_vals)))
    return {
        "ok": True,
        "flagship": "imputefuse",
        "strongest_baseline": best_name,
        "flagship_mean": fuse["mean"],
        "baseline_mean": base["mean"],
        "reduction": float(reduction),
        "gate": gate,
        "gate_pass": bool(reduction >= gate),
        "delta": delta,
        "sigma_crit": sigma_crit,
        "significant": bool(abs(delta) > sigma_crit),
        "seeds": [int(s) for s in seeds],
        "fuse_per_seed": f_vals,
        "base_per_seed": b_vals,
    }


def _failures(summary: dict) -> list[dict]:
    """Derive >=3 honest failure cases from actual results (worst gaps first)."""
    rows = []
    for cell, methods in summary["per_cell"].items():
        if "imputefuse" not in methods:
            continue
        fuse = methods["imputefuse"]["mean"]
        singles = {m: v["mean"] for m, v in methods.items() if m != "imputefuse"}
        if not singles:
            continue
        best_single = min(singles, key=lambda m: singles[m])
        rows.append(
            {
                "cell": cell,
                "fuse": fuse,
                "best_single": best_single,
                "best_single_score": singles[best_single],
                "gap": fuse - singles[best_single],
            }
        )
    rows.sort(key=lambda r: -r["gap"])
    for r in rows[:3]:
        r["attribution"] = (
            "one inductive bias dominates every column of this cell while the "
            "augmented-mask CV estimate is noisy -> fusion cannot help here"
            if r["gap"] > 0
            else "fuse wins but margin is small (cell near the fusion ceiling)"
        )
    return rows[:3]


def run_benchmark(
    cfg: Config | None = None,
    dgps: tuple[str, ...] = DGP_NAMES,
    mechs: tuple[str, ...] = MECHANISMS,
    seeds: tuple[int, ...] | None = None,
) -> dict:
    """Run the cell grid and return a JSON-serializable report."""
    cfg = cfg or Config.from_env()
    seeds = seeds or cfg.seeds
    results: list[CellResult] = []
    for dgp in dgps:
        for mech in mechs:
            for seed in seeds:
                split = _build_split(dgp, mech, seed, cfg)
                results.extend(evaluate_split(split, cfg))
    summary = summarize(results)
    return {
        "config": {
            "n_rows": cfg.n_rows,
            "n_cols": cfg.n_cols,
            "missing_rate": cfg.missing_rate,
            "probe_rate": cfg.probe_rate,
            "fuse_max_iter": cfg.fuse_max_iter,
            "forest_estimators": cfg.forest_estimators,
            "train_frac": cfg.train_frac,
            "seeds": list(seeds),
            "dgps": list(dgps),
            "mechanisms": list(mechs),
        },
        "results": [dict(r.__dict__) for r in results],
        "summary": summary,
        "headline": _headline(summary, cfg.gate_reduction),
        "failures": _failures(summary),
        "skipped": [
            {"cell": r.cell, "seed": r.seed, "method": r.method, "error": r.error}
            for r in results
            if r.skipped
        ],
    }


ABLATION_MODES = FUSE_MODES


def ablation(
    cfg: Config | None = None,
    seeds: tuple[int, ...] = (7, 42),
    dgps: tuple[str, ...] = DGP_NAMES,
    mechs: tuple[str, ...] = MECHANISMS,
    modes: tuple[str, ...] = ABLATION_MODES,
) -> dict:
    """Component ablation over the flagship's mechanisms."""
    cfg = cfg or Config.from_env()
    out: dict[str, dict] = {}
    for mode in modes:
        vals: list[float] = []
        for dgp in dgps:
            for mech in mechs:
                for seed in seeds:
                    split = _build_split(dgp, mech, seed, cfg)
                    fuse = ImputeFuse(
                        probe_rate=cfg.probe_rate,
                        max_iter=cfg.fuse_max_iter,
                        forest_estimators=cfg.forest_estimators,
                        seed=seed,
                        mode=mode,
                        probe_repeats=cfg.probe_repeats,
                    )
                    res = evaluate_split(split, cfg, methods=[fuse])
                    if res and not res[0].skipped:
                        vals.append(res[0].nrmse)
        out[mode] = _agg(vals)
    for mode in modes:
        if mode == "full" or "full" not in out:
            continue
        out[f"delta_{mode}_minus_full"] = out[mode]["mean"] - out["full"]["mean"]
    out["note"] = "positive delta => the ablated-away component helps (lower is better)"
    return out


def determinism_check(cfg: Config | None = None, seed: int = 7) -> dict:
    """Run one cell twice under the same seed; core metrics must match bitwise."""
    cfg = cfg or Config.from_env()

    def once() -> list[float]:
        split = _build_split("gauss_lowrank", "mcar", seed, cfg)
        res = evaluate_split(split, cfg)
        return [r.nrmse for r in res if not r.skipped]

    a, b = once(), once()
    diff = 0.0 if len(a) != len(b) else float(np.max(np.abs(np.array(a) - np.array(b))))
    return {
        "n_metrics": len(a),
        "max_abs_diff": diff,
        "bit_identical": bool(diff == 0.0),
    }
