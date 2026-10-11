"""Full benchmark: 4 DGP x 3 mechanisms x 3 seeds, plus ablation and determinism.

    python examples/run_full_benchmark.py        # ~5-6 min on one CPU core

This — not the quick demo — is the run whose output is quoted in README.md,
docs/architecture.md and the GitHub release. It writes benchmark.json and
ablation.json next to this file. Every printed number comes from this run:
nothing is hard-coded.

Author: 晨星
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))


from imputeforge.core.config import Config
from imputeforge.pipeline.pipeline import (
    ABLATION_MODES,
    ablation,
    determinism_check,
    run_benchmark,
)


def main() -> int:
    t0 = time.perf_counter()
    # Library defaults, spelled out so the shipped benchmark.json is exactly
    # what `Config()` + `run_benchmark(cfg)` reproduces. Do not shrink these for
    # speed: the reported numbers must come from the deployed configuration.
    cfg = Config(
        n_rows=500,
        n_cols=8,
        missing_rate=0.25,
        forest_estimators=12,
        mice_max_iter=4,
        fuse_max_iter=3,
        seeds=(7, 42, 123),
    )

    print("=" * 78)
    print(" ImputeForge demo — tabular missing-value imputation")
    print(" author: 晨星")
    print("=" * 78)

    report = run_benchmark(cfg)
    agg = report["summary"]["aggregate"]
    order = sorted(agg, key=lambda m: agg[m]["mean"])
    print("\n[1/4] Aggregate NRMSE over 4 DGP x 3 mechanisms x 3 seeds (lower is better)")
    print(f"  {'method':<14}{'mean':>9}{'std':>9}{'n':>6}")
    for m in order:
        print(f"  {m:<14}{agg[m]['mean']:>9.4f}{agg[m]['std']:>9.4f}{agg[m]['n']:>6}")

    h = report["headline"]
    print("\n[2/4] Headline vs strongest baseline")
    print(f"  flagship           : {h['flagship']}  {h['flagship_mean']:.4f}")
    print(f"  strongest baseline : {h['strongest_baseline']}  {h['baseline_mean']:.4f}")
    print(
        f"  reduction          : {100 * h['reduction']:+.2f}% (pre-registered gate "
        f"{100 * h['gate']:.0f}%) -> {'PASS' if h['gate_pass'] else 'MISS'}"
    )
    print(
        f"  significance       : |delta|={abs(h['delta']):.4f} vs 0.5(s1+s2)={h['sigma_crit']:.4f}"
        f" -> {'SIGNIFICANT' if h['significant'] else 'NOT significant'}"
    )
    print(f"  per-seed flagship  : {[round(v, 4) for v in h['fuse_per_seed']]}")
    print(f"  per-seed baseline  : {[round(v, 4) for v in h['base_per_seed']]}")

    print("\n  per-cell (fuse vs best single baseline)")
    for cell, d in sorted(report["summary"]["per_cell"].items()):
        fuse = d["imputefuse"]["mean"]
        name, val = min(
            ((k, v["mean"]) for k, v in d.items() if k != "imputefuse"), key=lambda kv: kv[1]
        )
        print(
            f"    {cell:<28} fuse={fuse:.4f}  best={name}({val:.4f})  {100 * (val - fuse) / val:+6.2f}%"
        )

    abl = ablation(cfg, seeds=(7, 42))
    print("\n[3/4] Ablation (2 seeds x 12 cells per mode; lower is better)")
    for mode in ABLATION_MODES:
        print(
            f"    {mode:<20} {abl[mode]['mean']:.4f} ± {abl[mode]['std']:.4f}  n={abl[mode]['n']}"
        )
    for mode in ABLATION_MODES:
        if mode == "full":
            continue
        print(f"    delta({mode} - full)  {abl[f'delta_{mode}_minus_full']:+.4f}")

    det = determinism_check(cfg, seed=7)
    print("\n[4/4] Determinism (same seed twice)")
    print(f"    metrics compared : {det['n_metrics']}")
    print(f"    max abs diff     : {det['max_abs_diff']:.3e}")
    print(f"    bit identical    : {det['bit_identical']}")

    print("\n  failure cases (derived from results, worst first)")
    for f in report["failures"]:
        print(
            f"    {f['cell']:<28} fuse={f['fuse']:.4f} best={f['best_single']}"
            f"({f['best_single_score']:.4f}) gap={f['gap']:+.4f}"
        )
        print(f"      -> {f['attribution']}")

    (HERE / "benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (HERE / "ablation.json").write_text(json.dumps(abl, indent=2), encoding="utf-8")
    print(f"\nwrote {HERE / 'benchmark.json'} and {HERE / 'ablation.json'}")
    print(f"total {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
