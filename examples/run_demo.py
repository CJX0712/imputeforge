"""Quick end-to-end demo (CPU budget: <= 60 s).

    python examples/run_demo.py

Exercises the whole stack on a reduced grid — benchmark, one ablation group and a
determinism check — and writes demo_quick.json. It is a *smoke* run: the numbers
quoted in README / docs come from the full run,

    python examples/run_full_benchmark.py

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
from imputeforge.pipeline.pipeline import ablation, determinism_check, run_benchmark


def main() -> int:
    t0 = time.perf_counter()
    # Reduced grid and forest size only: the flagship keeps its shipped probe
    # settings (probe_repeats / max_iter) so the demo still shows the real
    # mechanism, just on fewer cells. Cost driver is probe scoring.
    cfg = Config(
        n_rows=220,
        n_cols=8,
        missing_rate=0.25,
        forest_estimators=6,
        mice_max_iter=2,
        fuse_max_iter=2,
        seeds=(7,),
    )

    print("=" * 78)
    print(" ImputeForge quick demo — tabular missing-value imputation")
    print(" author: 晨星   |   full numbers: python examples/run_full_benchmark.py")
    print("=" * 78)

    report = run_benchmark(cfg)
    agg = report["summary"]["aggregate"]
    order = sorted(agg, key=lambda m: agg[m]["mean"])
    print("\n[1/4] Aggregate NRMSE, 4 DGP x 3 mechanisms x 1 seed (lower is better)")
    print(f"  {'method':<14}{'mean':>9}{'n':>6}")
    for m in order:
        print(f"  {m:<14}{agg[m]['mean']:>9.4f}{agg[m]['n']:>6}")

    print("\n[2/4] Flagship vs the best single method on this grid")
    print(f"  imputefuse   {agg['imputefuse']['mean']:.4f}")
    best = min((m for m in agg if m != "imputefuse"), key=lambda m: agg[m]["mean"])
    print(f"  {best:<13} {agg[best]['mean']:.4f}")

    # Ablation runs on the two cells where per-column selection and the MNAR
    # shift actually bite (nonlinear / heterogeneous structure under MAR); the
    # full 5-mode x 12-cell ablation belongs to examples/run_full_benchmark.py.
    abl = ablation(
        cfg,
        seeds=(7,),
        dgps=("nonlinear", "mixed_hetero"),
        mechs=("mar",),
        modes=("full", "no_selection", "no_shift"),
    )
    print("\n[3/4] Component ablation: nonlinear/mar + mixed_hetero/mar (lower is better)")
    for mode in ("full", "no_selection", "no_shift"):
        print(f"    {mode:<16} {abl[mode]['mean']:.4f}  n={abl[mode]['n']}")
    for mode in ("no_selection", "no_shift"):
        print(f"    delta({mode} - full)  {abl[f'delta_{mode}_minus_full']:+.4f}")

    det = determinism_check(cfg, seed=7)
    print("\n[4/4] Determinism (same seed twice)")
    print(f"    metrics compared : {det['n_metrics']}")
    print(f"    max abs diff     : {det['max_abs_diff']:.3e}")
    print(f"    bit identical    : {det['bit_identical']}")

    (HERE / "demo_quick.json").write_text(
        json.dumps({"benchmark": report, "ablation": abl, "determinism": det}, indent=2),
        encoding="utf-8",
    )
    print(f"\nwrote {HERE / 'demo_quick.json'}")
    print(f"total {time.perf_counter() - t0:.1f}s (budget 60s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
