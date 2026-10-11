"""Command line interface.

    python cli.py benchmark [--quick] [--out benchmark.json]
    python cli.py ablation  [--out ablation.json]
    python cli.py detcheck
    python cli.py list

Author: 晨星
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _fmt_table(rows: list[dict], headers: list[str]) -> str:
    widths = [
        max(len(h), *(len(str(r.get(h, ""))) for r in rows)) if rows else len(h) for h in headers
    ]
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    sep = "  ".join("-" * w for w in widths)
    body = "\n".join(
        "  ".join(str(r.get(h, "")).ljust(w) for h, w in zip(headers, widths)) for r in rows
    )
    return f"{line}\n{sep}\n{body}"


def cmd_benchmark(args: argparse.Namespace) -> int:
    from imputeforge.core.config import Config
    from imputeforge.data.dgp import DGP_NAMES
    from imputeforge.data.masks import MECHANISMS
    from imputeforge.pipeline.pipeline import run_benchmark

    cfg = Config.from_env()
    if args.quick:
        cfg = Config(n_rows=260, forest_estimators=6, fuse_max_iter=2)
    seeds = tuple(args.seeds) if args.seeds else cfg.seeds
    dgps = DGP_NAMES if args.full_grid else DGP_NAMES[:2]
    mechs = MECHANISMS if args.full_grid else ("mcar", "mnar")
    report = run_benchmark(cfg, dgps=dgps, mechs=mechs, seeds=seeds)

    rows = [
        {"method": m, "mean": f"{v['mean']:.4f}", "std": f"{v['std']:.4f}", "n": v["n"]}
        for m, v in sorted(report["summary"]["aggregate"].items(), key=lambda kv: kv[1]["mean"])
    ]
    print("\nAggregate NRMSE (lower is better)")
    print(_fmt_table(rows, ["method", "mean", "std", "n"]))

    h = report["headline"]
    print("\nHeadline")
    print(f"  flagship            : {h['flagship']}")
    print(f"  strongest baseline  : {h['strongest_baseline']}")
    print(f"  flagship mean       : {h['flagship_mean']:.4f}")
    print(f"  baseline mean       : {h['baseline_mean']:.4f}")
    print(f"  reduction           : {100 * h['reduction']:.2f}%  (gate {100 * h['gate']:.0f}%)")
    print(
        f"  significance        : |delta|={abs(h['delta']):.4f} vs 0.5(s1+s2)={h['sigma_crit']:.4f}"
        f" -> {'SIGNIFICANT' if h['significant'] else 'not significant'}"
    )

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0 if report["skipped"] == [] else 1


def cmd_ablation(args: argparse.Namespace) -> int:
    from imputeforge.core.config import Config
    from imputeforge.pipeline.pipeline import ABLATION_MODES, ablation

    cfg = Config.from_env()
    modes = tuple(args.modes) if getattr(args, "modes", None) else ABLATION_MODES
    out = ablation(cfg, seeds=tuple(args.seeds), modes=modes)
    rows = [
        {
            "mode": m,
            "mean": f"{out[m]['mean']:.4f}",
            "std": f"{out[m]['std']:.4f}",
            "n": out[m]["n"],
        }
        for m in modes
        if m in out
    ]
    print("\nAblation (lower is better)")
    print(_fmt_table(rows, ["mode", "mean", "std", "n"]))
    for key, val in out.items():
        if key.startswith("delta_"):
            print(f"  {key}: {val:+.4f}")
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


def cmd_detcheck(args: argparse.Namespace) -> int:
    from imputeforge.core.config import Config
    from imputeforge.pipeline.pipeline import determinism_check

    cfg = Config.from_env()
    res = determinism_check(cfg, seed=args.seed)
    print(json.dumps(res, indent=2))
    return 0 if res["bit_identical"] else 1


def cmd_list(args: argparse.Namespace) -> int:
    from imputeforge.core.config import Config
    from imputeforge.fuse.imputefuse import CANDIDATES, FUSE_MODES
    from imputeforge.imputers import build_pool

    cfg = Config.from_env()
    pool = build_pool(
        seed=0, mice_max_iter=cfg.mice_max_iter, forest_estimators=cfg.forest_estimators
    )
    print("pool:      " + ", ".join(p.name for p in pool))
    print("flagship:  imputefuse (EM-anchored MICE)")
    print("candidates:" + ", ".join(CANDIDATES))
    print("modes:     " + ", ".join(FUSE_MODES))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="imputeforge", description="ImputeForge CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("benchmark", help="run the cell grid")
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--quick", action="store_true", help="small grid for CI smoke")
    p.add_argument("--full-grid", action="store_true", help="all DGPs x all mechanisms")
    p.set_defaults(fn=cmd_benchmark)

    p = sub.add_parser("ablation", help="component ablation of the flagship")
    p.add_argument("--seeds", type=int, nargs="*", default=(7, 42))
    p.add_argument("--modes", nargs="*", default=None, help="subset of fuse modes")
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_ablation)

    p = sub.add_parser("detcheck", help="determinism check (same seed twice)")
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(fn=cmd_detcheck)

    p = sub.add_parser("list", help="list methods, candidates and modes")
    p.set_defaults(fn=cmd_list)

    args = parser.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
