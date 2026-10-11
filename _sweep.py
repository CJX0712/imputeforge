"""Offline sweep: probe_repeats x Occam margin, scored against the strongest baseline.

Scratch experiment script (not part of the shipped API). Its only purpose is to
choose two defaults with evidence instead of taste, and its output is quoted in
docs/architecture.md.

    python _sweep.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from imputeforge.core.config import Config  # noqa: E402
from imputeforge.data.dgp import DGP_NAMES  # noqa: E402
from imputeforge.data.masks import MECHANISMS  # noqa: E402
from imputeforge.eval.metrics import column_nrmse  # noqa: E402
from imputeforge.fuse.imputefuse import ImputeFuse  # noqa: E402
from imputeforge.imputers.em import GaussianEMImputer  # noqa: E402
from imputeforge.pipeline.pipeline import _build_split  # noqa: E402

SEEDS = (7, 42)


def main() -> int:
    cfg = Config(n_rows=500, n_cols=8, forest_estimators=12, mice_max_iter=4)
    cells = [(d, m, s) for d in DGP_NAMES for m in MECHANISMS for s in SEEDS]

    base_vals: list[float] = []
    for dgp, mech, seed in cells:
        sp = _build_split(dgp, mech, seed, cfg)
        em = GaussianEMImputer(seed=seed).fit(sp.X_train)
        _, _ = column_nrmse(sp.X_test_full, em.transform(sp.X_test), sp.mask_test)
        agg, _ = column_nrmse(sp.X_test_full, em.transform(sp.X_test), sp.mask_test)
        base_vals.append(agg)
    base = float(np.mean(base_vals))
    print(f"baseline em_gaussian (2 seeds): {base:.4f}\n")

    print(f"{'repeats':>7}{'margin':>8}{'fuse':>10}{'reduction':>12}")
    for rep in (1, 2, 3):
        for margin in (0.0, 0.02, 0.05):
            vals = []
            for dgp, mech, seed in cells:
                sp = _build_split(dgp, mech, seed, cfg)
                f = ImputeFuse(
                    probe_rate=cfg.probe_rate,
                    max_iter=cfg.fuse_max_iter,
                    forest_estimators=cfg.forest_estimators,
                    seed=seed,
                    margin=margin,
                    probe_repeats=rep,
                )
                f.fit(sp.X_train)
                agg, _ = column_nrmse(sp.X_test_full, f.transform(sp.X_test), sp.mask_test)
                vals.append(agg)
            m = float(np.mean(vals))
            print(f"{rep:>7}{margin:>8.2f}{m:>10.4f}{100 * (base - m) / base:>11.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
