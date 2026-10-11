"""Global deterministic seeding. Single entry point: set_all(seed).

Author: 晨星
"""

from __future__ import annotations

import os
import random

import numpy as np

# BLAS thread pinning must happen before heavy numeric work; setdefault so
# users may still override externally. Determinism requires single-thread BLAS.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")


def set_all(seed: int) -> np.random.Generator:
    """Seed every RNG source once; return the canonical numpy Generator."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    return np.random.default_rng(seed)
