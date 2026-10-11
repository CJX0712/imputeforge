"""Imputer method pool: Tier-0 (sklearn) + Tier-1 (pure numpy offline).

Author: 晨星
"""

from imputeforge.imputers.base import BaseImputer, available_sklearn
from imputeforge.imputers.em import GaussianEMImputer
from imputeforge.imputers.knn import KNNImputerNP, KNNImputerSK
from imputeforge.imputers.mice import MiceForestImputer, MiceRidgeImputer
from imputeforge.imputers.simple import MeanImputer, MedianImputer
from imputeforge.imputers.softimpute import SoftImputer

__all__ = [
    "BaseImputer",
    "GaussianEMImputer",
    "KNNImputerNP",
    "KNNImputerSK",
    "MeanImputer",
    "MedianImputer",
    "MiceForestImputer",
    "MiceRidgeImputer",
    "SoftImputer",
    "available_sklearn",
    "build_pool",
]


def build_pool(seed: int, mice_max_iter: int = 4, forest_estimators: int = 12) -> list[BaseImputer]:
    """Build the method pool, degrading gracefully when sklearn is missing."""
    pool: list[BaseImputer] = [MeanImputer(), MedianImputer()]
    if available_sklearn():
        pool.append(KNNImputerSK(k=5))
        pool.append(MiceRidgeImputer(max_iter=mice_max_iter, seed=seed))
        pool.append(
            MiceForestImputer(max_iter=mice_max_iter, n_estimators=forest_estimators, seed=seed)
        )
    else:
        pool.append(KNNImputerNP(k=5))
    pool.append(GaussianEMImputer(seed=seed))
    pool.append(SoftImputer(seed=seed))
    return pool
