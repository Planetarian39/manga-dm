"""Shared analytic dark-matter relation helpers."""

from __future__ import annotations

import numpy as np

from src.config.constants import H_ACTUAL, M_PIVOT_H_INV

H_0 = H_ACTUAL


def log10_c_m200_relation_profile(
    M200: np.ndarray, log10_c0: float, alpha: float, h: float = H_0
) -> np.ndarray:
    M_pivot = M_PIVOT_H_INV / h
    log10_c = log10_c0 + alpha * (np.log10(M200) - np.log10(M_pivot))
    return 10 ** log10_c
