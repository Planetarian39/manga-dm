"""Gaussian Mixture Model fitting for log10(M200)–log10(c) posterior samples.

Extracted from ``src-orig/dm.py``.  Requires ``scikit-learn`` for BIC-based
component selection; falls back to single-Gaussian otherwise.
"""

from __future__ import annotations

import numpy as np


def fit_log10_mc_gmm(
    samples_2d: np.ndarray,
    max_components: int = 3,
    random_state: int = 42,
) -> dict:
    """Fit a GMM in log10(M200)–log10(c) space; return serialisable parameters.

    Parameters
    ----------
    samples_2d : ndarray, shape (n_samples, 2)
        Posterior samples.
    max_components : int
        Maximum number of Gaussian components to try.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    dict with keys:
        source, n_components, weights, means, covariances, bic, bic_by_n
    """
    samples = np.asarray(samples_2d, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != 2:
        raise ValueError("samples_2d must be of shape (n_samples, 2)")

    finite_mask = np.all(np.isfinite(samples), axis=1)
    samples = samples[finite_mask]
    n_samples = int(samples.shape[0])

    if n_samples < 5:
        mu = (
            np.mean(samples, axis=0)
            if n_samples > 0
            else np.array([np.nan, np.nan], dtype=float)
        )
        cov = (
            np.cov(samples, rowvar=False)
            if n_samples > 1
            else np.eye(2, dtype=float)
        )
        return {
            "source": "fallback_small_sample",
            "n_components": 1,
            "weights": [1.0],
            "means": [mu.tolist()],
            "covariances": [np.asarray(cov, dtype=float).tolist()],
            "bic": None,
            "bic_by_n": {},
        }

    try:
        import importlib
        sklearn_mixture = importlib.import_module("sklearn.mixture")
        GaussianMixture = getattr(sklearn_mixture, "GaussianMixture")
    except Exception:
        mu = np.mean(samples, axis=0)
        cov = np.cov(samples, rowvar=False)
        return {
            "source": "fallback_no_sklearn",
            "n_components": 1,
            "weights": [1.0],
            "means": [mu.tolist()],
            "covariances": [np.asarray(cov, dtype=float).tolist()],
            "bic": None,
            "bic_by_n": {},
        }

    best_model = None
    best_bic = np.inf
    bic_by_n = {}
    max_n = max(1, min(int(max_components), n_samples - 1))

    for n_comp in range(1, max_n + 1):
        try:
            model = GaussianMixture(
                n_components=n_comp,
                covariance_type="full",
                random_state=random_state,
                reg_covar=1e-6,
                n_init=5,
            )
            model.fit(samples)
            bic = float(model.bic(samples))
            bic_by_n[str(n_comp)] = bic
            if bic < best_bic:
                best_bic = bic
                best_model = model
        except Exception:
            continue

    if best_model is None:
        mu = np.mean(samples, axis=0)
        cov = np.cov(samples, rowvar=False)
        return {
            "source": "fallback_fit_failed",
            "n_components": 1,
            "weights": [1.0],
            "means": [mu.tolist()],
            "covariances": [np.asarray(cov, dtype=float).tolist()],
            "bic": None,
            "bic_by_n": bic_by_n,
        }

    return {
        "source": "gmm_bic",
        "n_components": int(best_model.n_components),
        "weights": best_model.weights_.astype(float).tolist(),
        "means": best_model.means_.astype(float).tolist(),
        "covariances": best_model.covariances_.astype(float).tolist(),
        "bic": float(best_bic),
        "bic_by_n": bic_by_n,
    }
