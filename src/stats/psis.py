"""Pareto-Smoothed Importance Sampling (PSIS) diagnostic tools.

Extracted from ``src-orig/m200.py``.  Implements the estimators from
Vehtari et al. (2019) / Zhang & Stephens (2009) for diagnosing the quality
of importance-weighted population-model inference.

Reference
---------
Vehtari, A., Simpson, D., Gelman, A., Yao, Y., & Gabry, J. (2019).
"Pareto Smoothed Importance Sampling". JMLR 25(72).
"""

from __future__ import annotations

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# 1.  GPD Pareto shape estimation  (Zhang & Stephens 2009)
# ═══════════════════════════════════════════════════════════════════════════


def gpdfit(z_sorted: np.ndarray) -> float:
    """Estimate the Pareto shape parameter *k* of the GPD via profile
    log-likelihood.

    Parameters
    ----------
    z_sorted : ndarray, 1-D
        Sorted (ascending), non-negative upper-tail raw importance
        weights, shifted so the minimum is 0.

    Returns
    -------
    k : float
        Pareto shape parameter.

        - k < 0.5  → IS estimate reliable
        - 0.5 ≤ k < 0.7 → marginal; apply caution
        - k ≥ 0.7  → IS estimate unreliable (rare-draw dominated)
    """
    M = len(z_sorted)
    if M < 5 or z_sorted[-1] <= 0.0:
        return 0.0

    m_grid = 30 + int(np.ceil(np.sqrt(M)))
    j = np.arange(1, m_grid + 1, dtype=float)
    b_ary = (1.0 - np.sqrt(m_grid / (j - 0.5))) / z_sorted[-1]
    b_ary = b_ary[b_ary < 0.0]
    if len(b_ary) == 0:
        return 0.0

    logml = M * (np.log(-b_ary) - 1.0) + np.array(
        [np.sum(np.log1p(-b * z_sorted)) for b in b_ary]
    )

    logml -= logml.max()
    w = np.exp(logml)
    w /= w.sum()
    b_hat = float(w @ b_ary)

    k = float(np.mean(np.log1p(-b_hat * z_sorted)))
    return max(k, 0.0)


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Effective sample size from log-weights
# ═══════════════════════════════════════════════════════════════════════════


def is_ess_from_log_weights(log_w: np.ndarray) -> float:
    """Compute the effective sample size (ESS) for importance weights.

    ESS = (Σ w_s)² / Σ w_s², equivalently 1 / Σ w̃_s² for normalised
    weights.  Range: (0, S] where S = len(log_w).

    Parameters
    ----------
    log_w : ndarray
        Unnormalised log importance weights.

    Returns
    -------
    float
        ESS value.
    """
    log_w = log_w - log_w.max()
    w = np.exp(log_w)
    ess = (w.sum() ** 2) / np.sum(w**2)
    return float(ess)


# ═══════════════════════════════════════════════════════════════════════════
# 3.  PSIS diagnostic wrapper  (caller must supply tensors)
# ═══════════════════════════════════════════════════════════════════════════


def compute_psis_importance_diagnostics(
    log10_M200_posterior_samples,
    log10_c_posterior_samples,
    log10_M200_prior_mu: np.ndarray,
    log10_M200_prior_sigma: np.ndarray,
    log10_M200_prior_lower: np.ndarray,
    log10_M200_prior_upper: np.ndarray,
    log10_c_prior_mu: np.ndarray,
    log10_c_prior_sigma: np.ndarray,
    fit_results: dict,
    plot_suffix: str = "",
    sample_cap: int | None = None,
    save_plots: bool = True,
) -> dict | None:
    """Compute PSIS diagnostics for a fitted population model.

    For each galaxy *i*, evaluates per-sample importance log-weights at
    the posterior-median population hyperparameters Φ*:

        log w_is = log p_pop(θ_is | Φ*) - log p_stage1(θ_is)

    Reports per-galaxy ESS and the Pareto shape k̂ from a GPD fit to the
    upper tail of the weight distribution.

    Parameters
    ----------
    log10_M200_posterior_samples : list[np.ndarray]
    log10_c_posterior_samples : list[np.ndarray]
    log10_M200_prior_mu, _sigma, _lower, _upper : np.ndarray
        Per-galaxy TruncNorm prior parameters for log10(M200).
    log10_c_prior_mu, log10_c_prior_sigma : np.ndarray
        Per-galaxy Normal prior parameters for log10(c).
    fit_results : dict
        Must contain ``likelihood_mode == "samples"`` and hyperparameter
        keys: ``M200_mu_median``, ``M200_sigma_median``,
        ``log10_c0_median``, ``alpha_median``, ``sigma_int_median``.
    plot_suffix : str
    sample_cap : int | None
    save_plots : bool

    Returns
    -------
    dict or None
        Keys: ``k_hat`` (array), ``ess`` (array), ``ess_frac`` (array),
        ``n_bad_k`` (int: k̂ ≥ 0.7), ``n_warn_k`` (int: 0.5 ≤ k̂ < 0.7).
        Returns None if inputs are insufficient.
    """
    # Import is deferred to avoid hard dependency on scipy before it's needed
    from scipy.stats import norm, truncnorm

    if fit_results is None or fit_results.get("likelihood_mode") != "samples":
        return None

    try:
        log10_c0 = float(fit_results["log10_c0_median"])
        alpha = float(fit_results["alpha_median"])
        sigma_int = float(fit_results["sigma_int_median"])
        M200_mu = float(fit_results["M200_mu_median"])
        M200_sigma = float(fit_results["M200_sigma_median"])
        nu_pop = float(fit_results.get("nu_pop_median", 30.0))
    except (KeyError, TypeError, ValueError) as exc:
        print(f"PSIS diagnostic: missing fit_results keys: {exc}")
        return None

    if sigma_int <= 0 or M200_sigma <= 0:
        return None

    n_galaxies = len(log10_M200_posterior_samples)

    k_hat = np.full(n_galaxies, np.nan)
    ess_array = np.full(n_galaxies, np.nan)
    ess_frac = np.full(n_galaxies, np.nan)

    # Pivot mass for the c-M200 scaling relation
    M_PIVOT_H_INV = 1e12
    H_0 = 0.674
    log10_M_pivot = np.log10(M_PIVOT_H_INV / H_0)

    for i in range(n_galaxies):
        m_samps = np.asarray(
            log10_M200_posterior_samples[i], dtype=float
        ).ravel()
        c_samps = np.asarray(
            log10_c_posterior_samples[i], dtype=float
        ).ravel()

        if sample_cap is not None and len(m_samps) > sample_cap:
            idx = np.random.default_rng(42 + i).choice(
                len(m_samps), size=sample_cap, replace=False
            )
            m_samps = m_samps[idx]
            c_samps = c_samps[idx]

        valid = (
            np.isfinite(m_samps)
            & np.isfinite(c_samps)
            & (m_samps >= log10_M200_prior_lower[i])
            & (m_samps <= log10_M200_prior_upper[i])
        )
        m_samps = m_samps[valid]
        c_samps = c_samps[valid]

        n = len(m_samps)
        if n < 5:
            continue

        # Stage-1 prior log-density
        a_i = (log10_M200_prior_lower[i] - log10_M200_prior_mu[i]) / max(
            log10_M200_prior_sigma[i], 1e-12
        )
        b_i = (log10_M200_prior_upper[i] - log10_M200_prior_mu[i]) / max(
            log10_M200_prior_sigma[i], 1e-12
        )
        log_p_log10_M200 = truncnorm.logpdf(
            m_samps, a_i, b_i,
            loc=log10_M200_prior_mu[i],
            scale=log10_M200_prior_sigma[i],
        )
        log_p_log10_c = norm.logpdf(
            c_samps,
            loc=log10_c_prior_mu[i],
            scale=log10_c_prior_sigma[i],
        )
        log_p_stage1 = log_p_log10_M200 + log_p_log10_c

        # Population-model conditional log-density at Φ* (posterior median)
        mu_c_given_m = log10_c0 + alpha * (m_samps - log10_M_pivot)
        log_p_pop = norm.logpdf(
            c_samps, loc=mu_c_given_m, scale=sigma_int
        ) + norm.logpdf(m_samps, loc=M200_mu, scale=M200_sigma)

        log_w = log_p_pop - log_p_stage1
        log_w -= log_w.max()

        ess_val = is_ess_from_log_weights(log_w)
        ess_array[i] = ess_val
        max_possible = float(n)
        ess_frac[i] = ess_val / max_possible if max_possible > 0 else np.nan

        # Sort raw weights for GPD tail fit
        weights_raw = np.exp(log_w)
        weights_raw_sorted = np.sort(weights_raw)
        tail_size = min(int(np.ceil(0.2 * n)), 3 * int(np.ceil(np.sqrt(n))))
        if tail_size >= 5:
            tail_weights = weights_raw_sorted[-tail_size:]
            z_tail = tail_weights - tail_weights[0]  # shift min to 0
            k_hat[i] = gpdfit(z_tail)
        else:
            k_hat[i] = 0.0

    n_bad_k = int(np.sum(k_hat >= 0.7))
    n_warn_k = int(np.sum((k_hat >= 0.5) & (k_hat < 0.7)))

    return {
        "k_hat": k_hat,
        "ess": ess_array,
        "ess_frac": ess_frac,
        "n_bad_k": n_bad_k,
        "n_warn_k": n_warn_k,
    }
