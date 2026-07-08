"""Population-level c–M200 relation inference model.

Migrated from ``src-orig/m200.py``.  Contains the hierarchical Bayesian
model ``fit_m200_c_mcmc`` and its direct helpers, plus literature
reference curves.

The PyMC model code in ``fit_m200_c_mcmc`` is **preserved verbatim**.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from scipy.stats import norm, truncnorm
from scipy.optimize import brentq

# New layered imports
from src.config.constants import (
    M_PIVOT_H_INV,
    LOG10_C0_DM14, ALPHA_DM14, LOG10_C_SIGMA_DM14,
    LOG10_C0_LI20, LOG10_C0_SIGMA_LI20, ALPHA_LI20,
    ALPHA_SIGMA_LI20, LOG10_C_SCATTER_LI20,
    LOG10_C0_YASIN23, LOG10_C0_SIGMA_YASIN23, ALPHA_YASIN23,
    ALPHA_SIGMA_YASIN23, LOG10_C_SCATTER_YASIN23,
    LOG10_C0_PRIOR_MEAN, LOG10_C0_PRIOR_SIGMA,
    ALPHA_PRIOR_MEAN, ALPHA_PRIOR_SIGMA,
    LOG_SIGMA_INT_PRIOR_MEAN, LOG_SIGMA_INT_PRIOR_SIGMA,
    NU_POP_PRIOR_ALPHA, NU_POP_PRIOR_BETA,
    DEFENSIVE_IS_EPSILON,
)
from src.config.settings import settings
from src.models.relations import H_0, log10_c_m200_relation_profile
from src.stats.arviz_compat import (
    ensure_arviz_compat,
    get_arviz_api,
    get_az,
    get_summary_interval_columns as get_summary_eti_columns,
    require_pymc_stack as _require_pymc_stack,
    summary_with_compat,
)
from src.stats.psis import (
    gpdfit as _gpdfit,
    is_ess_from_log_weights as _is_ess_from_log_weights,
)

az = None
_get_az = get_az

# ── Config / constant shims ──────────────────────────────────────────
HDI_PROB1 = settings.HDI_PROB1
HDI_PROB2 = settings.HDI_PROB2
NFW_PARAM_CM200_FILENAME = settings.nfw_param_cm200_filename
NFW_PARAM_CM200_SAMPLE_FILENAME = settings.nfw_param_cm200_sample_filename
M200_C_FIT_RESULTS_FILENAME = "m200_c_fit_results.json"

root_dir = settings.root_dir
data_dir = settings.data_dir
result_dir = settings.result_dir


def _sync_settings_paths(
    result_dir_override: str | Path | None = None,
    *,
    create: bool = False,
) -> Path:
    """Refresh legacy-style path globals from the current settings object."""
    global root_dir, data_dir, result_dir

    root_dir = settings.root_dir
    data_dir = settings.data_dir
    result_dir = settings.resolve_result_dir(result_dir_override)
    if create:
        result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir


def _resolve_result_dir(result_dir_override: str | Path | None = None) -> Path:
    return settings.resolve_result_dir(result_dir_override)


def _set_result_dir(result_dir_override: str | Path | None = None) -> Path:
    return _sync_settings_paths(result_dir_override, create=True)


def _json_scalar_results(fit_results: dict) -> dict:
    scalar_results = {}
    for key, value in fit_results.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            scalar_results[key] = value
            continue
        if isinstance(value, np.generic):
            scalar_results[key] = value.item()
    return scalar_results


def save_m200_c_fit_results(
    fit_results: dict,
    result_dir_override: str | Path | None = None,
    filename: str = M200_C_FIT_RESULTS_FILENAME,
) -> Path:
    """Persist scalar population-fit results for later diagnostics."""
    active_result_dir = _set_result_dir(result_dir_override)
    output_path = active_result_dir / filename
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(_json_scalar_results(fit_results), handle, indent=2, sort_keys=True)
    return output_path


def load_m200_c_fit_results(
    result_dir_override: str | Path | None = None,
    filename: str = M200_C_FIT_RESULTS_FILENAME,
) -> dict:
    """Load persisted scalar population-fit results."""
    result_path = _resolve_result_dir(result_dir_override) / filename
    if not result_path.exists():
        raise FileNotFoundError(
            f"population fit results not found: {result_path}. Run 'manga stage2 --fit' first."
        )
    with open(result_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def reference_log10_c_band(
    M200: np.ndarray,
    log10_c0: float,
    alpha: float,
    log10_c_scatter: float,
    log10_c0_sigma: float = 0.0,
    alpha_sigma: float = 0.0,
    sigma_scale: float = 2.0,
    h: float = H_0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = log10_c_m200_relation_profile(M200, log10_c0, alpha, h=h)
    log10_center = np.log10(center)

    if log10_c0_sigma > 0 or alpha_sigma > 0:
        curves = np.vstack(
            [
                log10_c_m200_relation_profile(
                    M200,
                    log10_c0 + log10_c0_sign * sigma_scale * log10_c0_sigma,
                    alpha + alpha_sign * sigma_scale * alpha_sigma,
                    h=h,
                )
                for log10_c0_sign in (-1.0, 1.0)
                for alpha_sign in (-1.0, 1.0)
            ]
        )
        log10_low = np.nanmin(np.log10(curves), axis=0)
        log10_high = np.nanmax(np.log10(curves), axis=0)
    else:
        log10_low = log10_center.copy()
        log10_high = log10_center.copy()

    if log10_c_scatter > 0:
        log10_low = log10_low - sigma_scale * log10_c_scatter
        log10_high = log10_high + sigma_scale * log10_c_scatter

    return center, 10 ** log10_low, 10 ** log10_high

def _prepare_gmm_tensors(
    log10_gmm_weights: np.ndarray,
    log10_gmm_means: np.ndarray,
    log10_gmm_covariances: np.ndarray,
    log10_M_pivot: float,
):
    """
    Pad per-galaxy GMM parameters into dense arrays for vectorized likelihood evaluation.

    Returns arrays of shape:
      weights: (N, K)
      means: (N, K, 2)
      covariances: (N, K, 2, 2)
      mask: (N, K)
    """
    n_gal = len(log10_gmm_weights)
    max_components = max(len(np.asarray(w).reshape(-1)) for w in log10_gmm_weights)

    weights = np.zeros((n_gal, max_components), dtype=float)
    means = np.zeros((n_gal, max_components, 2), dtype=float)
    covariances = np.repeat(
        np.eye(2, dtype=float)[None, None, :, :], n_gal * max_components, axis=0
    )
    covariances = covariances.reshape(n_gal, max_components, 2, 2)
    component_mask = np.zeros((n_gal, max_components), dtype=bool)

    for i in range(n_gal):
        weights_i = np.asarray(log10_gmm_weights[i], dtype=float).reshape(-1)
        means_i = np.asarray(log10_gmm_means[i], dtype=float)
        covs_i = np.asarray(log10_gmm_covariances[i], dtype=float)
        n_comp = len(weights_i)

        weights_i = np.clip(weights_i, 1e-12, None)
        weights_i = weights_i / np.sum(weights_i)

        means_i_shifted = means_i.copy()
        means_i_shifted[:, 0] -= log10_M_pivot

        weights[i, :n_comp] = weights_i
        means[i, :n_comp, :] = means_i_shifted
        covariances[i, :n_comp, :, :] = covs_i
        component_mask[i, :n_comp] = True

    return weights, means, covariances, component_mask


def _prepare_sample_posterior_tensors(
    log10_M200_posterior_samples: np.ndarray,
    log10_c_posterior_samples: np.ndarray,
    log10_M200_prior_mu: np.ndarray,
    log10_M200_prior_sigma: np.ndarray,
    log10_M200_prior_lower: np.ndarray,
    log10_M200_prior_upper: np.ndarray,
    log10_c_prior_mu: np.ndarray,
    log10_c_prior_sigma: np.ndarray,
    log10_M_pivot: float,
    sample_cap: int | None = None,
):
    """
    Prepare per-galaxy posterior samples and their first-stage prior log-density.
    The stage-two likelihood uses E_q[p_pop(theta)/p_stage1(theta)], where q is the
    saved first-stage posterior sample distribution.
    """
    if sample_cap is not None:
        sample_cap = int(sample_cap)
        if sample_cap < 1:
            raise ValueError("sample_cap must be at least 1 when provided")

    def _select_sample_indices(n_samples: int) -> np.ndarray:
        if sample_cap is None or n_samples <= sample_cap:
            return np.arange(n_samples, dtype=int)
        return np.linspace(0, n_samples - 1, num=sample_cap, dtype=int)

    n_gal = len(log10_M200_posterior_samples)
    max_samples = max(
        min(len(np.asarray(samples).reshape(-1)), sample_cap)
        if sample_cap is not None
        else len(np.asarray(samples).reshape(-1))
        for samples in log10_M200_posterior_samples
    )

    sample_points = np.zeros((n_gal, max_samples, 2), dtype=float)
    sample_log_prior = np.full((n_gal, max_samples), -np.inf, dtype=float)
    sample_mask = np.zeros((n_gal, max_samples), dtype=bool)

    for i in range(n_gal):
        log10_m200 = np.asarray(log10_M200_posterior_samples[i], dtype=float).reshape(-1)
        log10_c = np.asarray(log10_c_posterior_samples[i], dtype=float).reshape(-1)
        n_samples = min(len(log10_m200), len(log10_c))
        if n_samples == 0:
            continue

        selected_idx = _select_sample_indices(n_samples)
        log10_m200 = log10_m200[selected_idx]
        log10_c = log10_c[selected_idx]
        n_samples = len(selected_idx)

        sample_points[i, :n_samples, 0] = log10_m200 - log10_M_pivot
        sample_points[i, :n_samples, 1] = log10_c

        a = (
            (float(log10_M200_prior_lower[i]) - float(log10_M200_prior_mu[i]))
            / float(log10_M200_prior_sigma[i])
        )
        b = (
            (float(log10_M200_prior_upper[i]) - float(log10_M200_prior_mu[i]))
            / float(log10_M200_prior_sigma[i])
        )
        log_prior_m200 = truncnorm.logpdf(
            log10_m200,
            a=a,
            b=b,
            loc=float(log10_M200_prior_mu[i]),
            scale=float(log10_M200_prior_sigma[i]),
        )
        log_prior_c = norm.logpdf(
            log10_c,
            loc=float(log10_c_prior_mu[i]),
            scale=float(log10_c_prior_sigma[i]),
        )
        sample_log_prior[i, :n_samples] = log_prior_m200 + log_prior_c
        sample_mask[i, :n_samples] = True

    return sample_points, sample_log_prior, sample_mask


def _sample_inputs_are_usable(
    sample_m200: np.ndarray | None,
    sample_c: np.ndarray | None,
    prior_mu: np.ndarray | None,
    prior_sigma: np.ndarray | None,
    prior_lower: np.ndarray | None,
    prior_upper: np.ndarray | None,
    c_prior_mu: np.ndarray | None,
    c_prior_sigma: np.ndarray | None,
) -> bool:
    if (
        sample_m200 is None
        or sample_c is None
        or prior_mu is None
        or prior_sigma is None
        or prior_lower is None
        or prior_upper is None
        or c_prior_mu is None
        or c_prior_sigma is None
    ):
        return False

    if len(sample_m200) == 0 or len(sample_c) == 0:
        return False

    for i, (m200_s, c_s) in enumerate(zip(sample_m200, sample_c)):
        if m200_s is None or c_s is None:
            return False

        try:
            m200_arr = np.asarray(m200_s, dtype=float).reshape(-1)
            c_arr = np.asarray(c_s, dtype=float).reshape(-1)
        except Exception:
            return False

        if (
            m200_arr.size == 0
            or c_arr.size == 0
            or m200_arr.size != c_arr.size
            or not np.all(np.isfinite(m200_arr))
            or not np.all(np.isfinite(c_arr))
            or not np.isfinite(prior_mu[i])
            or not np.isfinite(prior_sigma[i])
            or not np.isfinite(prior_lower[i])
            or not np.isfinite(prior_upper[i])
            or not np.isfinite(c_prior_mu[i])
            or not np.isfinite(c_prior_sigma[i])
            or float(prior_sigma[i]) <= 0
            or float(c_prior_sigma[i]) <= 0
            or float(prior_lower[i]) >= float(prior_upper[i])
        ):
            return False

    return True


def _build_valid_sample_mask(
    log10_M200_posterior_samples: np.ndarray,
    log10_c_posterior_samples: np.ndarray,
    log10_M200_prior_mu: np.ndarray,
    log10_M200_prior_sigma: np.ndarray,
    log10_M200_prior_lower: np.ndarray,
    log10_M200_prior_upper: np.ndarray,
    log10_c_prior_mu: np.ndarray,
    log10_c_prior_sigma: np.ndarray,
) -> np.ndarray:
    return np.array(
        [
            (
                m200_s is not None
                and c_s is not None
                and len(np.asarray(m200_s).reshape(-1)) >= 5
                and len(np.asarray(c_s).reshape(-1)) >= 5
                and len(np.asarray(m200_s).reshape(-1))
                == len(np.asarray(c_s).reshape(-1))
                and np.all(np.isfinite(np.asarray(m200_s, dtype=float)))
                and np.all(np.isfinite(np.asarray(c_s, dtype=float)))
                and np.isfinite(log10_M200_prior_mu[i])
                and np.isfinite(log10_M200_prior_sigma[i])
                and np.isfinite(log10_M200_prior_lower[i])
                and np.isfinite(log10_M200_prior_upper[i])
                and np.isfinite(log10_c_prior_mu[i])
                and np.isfinite(log10_c_prior_sigma[i])
                and float(log10_M200_prior_sigma[i]) > 0
                and float(log10_c_prior_sigma[i]) > 0
                and float(log10_M200_prior_lower[i]) < float(log10_M200_prior_upper[i])
            )
            for i, (m200_s, c_s) in enumerate(
                zip(log10_M200_posterior_samples, log10_c_posterior_samples)
            )
        ],
        dtype=bool,
    )


def _is_valid_gmm_entry(weights_entry, means_entry, cov_entry) -> bool:
    try:
        weights_array = np.asarray(weights_entry, dtype=float).reshape(-1)
        means_array = np.asarray(means_entry, dtype=float)
        cov_array = np.asarray(cov_entry, dtype=float)
    except Exception:
        return False

    return (
        weights_entry is not None
        and means_entry is not None
        and cov_entry is not None
        and np.ndim(weights_array) == 1
        and np.shape(means_array)[-1] == 2
        and np.shape(cov_array)[-2:] == (2, 2)
        and len(weights_array) == np.shape(means_array)[0] == np.shape(cov_array)[0]
        and len(weights_array) >= 1
        and np.all(np.isfinite(weights_array))
        and np.all(np.isfinite(means_array))
        and np.all(np.isfinite(cov_array))
        and np.sum(weights_array) > 0
    )


def _build_valid_gmm_mask(
    log10_gmm_weights: np.ndarray,
    log10_gmm_means: np.ndarray,
    log10_gmm_covariances: np.ndarray,
) -> np.ndarray:
    return np.array(
        [
            _is_valid_gmm_entry(weights, means, covariances)
            for weights, means, covariances in zip(
                log10_gmm_weights,
                log10_gmm_means,
                log10_gmm_covariances,
            )
        ],
        dtype=bool,
    )


def _compute_dataset_tag(dataset_label: str) -> tuple[str, str]:
    label = str(dataset_label).strip() or "all"
    tag = label.replace(" ", "_").replace(">=", "ge").replace("<", "lt")
    return label, tag


def _compute_linear_fit(
    M200: np.ndarray,
    c: np.ndarray,
    log10_M_pivot: float,
    mask: np.ndarray | None = None,
) -> tuple[float | None, float | None]:
    if mask is not None:
        if np.sum(mask) < 3:
            return None, None
        M200 = M200[mask]
        c = c[mask]

    if len(M200) < 3:
        return None, None

    log10_M = np.log10(M200)
    log10_c = np.log10(c)
    x = log10_M - log10_M_pivot
    X = np.column_stack([x, np.ones_like(x)])
    coeffs, *_ = np.linalg.lstsq(X, log10_c, rcond=None)
    return coeffs[1], coeffs[0]


def fit_m200_c_mcmc(
    M200_obs: np.ndarray,
    c_obs: np.ndarray,
    log10_M200_posterior_samples: np.ndarray = None,
    log10_c_posterior_samples: np.ndarray = None,
    log10_M200_prior_mu: np.ndarray = None,
    log10_M200_prior_sigma: np.ndarray = None,
    log10_M200_prior_lower: np.ndarray = None,
    log10_M200_prior_upper: np.ndarray = None,
    log10_c_prior_mu: np.ndarray = None,
    log10_c_prior_sigma: np.ndarray = None,
    log10_gmm_weights: np.ndarray = None,
    log10_gmm_means: np.ndarray = None,
    log10_gmm_covariances: np.ndarray = None,
    sample_cap: int | None = None,
    dataset_label: str = "all",
    verbose: bool = True,
    posterior_diagnostics_callback=None,
):
    """
    Fit the non-linear c-M200 relation using a Hierarchical Bayesian Model (HBM).
    The observation likelihood is provided by either saved posterior samples or GMM summaries.
    """
    pm, pt = _require_pymc_stack()
    az_api = _get_az()
    dataset_label, dataset_tag = _compute_dataset_tag(dataset_label)
    print(f"\n--- Fitting using Hierarchical Bayesian Model (HBM) [{dataset_label}] ---")

    M200_obs = np.asarray(M200_obs, dtype=float)
    c_obs = np.asarray(c_obs, dtype=float)
    valid_mask = np.isfinite(M200_obs) & np.isfinite(c_obs) & (M200_obs > 0) & (c_obs > 0)

    has_gmm_input = (
        log10_gmm_weights is not None
        and log10_gmm_means is not None
        and log10_gmm_covariances is not None
    )
    has_sample_input = (
        log10_M200_posterior_samples is not None
        and log10_c_posterior_samples is not None
        and log10_M200_prior_mu is not None
        and log10_M200_prior_sigma is not None
        and log10_M200_prior_lower is not None
        and log10_M200_prior_upper is not None
        and log10_c_prior_mu is not None
        and log10_c_prior_sigma is not None
    )

    valid_sample_mask = None
    if has_sample_input:
        valid_sample_mask = _build_valid_sample_mask(
            log10_M200_posterior_samples,
            log10_c_posterior_samples,
            log10_M200_prior_mu,
            log10_M200_prior_sigma,
            log10_M200_prior_lower,
            log10_M200_prior_upper,
            log10_c_prior_mu,
            log10_c_prior_sigma,
        )

    valid_gmm_mask = None
    if has_gmm_input:
        valid_gmm_mask = _build_valid_gmm_mask(
            log10_gmm_weights,
            log10_gmm_means,
            log10_gmm_covariances,
        )

    use_sample_likelihood = (
        has_sample_input and valid_sample_mask is not None and np.any(valid_sample_mask)
    )
    use_gmm_likelihood = False

    if use_sample_likelihood:
        valid_mask = valid_mask & valid_sample_mask
    elif has_gmm_input:
        valid_mask = valid_mask & valid_gmm_mask
        use_gmm_likelihood = True

    if not np.all(valid_mask):
        msg = "mu/cov"
        if use_sample_likelihood:
            msg = "mu/cov/posterior samples"
        elif use_gmm_likelihood:
            msg = "mu/cov/GMM"

        print(f"Warning: Dropping {np.sum(~valid_mask)} points due to invalid {msg} data.")
        M200_obs = M200_obs[valid_mask]
        c_obs = c_obs[valid_mask]

        if has_sample_input:
            log10_M200_posterior_samples = log10_M200_posterior_samples[valid_mask]
            log10_c_posterior_samples = log10_c_posterior_samples[valid_mask]
            log10_M200_prior_mu = np.asarray(log10_M200_prior_mu, dtype=float)[valid_mask]
            log10_M200_prior_sigma = np.asarray(log10_M200_prior_sigma, dtype=float)[valid_mask]
            log10_M200_prior_lower = np.asarray(log10_M200_prior_lower, dtype=float)[valid_mask]
            log10_M200_prior_upper = np.asarray(log10_M200_prior_upper, dtype=float)[valid_mask]
            log10_c_prior_mu = np.asarray(log10_c_prior_mu, dtype=float)[valid_mask]
            log10_c_prior_sigma = np.asarray(log10_c_prior_sigma, dtype=float)[valid_mask]

        if has_gmm_input:
            log10_gmm_weights = log10_gmm_weights[valid_mask]
            log10_gmm_means = log10_gmm_means[valid_mask]
            log10_gmm_covariances = log10_gmm_covariances[valid_mask]

    use_sample_likelihood = _sample_inputs_are_usable(
        log10_M200_posterior_samples,
        log10_c_posterior_samples,
        log10_M200_prior_mu,
        log10_M200_prior_sigma,
        log10_M200_prior_lower,
        log10_M200_prior_upper,
        log10_c_prior_mu,
        log10_c_prior_sigma,
    )
    use_gmm_likelihood = (
        not use_sample_likelihood
        and has_gmm_input
        and log10_gmm_weights is not None
        and log10_gmm_means is not None
        and log10_gmm_covariances is not None
        and len(log10_gmm_weights) == len(M200_obs)
    )

    if len(M200_obs) < 3:
        print("Not enough valid data points for MCMC fitting.")
        return None

    if not use_sample_likelihood and not use_gmm_likelihood:
        print("No valid sample or GMM likelihood inputs available for MCMC fitting.")
        return None

    N_galaxies = len(M200_obs)
    log10_M_pivot = np.log10(M_PIVOT_H_INV / H_0)

    log10_M200_obs = np.log10(M200_obs)
    log10_c_obs = np.log10(c_obs)
    M200 = M200_obs
    c = c_obs
    mu_obs_shifted = np.column_stack([log10_M200_obs - log10_M_pivot, log10_c_obs])

    gmm_weights_padded = None
    gmm_means_padded = None
    gmm_covs_padded = None
    gmm_component_mask = None
    sample_points = None
    sample_log_prior = None
    sample_mask = None

    if use_sample_likelihood:
        sample_points, sample_log_prior, sample_mask = _prepare_sample_posterior_tensors(
            log10_M200_posterior_samples=log10_M200_posterior_samples,
            log10_c_posterior_samples=log10_c_posterior_samples,
            log10_M200_prior_mu=log10_M200_prior_mu,
            log10_M200_prior_sigma=log10_M200_prior_sigma,
            log10_M200_prior_lower=log10_M200_prior_lower,
            log10_M200_prior_upper=log10_M200_prior_upper,
            log10_c_prior_mu=log10_c_prior_mu,
            log10_c_prior_sigma=log10_c_prior_sigma,
            log10_M_pivot=log10_M_pivot,
            sample_cap=sample_cap,
        )

    if use_gmm_likelihood:
        (
            gmm_weights_padded,
            gmm_means_padded,
            gmm_covs_padded,
            gmm_component_mask,
        ) = _prepare_gmm_tensors(
            log10_gmm_weights=log10_gmm_weights,
            log10_gmm_means=log10_gmm_means,
            log10_gmm_covariances=log10_gmm_covariances,
            log10_M_pivot=log10_M_pivot,
        )

    with pm.Model() as model:
        M200_mu = pm.Normal("M200_mu", mu=np.mean(mu_obs_shifted[:, 0]), sigma=1.0)
        M200_sigma = pm.HalfNormal("M200_sigma", sigma=1.0)

        log10_c0_t = pm.Normal(
            "log10_c0", mu=LOG10_C0_PRIOR_MEAN, sigma=LOG10_C0_PRIOR_SIGMA
        )
        alpha_t = pm.Normal("alpha", mu=ALPHA_PRIOR_MEAN, sigma=ALPHA_PRIOR_SIGMA)
        log_sigma_int = pm.Normal(
            "log_sigma_int",
            mu=LOG_SIGMA_INT_PRIOR_MEAN,
            sigma=LOG_SIGMA_INT_PRIOR_SIGMA,
        )
        sigma_int = pm.Deterministic("sigma_int", pt.exp(log_sigma_int))
        # Degrees of freedom for the population Student-t model
        nu_pop = pm.Gamma("nu_pop", alpha=NU_POP_PRIOR_ALPHA, beta=NU_POP_PRIOR_BETA)

        mu_pop = pt.stack([M200_mu, log10_c0_t + alpha_t * M200_mu])
        cov_pop = pt.stack(
            [
                pt.stack([M200_sigma**2, alpha_t * M200_sigma**2]),
                pt.stack(
                    [
                        alpha_t * M200_sigma**2,
                        alpha_t**2 * M200_sigma**2 + sigma_int**2,
                    ]
                ),
            ]
        )

        if use_sample_likelihood:
            print("Using prior-corrected posterior-sample likelihood for galaxies.")
            print(
                f"Vectorizing sample likelihood for {N_galaxies} galaxies "
                f"with up to {sample_points.shape[1]} posterior samples each."
            )
            if sample_cap is not None:
                print(f"Posterior sample cap per galaxy: {sample_cap}")

            samples_t = pt.as_tensor_variable(sample_points)
            prior_logp_t = pt.as_tensor_variable(sample_log_prior)
            mask_t = pt.as_tensor_variable(sample_mask)

            # Factored Student-t population:
            #   m       ~ t_nu(M200_mu, M200_sigma)
            #   ell | m ~ t_nu(c0 + alpha*m, sigma_int)
            # log p_pop = log t(m) + log t(ell | m)
            _log_t_norm = (pt.gammaln((nu_pop + 1) / 2)
                          - pt.gammaln(nu_pop / 2)
                          - 0.5 * pt.log(nu_pop * np.pi))
            dm = samples_t[..., 0] - M200_mu
            dc_given_m = samples_t[..., 1] - (log10_c0_t + alpha_t * samples_t[..., 0])
            log_p_pop_m = (_log_t_norm - pt.log(M200_sigma)
                           - ((nu_pop + 1) / 2) * pt.log1p((dm / M200_sigma)**2 / nu_pop))
            log_p_pop_c = (_log_t_norm - pt.log(sigma_int)
                           - ((nu_pop + 1) / 2) * pt.log1p((dc_given_m / sigma_int)**2 / nu_pop))
            sample_log_pop = log_p_pop_m + log_p_pop_c

            # Truncated IS (Ionides 2008): cap each per-galaxy log weight at
            #   log τ = logsumexp(log w_s) − ½ log S  (i.e. τ = mean_w · √S)
            # This bounds weight variance while introducing only small bias.
            masked_log_w = pt.where(mask_t, sample_log_pop - prior_logp_t, -1e30)
            sample_count = pt.sum(mask_t, axis=1)
            log_sum_w = pt.logsumexp(masked_log_w, axis=1)           # (N_gal,)
            log_tau = log_sum_w - 0.5 * pt.log(sample_count)          # (N_gal,)
            masked_terms = pt.where(
                mask_t,
                pt.minimum(masked_log_w, log_tau[:, None]),
                -1e30,
            )
            loglike_per_galaxy = pt.logsumexp(masked_terms, axis=1) - pt.log(sample_count)

            print(f"Built truncated-IS sample likelihood for {N_galaxies} galaxies.")
            pm.Potential("obs_samples", pt.sum(loglike_per_galaxy))
        elif use_gmm_likelihood:
            print("Using GMM likelihood for galaxies.")
            print(
                f"Vectorizing GMM likelihood for {N_galaxies} galaxies "
                f"with up to {gmm_weights_padded.shape[1]} components each."
            )

            weights_t = pt.as_tensor_variable(gmm_weights_padded)
            means_t = pt.as_tensor_variable(gmm_means_padded)
            covs_t = pt.as_tensor_variable(gmm_covs_padded)
            mask_t = pt.as_tensor_variable(gmm_component_mask)

            cov_pop_bcast = pt.broadcast_to(cov_pop, covs_t.shape)
            total_cov = covs_t + cov_pop_bcast
            eye2_bcast = pt.broadcast_to(pt.eye(2, dtype=total_cov.dtype), covs_t.shape)
            total_cov = total_cov + 1e-6 * eye2_bcast

            mu_pop_bcast = pt.broadcast_to(mu_pop, means_t.shape)
            delta = means_t - mu_pop_bcast
            a = total_cov[..., 0, 0]
            b = total_cov[..., 0, 1]
            c_cov = total_cov[..., 1, 0]
            d = total_cov[..., 1, 1]

            det = pt.maximum(a * d - b * c_cov, 1e-12)
            dx = delta[..., 0]
            dy = delta[..., 1]
            maha = (d * dx * dx - (b + c_cov) * dx * dy + a * dy * dy) / det
            log_norm = 2.0 * np.log(2.0 * np.pi) + pt.log(det)
            comp_logp = -0.5 * (log_norm + maha)

            log_weights = pt.log(pt.maximum(weights_t, 1e-30))
            masked_comp_logp = pt.where(mask_t, comp_logp + log_weights, -1e30)
            loglike_per_galaxy = pt.logsumexp(masked_comp_logp, axis=1)

            print(f"Built vectorized GMM likelihood for {N_galaxies} galaxies.")
            pm.Potential("obs_gmm", pt.sum(loglike_per_galaxy))

        draws = 1000
        tune = 1000
        chains = min(4, os.cpu_count())
        target_accept = 0.95
        sampler = "nutpie"
        init = "jitter+adapt_full"

        sample_kwargs = dict(
            init=init,
            draws=draws,
            tune=tune,
            chains=chains,
            cores=1 if use_gmm_likelihood else chains,
            target_accept=target_accept,
            random_seed=42,
            progressbar=True,
            return_inferencedata=True,
            compute_convergence_checks=True,
        )

        print(
            f"Starting MCMC {sampler} sampling with {chains} chains, {draws} draws, "
            f"{tune} tune steps..."
        )
        trace = pm.sample(nuts_sampler=sampler, **sample_kwargs)
        print("Skipping compute_log_likelihood/LOO for Potential-based likelihood.")

    var_names = ["log10_c0", "alpha", "sigma_int", "nu_pop"]
    summary = summary_with_compat(
        az_api.summary,
        trace,
        var_names=var_names,
        round_to=4,
        stat_focus="median",
    )
    eti_low_col, eti_high_col = get_summary_eti_columns(summary)
    loo = None

    log10_c0_median = float(summary.loc["log10_c0", "median"])
    alpha_median = float(summary.loc["alpha", "median"])
    sigma_int_median = float(summary.loc["sigma_int", "median"])
    nu_pop_median = float(summary.loc["nu_pop", "median"])

    posterior = trace.posterior
    log10_c0_samples = posterior["log10_c0"].values.flatten()
    alpha_samples = posterior["alpha"].values.flatten()
    sigma_int_samples = posterior["sigma_int"].values.flatten()

    if posterior_diagnostics_callback is not None:
        posterior_diagnostics_callback(
            trace=trace,
            posterior=posterior,
            az_api=az_api,
            log10_c0_samples=log10_c0_samples,
            alpha_samples=alpha_samples,
            dataset_label=dataset_label,
            dataset_tag=dataset_tag,
            result_dir=result_dir,
            hdi_prob1=HDI_PROB1,
            hdi_prob2=HDI_PROB2,
        )
    log10_c0_eti_low = float(summary.loc["log10_c0", eti_low_col])
    log10_c0_eti_high = float(summary.loc["log10_c0", eti_high_col])
    alpha_eti_low = float(summary.loc["alpha", eti_low_col])
    alpha_eti_high = float(summary.loc["alpha", eti_high_col])
    sigma_int_eti_low = float(summary.loc["sigma_int", eti_low_col])
    sigma_int_eti_high = float(summary.loc["sigma_int", eti_high_col])

    c_pred = log10_c_m200_relation_profile(M200, log10_c0_median, alpha_median, h=H_0)
    residuals_median = c - c_pred
    rmse_median = np.sqrt(np.mean(residuals_median**2))
    nrmse_median = rmse_median / (np.mean(c) if np.mean(c) > 0 else 1)

    log10_residuals_median = np.log10(c) - np.log10(c_pred)
    log10_rmse = np.sqrt(np.mean(log10_residuals_median**2))
    log10_c_err_total = np.full_like(log10_c_obs, max(sigma_int_median, 1e-6), dtype=float)
    dof = int(max(len(M200) - 2, 1))
    chi2_median = np.sum((log10_residuals_median / log10_c_err_total) ** 2)
    redchi_median = chi2_median / dof

    if verbose:
        print(f"--------- MCMC M200-c fit results [{dataset_label}] ---------")
        if use_sample_likelihood:
            likelihood_label = "Posterior samples (prior-corrected MC)"
        elif use_gmm_likelihood:
            likelihood_label = "GMM mixture"
        else:
            likelihood_label = "GMM mixture"

        print(f" Likelihood      : {likelihood_label}")
        print("--- Summary ---")
        print(summary)
        print("--- LOO ---")
        if loo is None:
            print("Not available for current likelihood implementation.")
        else:
            print(loo)
        print("--- median ---")
        print(
            f" log10_c0 median: {log10_c0_median:.3f}, {HDI_PROB2:.0%} "
            f"ETI=[{log10_c0_eti_low:.3f}, {log10_c0_eti_high:.3f}]"
        )
        print(
            f" alpha median   : {alpha_median:.3f}, {HDI_PROB2:.0%} "
            f"ETI=[{alpha_eti_low:.3f}, {alpha_eti_high:.3f}]"
        )
        print(
            f" sigma_int      : {sigma_int_median:.3f}, {HDI_PROB2:.0%} "
            f"ETI=[{sigma_int_eti_low:.3f}, {sigma_int_eti_high:.3f}]"
        )
        print(f" nu_pop (t d.f.): {nu_pop_median:.2f}")
        print(f" Log10 RMSE     : {log10_rmse:.3f}")
        print(f" NRMSE (median) : {nrmse_median:.3f}")
        print(f" Reduced Chi2 (median) : {redchi_median:.3f}")
        print("------------------------------------------")

    return {
        "log10_c0_median": log10_c0_median,
        "alpha_median": alpha_median,
        "log10_c0_eti_low": log10_c0_eti_low,
        "log10_c0_eti_high": log10_c0_eti_high,
        "alpha_eti_low": alpha_eti_low,
        "alpha_eti_high": alpha_eti_high,
        "log10_rmse": log10_rmse,
        "sigma_int_median": sigma_int_median,
        "sigma_int_eti_low": sigma_int_eti_low,
        "sigma_int_eti_high": sigma_int_eti_high,
        "alpha_samples": alpha_samples,
        "log10_c0_samples": log10_c0_samples,
        "M200_mu_median": float(np.median(posterior["M200_mu"].values)),
        "M200_sigma_median": float(np.median(posterior["M200_sigma"].values)),
        "nu_pop_median": nu_pop_median,
        "likelihood_mode": (
            "samples" if use_sample_likelihood else "gmm" if use_gmm_likelihood else "gaussian"
        ),
        "dataset_label": dataset_label,
        "loo": loo,
    }


