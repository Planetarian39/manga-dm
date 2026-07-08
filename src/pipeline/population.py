"""Stage 2 population-model orchestration."""

from __future__ import annotations

from pathlib import Path

from src.models.population import (
    fit_m200_c_mcmc,
    load_m200_c_fit_results,
    save_m200_c_fit_results,
    _set_result_dir,
)
from src.pipeline.selection import prepare_m200_c_data as get_m200_c_data
from src.stats.psis import compute_psis_importance_diagnostics
from src.viz.posterior import plot_population_posterior_diagnostics


def fit_m200_c_population(
    *,
    quality_cut: str | None = "recommended",
    result_dir_override: str | Path | None = None,
    ifu_ids: list[str] | None = None,
    use_gmm: bool = True,
    use_samples: bool = False,
    sample_cap: int | None = None,
    dataset_label: str = "all",
):
    """Load Stage 2 data and run the current population c-M200 fit."""
    _set_result_dir(result_dir_override)
    data = get_m200_c_data(
        result_dir_override=result_dir_override,
        ifu_ids=ifu_ids,
        quality_cut=quality_cut,
    )
    if not data:
        print("Failed to load data. Exiting.")
        return None

    M200 = data.get("M200")
    c = data.get("c")
    if M200 is None or c is None:
        print("M200/c columns are missing from the Stage 2 input table. Exiting.")
        return None

    sample_m200 = data.get("log10_M200_posterior_samples") if use_samples else None
    sample_c = data.get("log10_c_posterior_samples") if use_samples else None
    sample_m200_prior_mu = data.get("log10_M200_prior_mu") if use_samples else None
    sample_m200_prior_sigma = data.get("log10_M200_prior_sigma") if use_samples else None
    sample_m200_prior_lower = data.get("log10_M200_prior_lower") if use_samples else None
    sample_m200_prior_upper = data.get("log10_M200_prior_upper") if use_samples else None
    sample_c_prior_mu = data.get("log10_c_prior_mu") if use_samples else None
    sample_c_prior_sigma = data.get("log10_c_prior_sigma") if use_samples else None

    log10_gmm_weights = data.get("log10_gmm_weights") if use_gmm else None
    log10_gmm_means = data.get("log10_gmm_means") if use_gmm else None
    log10_gmm_covariances = data.get("log10_gmm_covariances") if use_gmm else None

    print("\n# 1. Fitting All Data")
    print("\nUsing MCMC for fitting...")
    fit_results = fit_m200_c_mcmc(
        M200,
        c,
        log10_M200_posterior_samples=sample_m200,
        log10_c_posterior_samples=sample_c,
        log10_M200_prior_mu=sample_m200_prior_mu,
        log10_M200_prior_sigma=sample_m200_prior_sigma,
        log10_M200_prior_lower=sample_m200_prior_lower,
        log10_M200_prior_upper=sample_m200_prior_upper,
        log10_c_prior_mu=sample_c_prior_mu,
        log10_c_prior_sigma=sample_c_prior_sigma,
        log10_gmm_weights=log10_gmm_weights,
        log10_gmm_means=log10_gmm_means,
        log10_gmm_covariances=log10_gmm_covariances,
        sample_cap=sample_cap,
        dataset_label=dataset_label,
        posterior_diagnostics_callback=plot_population_posterior_diagnostics,
    )
    if fit_results:
        save_m200_c_fit_results(fit_results, result_dir_override=result_dir_override)
    return fit_results


def run_m200_c_psis_diagnostics(
    *,
    quality_cut: str | None = "recommended",
    result_dir_override: str | Path | None = None,
    ifu_ids: list[str] | None = None,
    sample_cap: int | None = None,
) -> dict | None:
    """Run PSIS diagnostics using saved population-fit results."""
    fit_results = load_m200_c_fit_results(result_dir_override=result_dir_override)
    data = get_m200_c_data(
        result_dir_override=result_dir_override,
        ifu_ids=ifu_ids,
        quality_cut=quality_cut,
    )
    if not data:
        raise FileNotFoundError("No Stage 2 data available for PSIS diagnostics.")

    required = (
        "log10_M200_posterior_samples",
        "log10_c_posterior_samples",
        "log10_M200_prior_mu",
        "log10_M200_prior_sigma",
        "log10_M200_prior_lower",
        "log10_M200_prior_upper",
        "log10_c_prior_mu",
        "log10_c_prior_sigma",
    )
    missing = [key for key in required if data.get(key) is None]
    if missing:
        raise ValueError(
            "PSIS diagnostics require saved posterior samples and prior parameters; "
            f"missing: {', '.join(missing)}"
        )

    diagnostics = compute_psis_importance_diagnostics(
        log10_M200_posterior_samples=data["log10_M200_posterior_samples"],
        log10_c_posterior_samples=data["log10_c_posterior_samples"],
        log10_M200_prior_mu=data["log10_M200_prior_mu"],
        log10_M200_prior_sigma=data["log10_M200_prior_sigma"],
        log10_M200_prior_lower=data["log10_M200_prior_lower"],
        log10_M200_prior_upper=data["log10_M200_prior_upper"],
        log10_c_prior_mu=data["log10_c_prior_mu"],
        log10_c_prior_sigma=data["log10_c_prior_sigma"],
        fit_results=fit_results,
        plot_suffix="_all",
        sample_cap=sample_cap,
    )
    if diagnostics is None:
        raise ValueError(
            "PSIS diagnostics could not be computed. Ensure the saved population fit used posterior samples."
        )

    print(
        "PSIS diagnostics complete: "
        f"bad k={diagnostics.get('n_bad_k', 0)}, "
        f"warning k={diagnostics.get('n_warn_k', 0)}"
    )
    return diagnostics
