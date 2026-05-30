"""Population-level c–M200 relation inference model.

Migrated from ``src-orig/m200.py``.  Contains the hierarchical Bayesian
model ``fit_m200_c_mcmc`` and its direct helpers, plus literature
reference curves.

The PyMC model code in ``fit_m200_c_mcmc`` is **preserved verbatim**.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import pymc as pm
import pytensor.tensor as pt

from scipy.stats import truncnorm
from scipy.optimize import brentq

# New layered imports
from src.config.constants import (
    H0_PHYS, H_ACTUAL, M_PIVOT_H_INV,
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
    COLOR_DM14, COLOR_LI20, COLOR_YASIN23,
    COLOR_HIGH_N, COLOR_LOW_N, COLOR_POSTERIOR_MEDIAN,
    COLOR_DATA_POINTS, COLOR_SIGMA_BAND, COLOR_HDI_BAND,
    QUALITY_FILTER_PRESETS,
)
from src.config.settings import settings
from src.data.results import (
    load_posterior_sample_map,
    merge_posterior_samples_file,
)
from src.stats.arviz_compat import (
    ensure_arviz_compat, get_arviz_api, get_az,
    require_pymc_stack, get_summary_interval_columns,
    summary_with_compat,
)
from src.stats.psis import (
    gpdfit as _gpdfit,
    is_ess_from_log_weights as _is_ess_from_log_weights,
    compute_psis_importance_diagnostics,
)
from src.viz.posterior import annotate_pair_marginals_m200

az = get_az()
_annotate_pair_marginals_m200 = annotate_pair_marginals_m200

# ── Config / constant shims ──────────────────────────────────────────
H_0 = H_ACTUAL  # the old code uses H_0
HDI_PROB1 = settings.HDI_PROB1
HDI_PROB2 = settings.HDI_PROB2
NFW_PARAM_CM200_FILENAME = settings.nfw_param_cm200_filename
NFW_PARAM_CM200_SAMPLE_FILENAME = settings.nfw_param_cm200_sample_filename

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


def log10_c_m200_relation_profile(
    M200: np.ndarray, log10_c0: float, alpha: float, h: float = H_0
) -> np.ndarray:
    M_pivot = M_PIVOT_H_INV / h
    log10_c = log10_c0 + alpha * (np.log10(M200) - np.log10(M_pivot))
    return 10 ** log10_c


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

def _parse_object_column(df: pd.DataFrame, column_name: str) -> np.ndarray | None:
    import ast

    if column_name not in df.columns:
        return None

    try:
        return np.array(
            [ast.literal_eval(x) if isinstance(x, str) else x for x in df[column_name]],
            dtype=object,
        )
    except Exception as e:
        print(f"Warning: Could not parse {column_name}: {e}")
        return None


def get_m200_c_data(
    result_dir_override: str | Path | None = None,
    ifu_ids: list[str] | None = None,
    nrmse_threshold: float | None = None,
    quality_cut: str | None = None,
    max_redchi: float | None = None,
    ppc_p_min: float | None = None,
    ppc_p_max: float | None = None,
    ppc_value_coverage_min: float | None = None,
    ppc_overlap_min: float | None = None,
    max_abs_c_m200_corr: float | None = None,
    filter_mass: str | None = None,
    filter_n: str | None = None,
):
    """
    Load data from the results CSV file.
    Returns a dict of arrays keyed by column name.
    """
    active_result_dir = _resolve_result_dir(result_dir_override)
    nfw_param_file = active_result_dir / NFW_PARAM_CM200_FILENAME
    if not nfw_param_file.exists():
        print(f"Warning: Data file {nfw_param_file} not found.")
        return None

    df = pd.read_csv(nfw_param_file, index_col=0)
    df.index = df.index.map(str)

    df = _filter_dataframe_by_success(df)
    df = _filter_dataframe_by_ifu_ids(df, ifu_ids)
    df = _filter_dataframe_by_nrmse(df, nrmse_threshold)
    df = _filter_dataframe_by_quality_metrics(
        df,
        quality_cut=quality_cut,
        max_redchi=max_redchi,
        ppc_p_min=ppc_p_min,
        ppc_p_max=ppc_p_max,
        ppc_value_coverage_min=ppc_value_coverage_min,
        ppc_overlap_min=ppc_overlap_min,
        max_abs_c_m200_corr=max_abs_c_m200_corr,
    )

    if df.empty:
        print("Warning: No successful fits found in data.")
        return None

    df = _attach_result_catalog_columns(df)
    df = _filter_dataframe_by_tertile_group(
        df,
        column="log10_mstar",
        tertile_label=filter_mass,
        option_name="--filter-mass",
        display_name="stellar-mass",
    )
    df = _filter_dataframe_by_tertile_group(
        df,
        column="sersic_n",
        tertile_label=filter_n,
        option_name="--filter-n",
        display_name="Sersic-n",
    )

    if df.empty:
        print("Warning: No successful fits remain after applying the requested filters.")
        return None

    log10_mstar = df["log10_mstar"].values if "log10_mstar" in df.columns else np.full(len(df), np.nan)
    sersic_n = df["sersic_n"].values if "sersic_n" in df.columns else np.zeros(len(df))
    nrmse = df["nrmse"].values if "nrmse" in df.columns else np.zeros(len(df))
    M200 = df["M200"].values if "M200" in df.columns else None
    c = df["c"].values if "c" in df.columns else None

    log10_gmm_source = (
        df["log10_gmm_source"].values if "log10_gmm_source" in df.columns else None
    )
    log10_gmm_n_components = (
        df["log10_gmm_n_components"].values
        if "log10_gmm_n_components" in df.columns
        else None
    )
    log10_gmm_weights = _parse_object_column(df, "log10_gmm_weights")
    log10_gmm_means = _parse_object_column(df, "log10_gmm_means")
    log10_gmm_covariances = _parse_object_column(df, "log10_gmm_covariances")
    log10_gmm_bic = df["log10_gmm_bic"].values if "log10_gmm_bic" in df.columns else None
    log10_gmm_bic_by_n = _parse_object_column(df, "log10_gmm_bic_by_n")

    sample_file = active_result_dir / NFW_PARAM_CM200_SAMPLE_FILENAME
    log10_M200_posterior_samples = np.array([None] * len(df), dtype=object)
    log10_c_posterior_samples = np.array([None] * len(df), dtype=object)
    sample_map = _load_posterior_sample_map(
        sample_file, plate_ifus=df.index.astype(str).tolist()
    )
    for idx, plate_ifu in enumerate(df.index):
        samples = sample_map.get(str(plate_ifu))
        if samples is None:
            continue
        log10_M200_posterior_samples[idx] = samples[0]
        log10_c_posterior_samples[idx] = samples[1]

    return {
        "plate_ifu": df.index.to_numpy(dtype=str),
        "log10_mstar": log10_mstar,
        "sersic_n": sersic_n,
        "nrmse": nrmse,
        "M200": M200,
        "c": c,
        "log10_gmm_source": log10_gmm_source,
        "log10_gmm_n_components": log10_gmm_n_components,
        "log10_gmm_weights": log10_gmm_weights,
        "log10_gmm_means": log10_gmm_means,
        "log10_gmm_covariances": log10_gmm_covariances,
        "log10_gmm_bic": log10_gmm_bic,
        "log10_gmm_bic_by_n": log10_gmm_bic_by_n,
        "log10_M200_posterior_samples": log10_M200_posterior_samples,
        "log10_c_posterior_samples": log10_c_posterior_samples,
        "log10_M200_prior_mu": (
            df["log10_M200_prior_mu"].values
            if "log10_M200_prior_mu" in df.columns
            else None
        ),
        "log10_M200_prior_sigma": (
            df["log10_M200_prior_sigma"].values
            if "log10_M200_prior_sigma" in df.columns
            else None
        ),
        "log10_M200_prior_lower": (
            df["log10_M200_prior_lower"].values
            if "log10_M200_prior_lower" in df.columns
            else None
        ),
        "log10_M200_prior_upper": (
            df["log10_M200_prior_upper"].values
            if "log10_M200_prior_upper" in df.columns
            else None
        ),
        "log10_c_prior_mu": (
            df["log10_c_prior_mu"].values if "log10_c_prior_mu" in df.columns else None
        ),
        "log10_c_prior_sigma": (
            df["log10_c_prior_sigma"].values
            if "log10_c_prior_sigma" in df.columns
            else None
        ),
    }


def _find_first_column_name(column_names, candidates: list[str]) -> str | None:
    lower_to_actual = {str(name).lower(): str(name) for name in column_names}
    for candidate in candidates:
        matched = lower_to_actual.get(str(candidate).lower())
        if matched is not None:
            return matched
    return None


def _table_column_to_array(table, candidates: list[str], dtype=float, fill_value=np.nan) -> np.ndarray:
    column_name = _find_first_column_name(getattr(table, "colnames", []), candidates)
    if column_name is None:
        return np.full(len(table), fill_value, dtype=dtype)

    column = table[column_name]
    try:
        masked_values = np.ma.asarray(column, dtype=dtype)
        values = np.ma.filled(masked_values, fill_value)
    except Exception:
        try:
            values = np.asarray(column, dtype=dtype)
        except Exception:
            values = np.asarray(column)
            return values.astype(dtype, copy=False)
    return np.asarray(values, dtype=dtype)


def _axis_ratio_to_inclination_deg(axis_ratio: np.ndarray, intrinsic_thickness: float = 0.2) -> np.ndarray:
    axis_ratio = np.asarray(axis_ratio, dtype=float)
    axis_ratio_sq = axis_ratio**2
    intrinsic_sq = intrinsic_thickness**2
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_i_sq = (axis_ratio_sq - intrinsic_sq) / (1.0 - intrinsic_sq)
    cos_i_sq = np.clip(cos_i_sq, 0.0, 1.0)
    return np.degrees(np.arccos(np.sqrt(cos_i_sq)))


def _build_base_sample_catalog() -> pd.DataFrame:
    fits_util = FitsUtil(data_dir)
    drpall_util = DrpallUtil(fits_util.get_drpall_file())
    table = drpall_util.get_all_fits()

    plate_ifu = _table_column_to_array(
        table,
        ["PLATEIFU", "plateifu", "PLATE_IFU", "plate_ifu"],
        dtype=str,
        fill_value="",
    )
    redshift = _table_column_to_array(table, ["NSA_Z", "nsa_z", "NSA_ZDIST", "nsa_zdist", "Z", "z"])
    stellar_mass = _table_column_to_array(table, ["NSA_ELPETRO_MASS", "nsa_elpetro_mass"])
    stellar_mass_alt = _table_column_to_array(table, ["NSA_SERSIC_MASS", "nsa_sersic_mass"])
    stellar_mass = np.where(np.isfinite(stellar_mass) & (stellar_mass > 0), stellar_mass, stellar_mass_alt)
    log10_mstar = np.full_like(stellar_mass, np.nan, dtype=float)
    valid_mass = np.isfinite(stellar_mass) & (stellar_mass > 0)
    log10_mstar[valid_mass] = np.log10(stellar_mass[valid_mass])

    axis_ratio = _table_column_to_array(table, ["NSA_SERSIC_BA", "nsa_elpetro_ba"])
    inclination = _axis_ratio_to_inclination_deg(axis_ratio)
    sersic_n = _table_column_to_array(table, ["NSA_SERSIC_N", "nsa_sersic_n"])

    catalog = pd.DataFrame(
        {
            "redshift": np.asarray(redshift, dtype=float),
            "log10_mstar": np.asarray(log10_mstar, dtype=float),
            "inclination": np.asarray(inclination, dtype=float),
            "sersic_n": np.asarray(sersic_n, dtype=float),
        },
        index=pd.Index(np.asarray(plate_ifu, dtype=str), name="plate_ifu"),
    )
    return catalog[~catalog.index.duplicated(keep="first")]


def _load_all_sample_catalog() -> pd.DataFrame:
    plate_ifu_file = data_dir / "plateifus.txt"
    if not plate_ifu_file.exists():
        raise FileNotFoundError(f"All-sample file not found: {plate_ifu_file}")

    with open(plate_ifu_file, "r", encoding="utf-8") as handle:
        plate_ifus = [line.strip() for line in handle if line.strip()]

    catalog = _build_base_sample_catalog()
    return catalog.reindex(pd.Index(plate_ifus, dtype=str, name="plate_ifu")).copy()


def _load_screened_sample_catalog() -> pd.DataFrame:
    rc_param_file = result_dir / "rc_param.csv"
    if not rc_param_file.exists():
        raise FileNotFoundError(f"Screened-sample file not found: {rc_param_file}")

    rc_df = pd.read_csv(rc_param_file, index_col=0)
    rc_df.index = rc_df.index.map(str)

    catalog = _build_base_sample_catalog()

    screened_catalog = catalog.reindex(rc_df.index).copy()
    if "inc_deg" in rc_df.columns:
        screened_catalog.loc[:, "inclination"] = rc_df.loc[screened_catalog.index, "inc_deg"].to_numpy(dtype=float)
    return screened_catalog


def _load_stage_result_table(
    result_dir_override: str | Path | None = None,
    ifu_ids: list[str] | None = None,
    success_only: bool = True,
) -> pd.DataFrame:
    nfw_param_file = _resolve_result_dir(result_dir_override) / NFW_PARAM_CM200_FILENAME
    if not nfw_param_file.exists():
        raise FileNotFoundError(f"Data file not found: {nfw_param_file}")

    df = pd.read_csv(nfw_param_file, index_col=0)
    df.index = df.index.map(str)
    if success_only:
        df = _filter_dataframe_by_success(df)
    return _filter_dataframe_by_ifu_ids(df, ifu_ids)


def generate_robustness_sample(
    n_sample: int = 60,
    nrmse_threshold: float | None = 0.15,
    n_bins: int = 10,
    output_filename: str | None = None,
    random_seed: int = 42,
    result_dir_override: str | Path | None = None,
    ifu_ids: list[str] | None = None,
    sample_cap: int | None = None,
    mode: str = "mass-parent",
) -> Path:
    """Draw a representative robustness subsample from the galaxies in ``ifu_ids``.

    Parameters
    ----------
    n_sample : int
        Target subsample size.
    nrmse_threshold : float | None
        Optional NRMSE threshold used to define the final sample. Ignored when
        ``ifu_ids`` is provided because the IFU list is treated as the explicit
        selection pool.
    n_bins : int
        Number of histogram bins used for ``*-parent`` modes.
    output_filename : str | None
        Output filename without directory.
    random_seed : int
        Random seed for reproducibility.
    result_dir_override : str | Path | None
        Override for the result directory.
    ifu_ids : list[str] | None
        IFU IDs from ``--ifu-file``. When provided, the pool is restricted to
        these IDs (intersected with NRMSE-passing galaxies). Required.
    sample_cap : int | None
        Maximum number of posterior samples to use per galaxy for
        ``psis-khat`` mode. Ignored by the other sampling modes.
    mode : str
        Sampling strategy. One of:

        ``random``
            Draw uniformly at random from the candidate IFU pool without
            replacement.

        ``mass-quintile``
            Draw uniformly across 5 stellar-mass quintiles of the pool.
        ``mass-parent``
            Weight pool to match the stellar-mass distribution of
            ``data/plateifus.txt``.
        ``sersic-quintile``
            Draw uniformly across 5 Sersic-n quintiles of the pool.
        ``sersic-parent``
            Weight pool to match the Sersic-n distribution of
            ``data/plateifus.txt``.
        ``psis-khat``
            Fit the population model on the candidate pool using saved posterior
            samples, compute per-galaxy PSIS diagnostics, and select the
            ``n_sample`` galaxies with the smallest Pareto-tail shape k-hat.

    Returns
    -------
    Path
        Absolute path to the written sample list.
    """
    _VALID_MODES = (
        "random",
        "mass-quintile",
        "mass-parent",
        "sersic-quintile",
        "sersic-parent",
        "psis-khat",
    )
    if mode not in _VALID_MODES:
        raise ValueError(f"--gen-sample-mode must be one of {_VALID_MODES}, got '{mode}'")
    if ifu_ids is None:
        raise ValueError("--ifu-file is required when --gen-sample is used.")

    if mode == "psis-khat":
        selected_ifus = _select_lowest_psis_khat_ifus(
            n_sample=n_sample,
            nrmse_threshold=nrmse_threshold,
            result_dir_override=result_dir_override,
            ifu_ids=ifu_ids,
            sample_cap=sample_cap,
        )
        n_selected = len(selected_ifus)
        if output_filename is None:
            output_filename = f"sample_{mode}_{n_selected}.txt"
        output_path = data_dir / output_filename
        with open(output_path, "w", encoding="utf-8") as fh:
            for ifu in selected_ifus:
                fh.write(f"{ifu}\n")

        print(f"[gen-sample] Wrote subsample IFU list to: {output_path}")
        return output_path

    rng = np.random.default_rng(random_seed)

    # ── Build the candidate pool from the explicit IFU selection ────────────
    result_table = _load_stage_result_table(result_dir_override=result_dir_override)
    success_mask = _build_success_mask(result_table)
    if nrmse_threshold is not None and "nrmse" in result_table.columns:
        final_mask = success_mask & (
            np.isfinite(result_table["nrmse"].to_numpy(dtype=float))
            & (result_table["nrmse"].to_numpy(dtype=float) <= float(nrmse_threshold))
        )
    else:
        final_mask = success_mask.copy()

    result_ifus = set(result_table.index[final_mask].tolist())
    pool_ifus = [ifu for ifu in ifu_ids if ifu in result_ifus]
    if len(pool_ifus) == 0:
        raise ValueError(
            "No galaxies from --ifu-file are available with successful stage-one results. "
            "Check --ifu-file and --result-dir."
        )

    screened_catalog = _load_screened_sample_catalog()
    pool_catalog = screened_catalog.reindex(
        [ifu for ifu in pool_ifus if ifu in screened_catalog.index]
    )
    pool_ifus_arr = np.asarray(pool_catalog.index.tolist())

    feature_col, feature_label = (
        ("log10_mstar", "log10_mstar")
        if mode.startswith("mass")
        else ("sersic_n", "sersic_n")
    )
    pool_feature = pool_catalog[feature_col].to_numpy(dtype=float)

    print(f"[gen-sample] Mode: {mode}")
    print(f"[gen-sample] Pool size (ifu-file ∩ filtered successful results): {len(pool_ifus_arr)}")

    # ── Sampling ─────────────────────────────────────────────────────────────
    deficit = 0
    if mode == "random":
        draw_count = min(n_sample, len(pool_ifus_arr))
        selected_ifus = rng.choice(pool_ifus_arr, size=draw_count, replace=False).tolist()

    elif mode.endswith("quintile"):
        # Uniform draw across 5 equal-count bins defined on the pool itself
        n_q = 5
        q_edges = np.quantile(pool_feature[np.isfinite(pool_feature)], np.linspace(0, 1, n_q + 1))
        q_edges[0] -= 1e-9
        q_edges[-1] += 1e-9
        target_per_bin = np.full(n_q, n_sample // n_q, dtype=int)
        for k in range(n_sample % n_q):
            target_per_bin[k] += 1

        selected_ifus: list[str] = []
        deficit = 0
        for b in range(n_q):
            lo, hi = q_edges[b], q_edges[b + 1]
            in_bin = np.where((pool_feature > lo) & (pool_feature <= hi))[0]
            need = target_per_bin[b] + deficit
            if in_bin.size == 0:
                deficit = need
                continue
            actual = min(need, in_bin.size)
            deficit = need - actual
            chosen = rng.choice(in_bin, size=actual, replace=False)
            selected_ifus.extend(pool_ifus_arr[chosen].tolist())

        feat_vals = pool_feature[np.isfinite(pool_feature)]
        print(
            f"[gen-sample] Pool {feature_label} range: "
            f"[{feat_vals.min():.3f}, {feat_vals.max():.3f}], "
            f"quintile edges: {np.round(q_edges, 3).tolist()}"
        )

    else:  # *-parent: weight pool to match plateifus.txt distribution
        all_catalog = _load_all_sample_catalog()
        parent_feature = all_catalog[feature_col].to_numpy(dtype=float)
        parent_feature = parent_feature[np.isfinite(parent_feature)]
        if parent_feature.size == 0:
            raise ValueError(f"No valid {feature_col} values in the parent sample.")

        feat_min = float(np.nanmin(parent_feature))
        feat_max = float(np.nanmax(parent_feature))
        bin_edges = np.linspace(feat_min, feat_max, n_bins + 1)

        parent_counts, _ = np.histogram(parent_feature, bins=bin_edges)
        parent_frac = parent_counts / parent_counts.sum()

        target_per_bin = np.floor(parent_frac * n_sample).astype(int)
        remainder = n_sample - target_per_bin.sum()
        order = np.argsort(-parent_frac)
        for k in range(int(remainder)):
            target_per_bin[order[k]] += 1

        selected_ifus = []
        deficit = 0
        for b in range(n_bins):
            lo, hi = bin_edges[b], bin_edges[b + 1]
            in_bin = np.where((pool_feature >= lo) & (pool_feature < hi))[0]
            if b == n_bins - 1:
                in_bin = np.where((pool_feature >= lo) & (pool_feature <= hi))[0]
            need = target_per_bin[b] + deficit
            if in_bin.size == 0:
                deficit = need
                continue
            actual = min(need, in_bin.size)
            deficit = need - actual
            chosen = rng.choice(in_bin, size=actual, replace=False)
            selected_ifus.extend(pool_ifus_arr[chosen].tolist())

        print(
            f"[gen-sample] Parent {feature_label} range: [{feat_min:.3f}, {feat_max:.3f}]"
        )

    # ── Fill any remaining deficit from unselected pool ──────────────────────
    if deficit > 0:
        already = set(selected_ifus)
        spare = [ifu for ifu in pool_ifus_arr.tolist() if ifu not in already]
        if spare:
            extra = rng.choice(spare, size=min(deficit, len(spare)), replace=False)
            selected_ifus.extend(extra.tolist())

    n_selected = len(selected_ifus)
    sel_feat = screened_catalog.reindex(selected_ifus)[feature_col].to_numpy(dtype=float)
    sel_feat_finite = sel_feat[np.isfinite(sel_feat)]
    print(
        f"[gen-sample] Target: {n_sample}, drawn: {n_selected}"
    )
    if sel_feat_finite.size > 0:
        print(
            f"[gen-sample] Subsample {feature_label} range: "
            f"[{sel_feat_finite.min():.3f}, {sel_feat_finite.max():.3f}], "
            f"median: {np.median(sel_feat_finite):.3f}"
        )

    if output_filename is None:
        output_filename = f"sample_{mode}_{n_selected}.txt"
    output_path = data_dir / output_filename
    with open(output_path, "w", encoding="utf-8") as fh:
        for ifu in selected_ifus:
            fh.write(f"{ifu}\n")

    print(f"[gen-sample] Wrote subsample IFU list to: {output_path}")
    return output_path


def _select_lowest_psis_khat_ifus(
    n_sample: int,
    nrmse_threshold: float | None,
    result_dir_override: str | Path | None = None,
    ifu_ids: list[str] | None = None,
    sample_cap: int | None = None,
) -> list[str]:
    data = get_m200_c_data(
        result_dir_override=result_dir_override,
        ifu_ids=ifu_ids,
        nrmse_threshold=nrmse_threshold,
    )
    if not data:
        raise ValueError("No galaxies available for PSIS-ranked sampling.")

    sample_m200 = data.get("log10_M200_posterior_samples")
    sample_c = data.get("log10_c_posterior_samples")
    prior_mu = data.get("log10_M200_prior_mu")
    prior_sigma = data.get("log10_M200_prior_sigma")
    prior_lower = data.get("log10_M200_prior_lower")
    prior_upper = data.get("log10_M200_prior_upper")
    prior_c_mu = data.get("log10_c_prior_mu")
    prior_c_sigma = data.get("log10_c_prior_sigma")

    if any(
        value is None
        for value in (
            sample_m200,
            sample_c,
            prior_mu,
            prior_sigma,
            prior_lower,
            prior_upper,
            prior_c_mu,
            prior_c_sigma,
        )
    ):
        raise ValueError(
            "PSIS-ranked sampling requires saved posterior samples and stage-one priors "
            "for every candidate galaxy."
        )

    M200 = np.asarray(data["M200"], dtype=float)
    c = np.asarray(data["c"], dtype=float)
    plate_ifu = np.asarray(data["plate_ifu"], dtype=str)
    obs_mask = np.isfinite(M200) & np.isfinite(c) & (M200 > 0) & (c > 0)
    sample_mask = _build_valid_sample_mask(
        sample_m200,
        sample_c,
        prior_mu,
        prior_sigma,
        prior_lower,
        prior_upper,
        prior_c_mu,
        prior_c_sigma,
    )
    valid_mask = obs_mask & sample_mask

    if not np.any(valid_mask):
        raise ValueError("No valid posterior-sample entries remain for PSIS-ranked sampling.")
    if int(np.sum(valid_mask)) < 3:
        raise ValueError("PSIS-ranked sampling requires at least 3 valid galaxies.")

    sliced = _slice_pipeline_inputs(
        valid_mask,
        M200=M200,
        c=c,
        sample_m200=sample_m200,
        sample_c=sample_c,
        sample_m200_prior_mu=prior_mu,
        sample_m200_prior_sigma=prior_sigma,
        sample_m200_prior_lower=prior_lower,
        sample_m200_prior_upper=prior_upper,
        sample_c_prior_mu=prior_c_mu,
        sample_c_prior_sigma=prior_c_sigma,
    )
    sliced["plate_ifu"] = plate_ifu[valid_mask]

    print(
        f"[gen-sample] Mode: psis-khat (candidate pool with valid posterior samples: {len(sliced['plate_ifu'])})"
    )
    fit_results = fit_m200_c_mcmc(
        sliced["M200"],
        sliced["c"],
        log10_M200_posterior_samples=sliced["sample_m200"],
        log10_c_posterior_samples=sliced["sample_c"],
        log10_M200_prior_mu=sliced["sample_m200_prior_mu"],
        log10_M200_prior_sigma=sliced["sample_m200_prior_sigma"],
        log10_M200_prior_lower=sliced["sample_m200_prior_lower"],
        log10_M200_prior_upper=sliced["sample_m200_prior_upper"],
        log10_c_prior_mu=sliced["sample_c_prior_mu"],
        log10_c_prior_sigma=sliced["sample_c_prior_sigma"],
        sample_cap=sample_cap,
        dataset_label="gen_sample_psis",
        verbose=False,
    )
    diagnostics = compute_psis_importance_diagnostics(
        log10_M200_posterior_samples=sliced["sample_m200"],
        log10_c_posterior_samples=sliced["sample_c"],
        log10_M200_prior_mu=sliced["sample_m200_prior_mu"],
        log10_M200_prior_sigma=sliced["sample_m200_prior_sigma"],
        log10_M200_prior_lower=sliced["sample_m200_prior_lower"],
        log10_M200_prior_upper=sliced["sample_m200_prior_upper"],
        log10_c_prior_mu=sliced["sample_c_prior_mu"],
        log10_c_prior_sigma=sliced["sample_c_prior_sigma"],
        fit_results=fit_results,
        plot_suffix="_gen_sample_psis",
        sample_cap=sample_cap,
        save_plots=False,
    )
    if diagnostics is None:
        raise ValueError("Failed to compute PSIS diagnostics for PSIS-ranked sampling.")

    k_hat = np.asarray(diagnostics.get("k_hat"), dtype=float)
    finite_mask = np.isfinite(k_hat)
    if not np.any(finite_mask):
        raise ValueError("No finite PSIS k-hat values were computed for the candidate pool.")

    finite_k_hat = k_hat[finite_mask]
    order = np.argsort(finite_k_hat, kind="stable")
    ranked_ifus = sliced["plate_ifu"][finite_mask][order]
    ranked_k_hat = finite_k_hat[order]
    selected_ifus = ranked_ifus[: min(n_sample, len(ranked_ifus))].tolist()

    print(
        f"[gen-sample] PSIS k-hat range among candidates: "
        f"[{ranked_k_hat[0]:.4f}, {ranked_k_hat[-1]:.4f}]"
    )
    print(
        f"[gen-sample] Target: {n_sample}, selected: {len(selected_ifus)}, "
        f"largest selected k-hat: {ranked_k_hat[min(len(selected_ifus), len(ranked_k_hat)) - 1]:.4f}"
    )
    return selected_ifus


def _hist_step_density(ax: plt.Axes, values: np.ndarray, bins: np.ndarray, label: str, color: str, linestyle: str, linewidth: float) -> None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return

    ax.hist(
        values,
        bins=bins,
        density=True,
        histtype="step",
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        label=label,
    )

    median = float(np.nanmedian(values))
    ymin, ymax = ax.get_ylim()
    tick_top = ymin + 0.10 * (ymax - ymin if ymax > ymin else 1.0)
    ax.vlines(median, ymin, tick_top, colors=color, linestyles=linestyle, linewidth=linewidth * 0.8, alpha=0.85)


def _plot_attrition_metric_panel(
    ax: plt.Axes,
    stages: list[tuple[str, pd.DataFrame, str, str, float]],
    column_name: str,
    xlabel: str,
    *,
    show_legend: bool,
    legend_loc: str = "upper right",
) -> None:
    finite_values = []
    for _, frame, _, _, _ in stages:
        values = np.asarray(frame[column_name], dtype=float)
        values = values[np.isfinite(values)]
        if values.size > 0:
            finite_values.append(values)

    if finite_values:
        merged = np.concatenate(finite_values)
        x_min = float(np.nanmin(merged))
        x_max = float(np.nanmax(merged))
        if np.isclose(x_min, x_max):
            x_min -= 0.5
            x_max += 0.5
        bins = np.linspace(x_min, x_max, 26)
    else:
        bins = np.linspace(0.0, 1.0, 26)

    for label, frame, color, linestyle, linewidth in stages:
        # legend_label = f"{label} count={len(frame)}"
        legend_label = f"{label}"
        _hist_step_density(
            ax,
            frame[column_name].to_numpy(dtype=float),
            bins,
            legend_label,
            color,
            linestyle,
            linewidth,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_facecolor("#FCFCFC")
    ax.grid(True, alpha=0.18, linewidth=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=12)

    if show_legend:
        legend = ax.legend(frameon=False, fontsize=10.5, loc=legend_loc, handlelength=3.0)
        if legend is not None:
            legend._legend_box.align = "left"
            for text in legend.get_texts():
                text.set_ha("left")
                text.set_linespacing(1.1)


def _save_attrition_metric_panels(
    output_path: Path,
    stages: list[tuple[str, pd.DataFrame, str, str, float]],
) -> list[Path]:
    metric_specs = [
        ("redshift", "Redshift z", "redshift"),
        ("log10_mstar", r"$\log_{10}(M_\star / M_\odot)$", "mstar"),
        ("inclination", "Inclination [deg]", "inclination"),
        ("sersic_n", "Sersic n", "sersic_n"),
    ]
    saved_paths: list[Path] = []

    for column_name, xlabel, suffix in metric_specs:
        fig, ax = plt.subplots(figsize=(5.6, 4.2), facecolor="white")
        _plot_attrition_metric_panel(
            ax,
            stages,
            column_name,
            xlabel,
            show_legend=True,
            legend_loc="upper left",
        )
        panel_path = output_path.with_name(f"{output_path.stem}_{suffix}{output_path.suffix}")
        fig.savefig(panel_path, dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(panel_path.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(panel_path)

    return saved_paths


def plot_sample_attrition_pipeline(
    sample_specs: tuple[tuple[str | Path, str], tuple[str | Path, str]],
    output_path: Path | None = None,
) -> Path:
    sample_file_a, label_a = sample_specs[0]
    sample_file_b, label_b = sample_specs[1]
    sample_path_a, sample_catalog_a = _load_sample_catalog_from_ifu_file(sample_file_a)
    sample_path_b, sample_catalog_b = _load_sample_catalog_from_ifu_file(sample_file_b)

    if output_path is None:
        output_path = result_dir / "galaxy_select_compare.png"

    print("[attrition] Comparing two IFU-list samples from --plot-attrition.")
    print(f"[attrition] 样本 A 标签: {label_a}")
    print(f"[attrition] 样本 A 文件: {sample_path_a.resolve()}")
    print(f"[attrition] 样本 A 星系数: {len(sample_catalog_a)}")
    print(f"[attrition] 样本 B 标签: {label_b}")
    print(f"[attrition] 样本 B 文件: {sample_path_b.resolve()}")
    print(f"[attrition] 样本 B 星系数: {len(sample_catalog_b)}")

    fig = plt.figure(figsize=(12.0, 7.6), facecolor="white")
    grid = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.20)
    ax_redshift = fig.add_subplot(grid[0, 0])
    ax_mstar = fig.add_subplot(grid[0, 1])
    ax_inclination = fig.add_subplot(grid[1, 0])
    ax_sersic = fig.add_subplot(grid[1, 1])

    stages = [
        (label_a, sample_catalog_a, "#D55E00", "-", 1.5),
        (label_b, sample_catalog_b, "#0072B2", "-", 1.5),
    ]
    metric_specs = [
        (ax_redshift, "redshift", "Redshift z"),
        (ax_mstar, "log10_mstar", r"$\log_{10}(M_\star / M_\odot)$"),
        (ax_inclination, "inclination", "Inclination [deg]"),
        (ax_sersic, "sersic_n", "Sersic n"),
    ]

    for idx, (ax, column_name, xlabel) in enumerate(metric_specs):
        _plot_attrition_metric_panel(
            ax,
            stages,
            column_name,
            xlabel,
            show_legend=(idx == 0),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_path.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)
    saved_panel_paths = _save_attrition_metric_panels(output_path, stages)
    print(f"Sample attrition metric panels saved to {[str(path) for path in saved_panel_paths]}")
    return output_path


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
    return fit_m200_c_mcmc(
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
    )


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

    try:
        prior_draw_count = max(len(log10_c0_samples), len(alpha_samples), 1000)
        prior_idata, posterior_idata = _build_prior_posterior_density_data(
            posterior_samples={
                "log10_c0": np.asarray(log10_c0_samples, dtype=float),
                "alpha": np.asarray(alpha_samples, dtype=float),
            },
            prior_draw_count=prior_draw_count,
        )

        def _save_single_density_plot(var_name: str, samples: np.ndarray, prior_idata: object, posterior_idata: object, title_prefix: str, color: str, *, save_combined: bool = True) -> None:
            if save_combined:
                fig, axes = plt.subplots(2, 1, figsize=(6.2, 7.0))
                az_api.plot_density(
                    [prior_idata, posterior_idata],
                    data_labels=["Prior", "Posterior"],
                    var_names=[var_name],
                    ax=np.atleast_1d(axes[0]),
                    point_estimate=None,
                    hdi_prob=HDI_PROB2,
                    shade=0.15,
                    colors=["#9A9A9A", color],
                    outline=True,
                    textsize=8,
                )
                axes[0].set_title(f"{title_prefix} Prior vs Posterior")
                axes[0].set_xlabel(title_prefix)
                plot_posterior_1d_hdi(
                    samples,
                    title=f"{title_prefix} Posterior KDE",
                    base_color=color,
                    ax=axes[1],
                    hdi_probs=(HDI_PROB1, HDI_PROB2),
                    show_interval_bars=False,
                )
                fig.tight_layout()
                out = result_dir / f"c-M_relation_posterior_{dataset_tag}_{var_name}.png"
                fig.savefig(out, dpi=300, bbox_inches="tight")
                fig.savefig(out.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
                plt.close(fig)

            density_fig, density_ax = plt.subplots(1, 1, figsize=(6.2, 3.2))
            az_api.plot_density(
                [prior_idata, posterior_idata],
                data_labels=["Prior", "Posterior"],
                var_names=[var_name],
                ax=np.atleast_1d(density_ax),
                point_estimate=None,
                hdi_prob=HDI_PROB2,
                shade=0.15,
                colors=["#9A9A9A", color],
                outline=True,
                textsize=8,
            )
            density_ax.set_title(f"{title_prefix} Prior vs Posterior")
            density_ax.set_xlabel(title_prefix)
            density_fig.tight_layout()
            density_out = result_dir / f"c-M_relation_posterior_{dataset_tag}_{var_name}_density.png"
            density_fig.savefig(density_out, dpi=300, bbox_inches="tight")
            density_fig.savefig(density_out.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
            plt.close(density_fig)

            kde_fig, kde_ax = plt.subplots(1, 1, figsize=(6.2, 3.2))
            plot_posterior_1d_hdi(
                samples,
                title=f"{title_prefix} Posterior KDE",
                base_color=color,
                ax=kde_ax,
                hdi_probs=(HDI_PROB1, HDI_PROB2),
                show_interval_bars=False,
            )
            kde_fig.tight_layout()
            kde_out = result_dir / f"c-M_relation_posterior_{dataset_tag}_{var_name}_kde.png"
            kde_fig.savefig(kde_out, dpi=300, bbox_inches="tight")
            kde_fig.savefig(kde_out.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
            plt.close(kde_fig)

        skip_combined_plot_vars = {"log10_c0", "alpha"} if dataset_tag == "all" else set()
        _save_single_density_plot(
            "log10_c0",
            log10_c0_samples,
            prior_idata,
            posterior_idata,
            r"$\log_{10} c_0$",
            COLOR_LOW_N,
            save_combined="log10_c0" not in skip_combined_plot_vars,
        )
        _save_single_density_plot(
            "alpha",
            alpha_samples,
            prior_idata,
            posterior_idata,
            r"$\alpha$",
            COLOR_HIGH_N,
            save_combined="alpha" not in skip_combined_plot_vars,
        )

        print("Split prior/posterior and KDE plots saved for log10_c0 and alpha")
    except Exception as exc:
        print(f"Warning: Split prior/posterior and KDE plots failed: {exc}")

    try:
        pair_var_names_all = ["log10_c0", "alpha", "M200_mu", "M200_sigma", "sigma_int"]
        pair_axes_all = az_api.plot_pair(
            trace,
            var_names=pair_var_names_all,
            kind=["kde"],
            marginals=True,
            marginal_kwargs={
                "kind": "hist",
                "hist_kwargs": {
                    "bins": 30,
                    "histtype": "step",
                    "linewidth": 1.5,
                    "density": True,
                },
            },
            kde_kwargs={"hdi_probs": [HDI_PROB1, HDI_PROB2]},
            point_estimate=None,
            textsize=8,
            divergences=False,
        )
        _annotate_pair_marginals_m200(
            pair_axes_all,
            posterior,
            pair_var_names_all,
            title_fontsize=9,
            plot_median_line=True,
        )
        for var_name in pair_var_names_all:
            pair_axes_single = az_api.plot_pair(
                trace,
                var_names=[var_name],
                kind=["kde"],
                marginals=True,
                marginal_kwargs={"kind": "hist", "hist_kwargs": {"bins": 30, "histtype": "step", "linewidth": 1.5, "density": True}},
                kde_kwargs={"hdi_probs": [HDI_PROB1, HDI_PROB2]},
                point_estimate=None,
                textsize=8,
                divergences=False,
            )
            pair_axes_single_array = np.asarray(pair_axes_single, dtype=object)
            fig = pair_axes_single_array.flat[0].figure
            fig.set_size_inches(4.2, 4.2)
            fig.savefig(result_dir / f"c-M_relation_pair_{dataset_label}_{var_name}.png", dpi=300, bbox_inches="tight")
            fig.savefig((result_dir / f"c-M_relation_pair_{dataset_label}_{var_name}.png").with_suffix(".pdf"), format="pdf", bbox_inches="tight")
            plt.close(fig)
        pair_axes_all_array = np.asarray(pair_axes_all, dtype=object)
        for ax in pair_axes_all_array.flat:
            if ax is not None:
                ax.set_xticks([])
                ax.set_yticks([])

        pair_all_fig = pair_axes_all_array.flat[0].figure
        pair_all_fig.set_size_inches(12, 10)
        # pair_all_fig.suptitle("Halo Concentration-Mass (c-M) Relation Posterior Pair", fontsize=12, y=0.98)
        pair_all_path = result_dir / f"c-M_relation_pair_{dataset_tag}.png"
        pair_all_fig.savefig(pair_all_path, dpi=300, bbox_inches="tight")
        pair_all_fig.savefig(
            pair_all_path.with_suffix(".pdf"),
            format="pdf",
            bbox_inches="tight",
        )
        plt.close(pair_all_fig)
        print(f"All-parameter pair plot saved to {pair_all_path}")
    except Exception as exc:
        print(f"Warning: All-parameter pair plot failed: {exc}")

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
