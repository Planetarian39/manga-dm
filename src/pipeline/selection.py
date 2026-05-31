"""Sample selection and quality-filtering logic.

Consolidated from ``src-orig/plates.py`` and ``src-orig/m200.py``.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config.constants import PLATES_FILENAME, QUALITY_FILTER_PRESETS
from src.config.settings import settings
from src.data.catalog import (
    load_all_sample_catalog,
    load_screened_sample_catalog,
)
from src.data.results import (
    get_m200_c_data,
    load_m200_c_result_table,
)


def select_and_download(
    inc_min: float | None = None,
    inc_max: float | None = None,
    ifu_file: str | None = None,
    download: bool = False,
) -> list[str]:
    """Select galaxy sample by inclination and optionally download data.

    Parameters
    ----------
    inc_min, inc_max : float or None
        Inclination range in degrees (defaults from settings).
    ifu_file : str or None
        Output file for plate-IFU list.
    download : bool
        Whether to also trigger data download.

    Returns
    -------
    list[str]
        Selected plate-IFU strings.
    """
    if inc_min is None:
        inc_min = settings.INC_MIN
    if inc_max is None:
        inc_max = settings.INC_MAX

    from src.data.catalog import DrpallUtil
    from src.data.fits import FitsUtil

    fits_util = FitsUtil(settings.data_dir)
    drpall_file = fits_util.get_drpall_file()
    print(f"DRPALL file: {drpall_file}")

    drpall_util = DrpallUtil(drpall_file)
    plateifus, _ = drpall_util.search_plateifu_by_inc(inc_min, inc_max)
    selected = sorted(str(plateifu) for plateifu in plateifus)

    print(f"-- Galaxies with inclination between {inc_min} and {inc_max} degrees:")
    print(f"  Total found: {len(selected)}")
    print("== Filter selection of galaxies:")
    print(f"  Total selected galaxies: {len(selected)}")

    output_file = (
        settings.resolve_input_path(ifu_file)
        if ifu_file is not None
        else settings.data_dir / PLATES_FILENAME
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as fh:
        for plateifu in selected:
            fh.write(f"{plateifu}\n")
    print(f"  Selected plateifus saved to: {output_file}")

    if download:
        _download_selected_fits(fits_util, selected)

    return selected


def _download_selected_fits(fits_util, plateifu_list: list[str]) -> None:
    total = len(plateifu_list)
    if total == 0:
        print("No plateifu to download.")
        return

    max_workers = min(8, total)

    def _process(plateifu: str):
        errors = []
        try:
            fits_util.get_maps_file(plateifu, checksum=True)
        except Exception as exc:
            errors.append(f"maps:{exc}")

        try:
            fits_util.get_image_file(plateifu)
        except Exception as exc:
            errors.append(f"image:{exc}")

        return plateifu, errors

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process, plateifu): plateifu for plateifu in plateifu_list}
        for future in tqdm(
            as_completed(futures),
            total=total,
            desc="Downloading maps",
            unit="galaxy",
        ):
            try:
                plateifu, errors = future.result()
                if errors:
                    tqdm.write(f"Errors for {plateifu}: {', '.join(errors)}")
            except Exception as exc:
                plateifu = futures.get(future, "unknown")
                tqdm.write(f"Unhandled error for {plateifu}: {exc}")


def generate_robustness_sample(
    n: int = 10,
    result_dir_override: str | Path | None = None,
) -> None:
    """Generate *n* robustness sub-samples from the posterior pool."""
    print(f"Generating {n} robustness sub-samples...")
    generate_robustness_subsample(
        n_sample=n,
        result_dir_override=result_dir_override,
    )
    print("Done.")


def _build_success_mask(df: pd.DataFrame) -> np.ndarray:
    if "result" in df.columns:
        return df["result"].astype(str).str.lower().eq("success").to_numpy()
    if "success" in df.columns:
        return df["success"].astype(bool).to_numpy()
    return np.ones(len(df), dtype=bool)


def _filter_dataframe_by_success(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[_build_success_mask(df)].copy()


def _filter_dataframe_by_ifu_ids(
    df: pd.DataFrame,
    ifu_ids: list[str] | None,
) -> pd.DataFrame:
    if ifu_ids is None:
        return df
    requested = [str(ifu) for ifu in ifu_ids]
    return df.loc[df.index.astype(str).isin(requested)].copy()


def _filter_dataframe_by_nrmse(
    df: pd.DataFrame,
    nrmse_threshold: float | None,
) -> pd.DataFrame:
    if nrmse_threshold is None:
        return df
    column = "nrmse" if "nrmse" in df.columns else "NRMSE" if "NRMSE" in df.columns else None
    if column is None:
        return df
    values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    return df.loc[np.isfinite(values) & (values <= float(nrmse_threshold))].copy()


def _resolve_quality_filter_thresholds(
    quality_cut: str | None = None,
    *,
    max_redchi: float | None = None,
    ppc_p_min: float | None = None,
    ppc_p_max: float | None = None,
    ppc_value_coverage_min: float | None = None,
    ppc_overlap_min: float | None = None,
    max_abs_c_m200_corr: float | None = None,
) -> dict[str, float | None]:
    thresholds: dict[str, float | None] = {}
    if quality_cut:
        try:
            thresholds.update(QUALITY_FILTER_PRESETS[quality_cut])
        except KeyError as exc:
            valid = ", ".join(sorted(QUALITY_FILTER_PRESETS))
            raise ValueError(f"Unknown quality_cut '{quality_cut}'. Valid presets: {valid}") from exc

    overrides = {
        "max_redchi": max_redchi,
        "ppc_p_min": ppc_p_min,
        "ppc_p_max": ppc_p_max,
        "ppc_value_coverage_min": ppc_value_coverage_min,
        "ppc_overlap_min": ppc_overlap_min,
        "max_abs_c_m200_corr": max_abs_c_m200_corr,
    }
    for key, value in overrides.items():
        if value is not None:
            thresholds[key] = float(value)
    return thresholds


def _first_existing_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    lower_to_actual = {str(name).lower(): str(name) for name in df.columns}
    for candidate in candidates:
        match = lower_to_actual.get(candidate.lower())
        if match is not None:
            return match
    return None


def _numeric_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> np.ndarray | None:
    column = _first_existing_column(df, candidates)
    if column is None:
        return None
    return pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)


def _filter_dataframe_by_quality_metrics(
    df: pd.DataFrame,
    *,
    quality_cut: str | None = None,
    max_redchi: float | None = None,
    ppc_p_min: float | None = None,
    ppc_p_max: float | None = None,
    ppc_value_coverage_min: float | None = None,
    ppc_overlap_min: float | None = None,
    max_abs_c_m200_corr: float | None = None,
) -> pd.DataFrame:
    thresholds = _resolve_quality_filter_thresholds(
        quality_cut,
        max_redchi=max_redchi,
        ppc_p_min=ppc_p_min,
        ppc_p_max=ppc_p_max,
        ppc_value_coverage_min=ppc_value_coverage_min,
        ppc_overlap_min=ppc_overlap_min,
        max_abs_c_m200_corr=max_abs_c_m200_corr,
    )
    mask = np.ones(len(df), dtype=bool)

    checks = [
        ("max_redchi", ("redchi", "CHI_SQ_V"), lambda values, threshold: values <= threshold),
        ("ppc_p_min", ("dev_ppc_p",), lambda values, threshold: values >= threshold),
        ("ppc_p_max", ("dev_ppc_p",), lambda values, threshold: values <= threshold),
        (
            "ppc_value_coverage_min",
            ("PPC_ETI_VALUE_COVERAGE", "PPC_HDI_VALUE_COVERAGE"),
            lambda values, threshold: values >= threshold,
        ),
        (
            "ppc_overlap_min",
            ("PPC_ETI_OVERLAP", "PPC_HDI_OVERLAP"),
            lambda values, threshold: values >= threshold,
        ),
        (
            "max_abs_c_m200_corr",
            ("c_M200_corr",),
            lambda values, threshold: np.abs(values) <= threshold,
        ),
    ]
    for threshold_key, columns, predicate in checks:
        threshold = thresholds.get(threshold_key)
        if threshold is None:
            continue
        values = _numeric_column(df, columns)
        if values is None:
            continue
        mask &= np.isfinite(values) & predicate(values, float(threshold))

    return df.loc[mask].copy()


def _attach_result_catalog_columns(
    df: pd.DataFrame,
    result_dir_override: str | Path | None = None,
) -> pd.DataFrame:
    if {"log10_mstar", "sersic_n"}.issubset(df.columns):
        return df
    try:
        catalog = load_screened_sample_catalog(result_dir_override)
    except Exception:
        return df

    result = df.copy()
    for column in ("redshift", "log10_mstar", "inclination", "sersic_n"):
        if column not in result.columns and column in catalog.columns:
            result[column] = catalog.reindex(result.index)[column].to_numpy()
    return result


def _filter_dataframe_by_tertile_group(
    df: pd.DataFrame,
    *,
    column: str,
    tertile_label: str | None,
    option_name: str,
    display_name: str,
) -> pd.DataFrame:
    if tertile_label is None or str(tertile_label).lower() in {"all", "none"}:
        return df
    if column not in df.columns:
        raise ValueError(f"{option_name} requires {display_name} values in the sample table.")

    values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        raise ValueError(f"{option_name} cannot be applied because {display_name} is empty.")

    low_edge, high_edge = np.nanquantile(values[finite], [1.0 / 3.0, 2.0 / 3.0])
    label = str(tertile_label).lower()
    if label in {"low", "lower"}:
        mask = finite & (values <= low_edge)
    elif label in {"mid", "middle"}:
        mask = finite & (values > low_edge) & (values <= high_edge)
    elif label in {"high", "upper"}:
        mask = finite & (values > high_edge)
    else:
        raise ValueError(f"{option_name} must be one of low, mid, high, or all.")
    return df.loc[mask].copy()


def prepare_m200_c_result_table(
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
) -> pd.DataFrame | None:
    try:
        df = load_m200_c_result_table(result_dir_override)
    except FileNotFoundError as exc:
        print(f"Warning: {exc}")
        return None

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

    df = _attach_result_catalog_columns(df, result_dir_override)
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
    return df


def prepare_m200_c_data(
    result_dir_override: str | Path | None = None,
    **filter_kwargs,
) -> dict | None:
    df = prepare_m200_c_result_table(
        result_dir_override=result_dir_override,
        **filter_kwargs,
    )
    if df is None:
        return None
    return get_m200_c_data(result_dir=result_dir_override, dataframe=df)


def _load_stage_result_table(
    result_dir_override: str | Path | None = None,
    ifu_ids: list[str] | None = None,
    success_only: bool = True,
) -> pd.DataFrame:
    df = load_m200_c_result_table(result_dir_override)
    if success_only:
        df = _filter_dataframe_by_success(df)
    return _filter_dataframe_by_ifu_ids(df, ifu_ids)


def _slice_pipeline_inputs(valid_mask: np.ndarray, **arrays) -> dict:
    return {
        key: np.asarray(value, dtype=object if key.startswith("sample_") else None)[valid_mask]
        for key, value in arrays.items()
    }


def generate_robustness_subsample(
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
    """Draw a representative robustness subsample from the candidate galaxies."""
    valid_modes = (
        "random",
        "mass-quintile",
        "mass-parent",
        "sersic-quintile",
        "sersic-parent",
        "psis-khat",
    )
    if mode not in valid_modes:
        raise ValueError(f"--gen-sample-mode must be one of {valid_modes}, got '{mode}'")

    if ifu_ids is None:
        result_table = _load_stage_result_table(result_dir_override=result_dir_override)
        ifu_ids = result_table.index.astype(str).tolist()

    if mode == "psis-khat":
        selected_ifus = _select_lowest_psis_khat_ifus(
            n_sample=n_sample,
            nrmse_threshold=nrmse_threshold,
            result_dir_override=result_dir_override,
            ifu_ids=ifu_ids,
            sample_cap=sample_cap,
        )
        return _write_selected_ifus(selected_ifus, output_filename, mode)

    rng = np.random.default_rng(random_seed)
    result_table = _load_stage_result_table(result_dir_override=result_dir_override)
    success_mask = _build_success_mask(result_table)
    if nrmse_threshold is not None:
        nrmse_values = _numeric_column(result_table, ("nrmse", "NRMSE"))
        if nrmse_values is not None:
            success_mask &= np.isfinite(nrmse_values) & (
                nrmse_values <= float(nrmse_threshold)
            )

    result_ifus = set(result_table.index[success_mask].tolist())
    pool_ifus = [ifu for ifu in ifu_ids if ifu in result_ifus]
    if len(pool_ifus) == 0:
        raise ValueError(
            "No galaxies from the candidate IFU pool are available with successful stage-one results. "
            "Check --ifu-file and --result-dir."
        )

    screened_catalog = load_screened_sample_catalog(result_dir_override)
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
    print(f"[gen-sample] Pool size (candidate IFUs filtered by successful results): {len(pool_ifus_arr)}")

    deficit = 0
    if mode == "random":
        draw_count = min(n_sample, len(pool_ifus_arr))
        selected_ifus = rng.choice(pool_ifus_arr, size=draw_count, replace=False).tolist()
    elif mode.endswith("quintile"):
        selected_ifus, deficit = _draw_feature_quintile_sample(
            rng,
            pool_ifus_arr,
            pool_feature,
            n_sample,
        )
    else:
        selected_ifus, deficit = _draw_parent_matched_sample(
            rng,
            pool_ifus_arr,
            pool_feature,
            n_sample,
            n_bins,
            feature_col,
            feature_label,
        )

    if deficit > 0:
        already = set(selected_ifus)
        spare = [ifu for ifu in pool_ifus_arr.tolist() if ifu not in already]
        if spare:
            extra = rng.choice(spare, size=min(deficit, len(spare)), replace=False)
            selected_ifus.extend(extra.tolist())

    _print_sample_feature_summary(screened_catalog, selected_ifus, feature_col, feature_label, n_sample)
    return _write_selected_ifus(selected_ifus, output_filename, mode)


def _draw_feature_quintile_sample(
    rng: np.random.Generator,
    pool_ifus_arr: np.ndarray,
    pool_feature: np.ndarray,
    n_sample: int,
) -> tuple[list[str], int]:
    n_q = 5
    finite_feature = pool_feature[np.isfinite(pool_feature)]
    if finite_feature.size == 0:
        raise ValueError("No finite feature values are available for quintile sampling.")
    q_edges = np.quantile(finite_feature, np.linspace(0, 1, n_q + 1))
    q_edges[0] -= 1e-9
    q_edges[-1] += 1e-9
    target_per_bin = np.full(n_q, n_sample // n_q, dtype=int)
    for idx in range(n_sample % n_q):
        target_per_bin[idx] += 1

    selected_ifus: list[str] = []
    deficit = 0
    for bin_idx in range(n_q):
        lo, hi = q_edges[bin_idx], q_edges[bin_idx + 1]
        in_bin = np.where((pool_feature > lo) & (pool_feature <= hi))[0]
        need = int(target_per_bin[bin_idx] + deficit)
        if in_bin.size == 0:
            deficit = need
            continue
        actual = min(need, in_bin.size)
        deficit = need - actual
        chosen = rng.choice(in_bin, size=actual, replace=False)
        selected_ifus.extend(pool_ifus_arr[chosen].tolist())
    return selected_ifus, deficit


def _draw_parent_matched_sample(
    rng: np.random.Generator,
    pool_ifus_arr: np.ndarray,
    pool_feature: np.ndarray,
    n_sample: int,
    n_bins: int,
    feature_col: str,
    feature_label: str,
) -> tuple[list[str], int]:
    all_catalog = load_all_sample_catalog()
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
    for idx in range(int(remainder)):
        target_per_bin[order[idx]] += 1

    selected_ifus: list[str] = []
    deficit = 0
    for bin_idx in range(n_bins):
        lo, hi = bin_edges[bin_idx], bin_edges[bin_idx + 1]
        in_bin = np.where((pool_feature >= lo) & (pool_feature < hi))[0]
        if bin_idx == n_bins - 1:
            in_bin = np.where((pool_feature >= lo) & (pool_feature <= hi))[0]
        need = int(target_per_bin[bin_idx] + deficit)
        if in_bin.size == 0:
            deficit = need
            continue
        actual = min(need, in_bin.size)
        deficit = need - actual
        chosen = rng.choice(in_bin, size=actual, replace=False)
        selected_ifus.extend(pool_ifus_arr[chosen].tolist())

    print(f"[gen-sample] Parent {feature_label} range: [{feat_min:.3f}, {feat_max:.3f}]")
    return selected_ifus, deficit


def _print_sample_feature_summary(
    screened_catalog: pd.DataFrame,
    selected_ifus: list[str],
    feature_col: str,
    feature_label: str,
    n_sample: int,
) -> None:
    n_selected = len(selected_ifus)
    sel_feat = screened_catalog.reindex(selected_ifus)[feature_col].to_numpy(dtype=float)
    sel_feat_finite = sel_feat[np.isfinite(sel_feat)]
    print(f"[gen-sample] Target: {n_sample}, drawn: {n_selected}")
    if sel_feat_finite.size > 0:
        print(
            f"[gen-sample] Subsample {feature_label} range: "
            f"[{sel_feat_finite.min():.3f}, {sel_feat_finite.max():.3f}], "
            f"median: {np.median(sel_feat_finite):.3f}"
        )


def _write_selected_ifus(
    selected_ifus: list[str],
    output_filename: str | None,
    mode: str,
) -> Path:
    if output_filename is None:
        output_filename = f"sample_{mode}_{len(selected_ifus)}.txt"
    output_path = settings.data_dir / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
    from src.models.population import _build_valid_sample_mask, fit_m200_c_mcmc
    from src.stats.psis import compute_psis_importance_diagnostics

    data = prepare_m200_c_data(
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
