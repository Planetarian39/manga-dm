from __future__ import annotations

import numpy as np
import pandas as pd

az = None


def _import_arviz():
    global az
    if az is None:
        try:
            import arviz as _az
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "This command requires arviz. Lightweight commands that do not use posterior summaries "
                "or ArviZ plots should avoid calling util.arviz_compat helpers."
            ) from exc
        az = _az
    return az


def ensure_arviz_compat():
    az_api = _import_arviz()
    if hasattr(az_api, "InferenceData"):
        return az_api

    inference_data_cls = None
    try:
        from arviz.data import InferenceData as inference_data_cls
    except Exception:
        try:
            from arviz.data.inference_data import InferenceData as inference_data_cls
        except Exception:
            inference_data_cls = None

    if inference_data_cls is not None:
        az_api.InferenceData = inference_data_cls

    return az_api


def get_arviz_api():
    az_api = _import_arviz()
    if hasattr(az_api, "summary"):
        return az_api
    if hasattr(az_api, "preview") and hasattr(az_api.preview, "summary"):
        return az_api.preview
    return az_api


def get_summary_interval_columns(summary: pd.DataFrame) -> tuple[str, str]:
    columns = [str(col) for col in summary.columns]

    standard_columns = [
        col for col in columns if col.startswith(("hdi_", "eti_"))
    ]
    if len(standard_columns) >= 2:
        def _extract_percent(column_name: str) -> float:
            label = column_name
            for prefix in ("hdi_", "eti_"):
                if label.startswith(prefix):
                    label = label.replace(prefix, "", 1)
                    break
            label = label.replace("%", "")
            return float(label)

        standard_columns.sort(key=_extract_percent)
        return standard_columns[0], standard_columns[-1]

    lower_candidates = [col for col in columns if col.endswith("_lb")]
    upper_candidates = [col for col in columns if col.endswith("_ub")]
    if lower_candidates and upper_candidates:
        def _base_name(column_name: str) -> str:
            return column_name[:-3]

        lower_map = {_base_name(col): col for col in lower_candidates}
        upper_map = {_base_name(col): col for col in upper_candidates}
        common_bases = sorted(set(lower_map) & set(upper_map))
        if common_bases:
            base = common_bases[0]
            return lower_map[base], upper_map[base]

    raise KeyError("HDI/ETI interval columns not found in ArviZ summary output")


def _get_posterior_dataset(idata):
    if hasattr(idata, "posterior"):
        return idata.posterior
    posterior = idata["posterior"]
    if hasattr(posterior, "dataset"):
        posterior = posterior.dataset
    return posterior


def _summary_row_name(var_name: str, index: tuple[int, ...]) -> str:
    if not index:
        return var_name
    return f"{var_name}[{','.join(str(i) for i in index)}]"


def _posterior_medians(idata, var_names: list[str]) -> dict[str, float]:
    posterior = _get_posterior_dataset(idata)
    medians: dict[str, float] = {}

    for var_name in var_names:
        if var_name not in posterior:
            continue

        values = np.asarray(posterior[var_name].values, dtype=float)
        if values.ndim < 2:
            continue

        reshaped = values.reshape((-1,) + values.shape[2:])
        median_values = np.nanmedian(reshaped, axis=0)

        if np.ndim(median_values) == 0:
            medians[var_name] = float(median_values)
            continue

        for index in np.ndindex(median_values.shape):
            medians[_summary_row_name(var_name, index)] = float(median_values[index])

    return medians


def summary_with_compat(summary_fn, idata, *, var_names: list[str], round_to: int, stat_focus: str | None = None) -> pd.DataFrame:
    try:
        if stat_focus is not None:
            return summary_fn(idata, var_names=var_names, round_to=round_to, stat_focus=stat_focus)
        return summary_fn(idata, var_names=var_names, round_to=round_to)
    except TypeError as exc:
        if "stat_focus" not in str(exc):
            raise

    summary = summary_fn(idata, var_names=var_names, round_to=round_to)
    if "median" in summary.columns:
        return summary

    medians = _posterior_medians(idata, var_names)
    if not medians:
        return summary

    summary = summary.copy()
    summary["median"] = np.nan
    for row_name, value in medians.items():
        if row_name in summary.index:
            summary.loc[row_name, "median"] = round(float(value), round_to)
    return summary