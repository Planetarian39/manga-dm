"""Unified ArviZ compatibility layer.

Consolidated from:
- ``src-orig/util/arviz_compat.py`` (public API),
- ``src-orig/dm.py`` (local adapters),
- ``src-orig/rc.py`` (nested adapters),
- ``src-orig/m200.py`` (``_get_az``, ``_require_pymc_stack``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Lazy ArviZ import
_az = None
_pm = None
_pt = None


def _import_arviz():
    """Lazily import ArviZ and return the module reference."""
    global _az
    if _az is None:
        try:
            import arviz as __az
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "This command requires arviz. Lightweight commands that do "
                "not use posterior summaries or ArviZ plots should avoid "
                "calling arviz_compat helpers."
            ) from exc
        _az = __az
    return _az


def ensure_arviz_compat():
    """Ensure ``arviz.InferenceData`` is accessible and return the arviz module."""
    az_api = _import_arviz()
    if hasattr(az_api, "InferenceData"):
        return az_api

    inference_data_cls = None
    try:
        from arviz.data import InferenceData as inference_data_cls
    except Exception:
        try:
            from arviz.data.inference_data import (
                InferenceData as inference_data_cls,
            )
        except Exception:
            inference_data_cls = None

    if inference_data_cls is not None:
        az_api.InferenceData = inference_data_cls

    return az_api


def get_arviz_api():
    """Return the ArviZ module (or ``arviz.preview``) with a ``summary`` function."""
    az_api = _import_arviz()
    if hasattr(az_api, "summary"):
        return az_api
    if hasattr(az_api, "preview") and hasattr(az_api.preview, "summary"):
        return az_api.preview
    return az_api


def set_arviz_ci_defaults(ci_prob: float = 0.68, ci_kind: str = "eti") -> None:
    """Set ArviZ rcParams for credible interval defaults.

    Parameters
    ----------
    ci_prob : float
        Default credible-interval probability.
    ci_kind : str
        ``"eti"`` (equal-tailed) or ``"hdi"`` (highest-density).
    """
    try:
        az_mod = get_arviz_api()
        if "stats.ci_prob" in az_mod.rcParams:
            az_mod.rcParams["stats.ci_prob"] = ci_prob
        if "stats.ci_kind" in az_mod.rcParams:
            az_mod.rcParams["stats.ci_kind"] = ci_kind
    except Exception:
        pass


# ── Lazy ArviZ accessor (m200.py style) ──────────────────────────────

def get_az():
    """Lazy ArviZ accessor with compat fix applied."""
    global _az
    if _az is None:
        _az = ensure_arviz_compat()
    return _az


# ── PyMC / PyTensor stack ────────────────────────────────────────────

def require_pymc_stack():
    """Lazy-import pymc and pytensor; raise a helpful error if missing."""
    global _pm, _pt
    if _pm is None:
        try:
            import pymc as __pm
            import pytensor.tensor as __pt
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "This command requires the Bayesian fitting stack "
                "(pymc, pytensor). Commands like --plot-attrition should "
                "run without them, but fitting commands need them installed "
                "in the active Python environment."
            ) from exc
        _pm = __pm
        _pt = __pt
    return _pm, _pt


# ── InferenceData helpers ────────────────────────────────────────────

def get_posterior_dataset(idata):
    """Return the posterior xarray Dataset from an ArviZ InferenceData object."""
    if hasattr(idata, "posterior"):
        return idata.posterior
    try:
        posterior = idata["posterior"]
        if hasattr(posterior, "dataset"):
            posterior = posterior.dataset
        return posterior
    except Exception as exc:
        raise AttributeError(
            "posterior group not found on inference data"
        ) from exc


def get_prior_dataset(idata):
    """Return the prior xarray Dataset from an ArviZ InferenceData object."""
    if hasattr(idata, "prior"):
        return idata.prior
    try:
        prior = idata["prior"]
        if hasattr(prior, "dataset"):
            prior = prior.dataset
        return prior
    except Exception as exc:
        raise AttributeError(
            "prior group not found on inference data"
        ) from exc


# ── Summary helpers ──────────────────────────────────────────────────

def _summary_row_name(var_name: str, index: tuple[int, ...]) -> str:
    if not index:
        return var_name
    return f"{var_name}[{','.join(str(i) for i in index)}]"


def _posterior_medians(
    idata, var_names: list[str]
) -> dict[str, float]:
    """Compute per-parameter posterior medians from InferenceData."""
    posterior = get_posterior_dataset(idata)
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
            medians[_summary_row_name(var_name, index)] = float(
                median_values[index]
            )
    return medians


def get_summary_interval_columns(
    summary: pd.DataFrame,
) -> tuple[str, str]:
    """Return (lower_col, upper_col) for HDI/ETI interval columns in *summary*.

    Detects ``hdi_*``/``eti_*`` columns first, then falls back to
    ``*_lb``/``*_ub`` naming.
    """
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

    raise KeyError(
        "HDI/ETI interval columns not found in ArviZ summary output"
    )


def summary_with_compat(
    summary_fn,
    idata,
    *,
    var_names: list[str],
    round_to: int,
    stat_focus: str | None = None,
) -> pd.DataFrame:
    """Call ``summary_fn`` with backwards-compat for ``stat_focus``.

    Falls back to manually computing medians if the ArviZ version doesn't
    support ``stat_focus``.
    """
    try:
        if stat_focus is not None:
            return summary_fn(
                idata, var_names=var_names, round_to=round_to,
                stat_focus=stat_focus,
            )
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
