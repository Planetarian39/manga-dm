import os
import tomllib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
import pandas as pd
import xarray as xr
from scipy.stats import norm, t as t_dist, truncnorm

from util.arviz_compat import ensure_arviz_compat, get_summary_interval_columns, summary_with_compat
from util.drpall_util import DrpallUtil
from util.fits_util import FitsUtil
from util.plot_util import plot_posterior_1d_hdi

az = None


# Constants
M_PIVOT_H_INV = 1e12  # in Msun/h, pivot mass for c-M200 relation
H_0 = 0.674  # Hubble parameter

# log10 c = a + b log10(M/[10^12 h^-1 M⊙]),
# Dutton & Maccio 2014 c-M200 normalization at pivot mass (h=0.674)
LOG10_C0_DM14 = 0.905  # c200
ALPHA_DM14 = -0.101
LOG10_C_SIGMA_DM14 = 0.11

# Li+20 (SPARC) c-M200 normalization at pivot mass (h=0.674)
LOG10_C0_LI20 = 0.84
LOG10_C0_SIGMA_LI20 = 0.03
ALPHA_LI20 = -0.06
ALPHA_SIGMA_LI20 = 0.04
LOG10_C_SCATTER_LI20 = 0.20

# Yasin+23 (HI) c-M200 normalization at pivot mass (h=0.674)
LOG10_C0_YASIN23 = 0.91
LOG10_C0_SIGMA_YASIN23 = 0.05
ALPHA_YASIN23 = -0.11
ALPHA_SIGMA_YASIN23 = 0.03
LOG10_C_SCATTER_YASIN23 = 0.15


# Plotting colors (colorblind-safe palette + neutral grays)
COLOR_DATA_POINTS = "#4D4D4D"
COLOR_SIGMA_BAND = "#D9D9D9"
COLOR_HDI_BAND = "#BDBDBD"
COLOR_HIGH_N = "#D55E00"
COLOR_LOW_N = "#0072B2"

COLOR_POSTERIOR_MEDIAN = "#0072B2"
COLOR_DM14 = "#4D4D4D"      # Dutton & Maccio 2014
COLOR_LI20 = "#009E73"      # Li et al. 2020
COLOR_YASIN23 = "#D55E00"   # Yasin et al. 2023

LOG10_C0_PRIOR_MEAN = LOG10_C0_DM14
LOG10_C0_PRIOR_SIGMA = 0.5
ALPHA_PRIOR_MEAN = ALPHA_DM14
ALPHA_PRIOR_SIGMA = 0.3
LOG_SIGMA_INT_PRIOR_MEAN = np.log(0.15)
LOG_SIGMA_INT_PRIOR_SIGMA = 0.8

# Student-t degrees of freedom for population model  (Gamma prior: mean≈20, allows ν≈2–5 for heavy tails)
NU_POP_PRIOR_ALPHA = 2.0
NU_POP_PRIOR_BETA = 0.1

# Defensive IS mixing weight ε: proposal q = (1-ε)·p_pop + ε·p_stage1
# Guarantees w ≤ 1/ε, bounding weight variance regardless of p_pop/p_stage1 mismatch.
DEFENSIVE_IS_EPSILON = 0.1

# Recommended quality cuts for stage-two population inference.
# These act only on single-galaxy fit diagnostics and identifiability proxies,
# avoiding direct cuts on M200 or c that would bias the inferred c-M relation.
QUALITY_FILTER_PRESETS = {
    "recommended": {
        "max_redchi": 3.0,
        "ppc_p_min": 0.05,
        "ppc_p_max": 0.95,
        "ppc_overlap_min": 0.5,
        "max_abs_c_m200_corr": 0.95,
    },
    "strict": {
        "max_redchi": 2.0,
        "ppc_p_min": 0.10,
        "ppc_p_max": 0.90,
        "ppc_value_coverage_min": 0.80,
        "ppc_overlap_min": 0.60,
        "max_abs_c_m200_corr": 0.90,
    },
}

def load_config() -> dict:
    """Load configuration from config.toml."""
    config_path = Path("config.toml")
    if not config_path.exists():
        raise FileNotFoundError("Error: config.toml file not found")

    with open(config_path, "rb") as f:
        config = tomllib.load(f)
        if not config:
            raise ValueError("Error: config.toml file is empty")
    return config


# Initialize paths from config
config = load_config()
data_directory = config.get("file", {}).get("data_directory", "data")
result_directory = config.get("file", {}).get("result_directory", "results")
root_dir = Path(__file__).resolve().parent.parent
data_dir = root_dir / data_directory
result_dir = data_dir / result_directory

NFW_PARAM_CM200_FILENAME = config.get("file", {}).get(
    "nfw_param_cm200_filename", "nfw_param_cm200.csv"
)
NFW_PARAM_CM200_SAMPLE_FILENAME = config.get("file", {}).get(
    "nfw_param_cm200_sample_filename", "nfw_param_cm200_samples.nc"
)
HDI_PROB1 = config.get("thresholds", {}).get("HDI_PROB1", 0.68)
HDI_PROB2 = config.get("thresholds", {}).get("HDI_PROB2", 0.95)


def _get_az():
    global az
    if az is None:
        az = ensure_arviz_compat()
    return az


def _require_pymc_stack():
    try:
        import pytensor.tensor as pt
        import pymc as pm
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "This command requires the Bayesian fitting stack (pymc, pytensor). "
            "Commands like --plot-attrition should run without them, but fitting commands need them installed in the active Python environment."
        ) from exc

    return pm, pt


def _infer_plate_ifu_from_sample_file(
    sample_path: Path, sample_file_name: str
) -> str | None:
    suffix = f"_{sample_file_name}"
    if sample_path.name.endswith(suffix):
        return sample_path.name[: -len(suffix)]
    return None


def _is_plate_ifu_like(value: str) -> bool:
    parts = str(value).split("-", 1)
    return len(parts) == 2 and all(part.isdigit() for part in parts)


def _collect_per_ifu_sample_files(sample_file: Path) -> list[Path]:
    per_ifu_files: list[Path] = []
    per_ifu_pattern = f"*_{sample_file.name}"
    for candidate in sorted(sample_file.parent.glob(per_ifu_pattern)):
        plate_ifu = _infer_plate_ifu_from_sample_file(candidate, sample_file.name)
        if plate_ifu and _is_plate_ifu_like(plate_ifu):
            per_ifu_files.append(candidate)
    return per_ifu_files


def _resolve_result_dir(result_dir_override: str | Path | None = None) -> Path:
    if result_dir_override is None:
        return result_dir

    resolved = Path(result_dir_override)
    if not resolved.is_absolute():
        resolved = root_dir / resolved
    return resolved


def _set_result_dir(result_dir_override: str | Path | None = None) -> Path:
    global result_dir

    result_dir = _resolve_result_dir(result_dir_override)
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir


def _load_ifu_id_list(ifu_file: str | Path | None = None) -> list[str] | None:
    if ifu_file is None:
        return None

    path = _resolve_input_path(ifu_file)

    if not path.exists():
        raise FileNotFoundError(f"IFU list file not found: {path}")

    with open(path, "r", encoding="utf-8") as handle:
        ifu_ids = [line.strip() for line in handle if line.strip()]

    if not ifu_ids:
        raise ValueError(f"No IFU IDs found in file: {path}")
    return ifu_ids


def _resolve_input_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if not path.is_absolute():
        candidate_data = data_dir / path
        if candidate_data.exists():
            path = candidate_data
        else:
            path = root_dir / path

    return path


def _load_ifu_id_file_with_path(ifu_file: str | Path) -> tuple[Path, list[str]]:
    path = _resolve_input_path(ifu_file)

    if not path.exists():
        raise FileNotFoundError(f"IFU list file not found: {path}")

    with open(path, "r", encoding="utf-8") as handle:
        ifu_ids = [line.strip() for line in handle if line.strip()]

    if not ifu_ids:
        raise ValueError(f"No IFU IDs found in file: {path}")
    return path, ifu_ids


def _build_attrition_sample_specs(
    plot_attrition_args: list[str] | tuple[str, str, str, str],
) -> tuple[tuple[str, str], tuple[str, str]]:
    if len(plot_attrition_args) != 4:
        raise ValueError(
            "--plot-attrition expects exactly four arguments: FILE_A LABEL_A FILE_B LABEL_B"
        )

    sample_file_a, label_a, sample_file_b, label_b = [str(value).strip() for value in plot_attrition_args]
    if not sample_file_a or not label_a or not sample_file_b or not label_b:
        raise ValueError(
            "--plot-attrition requires non-empty values for FILE_A LABEL_A FILE_B LABEL_B."
        )

    return (sample_file_a, label_a), (sample_file_b, label_b)


def _load_sample_catalog_from_ifu_file(ifu_file: str | Path) -> tuple[Path, pd.DataFrame]:
    sample_path, ifu_ids = _load_ifu_id_file_with_path(ifu_file)
    catalog = _build_base_sample_catalog()
    sample_catalog = catalog.reindex(pd.Index(ifu_ids, dtype=str, name="plate_ifu")).copy()
    return sample_path, sample_catalog


def _filter_dataframe_by_ifu_ids(df: pd.DataFrame, ifu_ids: list[str] | None = None) -> pd.DataFrame:
    if not ifu_ids:
        return df

    requested = pd.Index([str(ifu) for ifu in ifu_ids], dtype=str)
    filtered = df.reindex(df.index.intersection(requested))
    filtered = filtered.dropna(how="all") if filtered.shape[1] > 0 else filtered
    return filtered


def _build_success_mask(df: pd.DataFrame) -> np.ndarray:
    if "result" not in df.columns:
        return np.ones(len(df), dtype=bool)
    return df["result"].astype(str).str.lower().eq("success").to_numpy(dtype=bool)


def _filter_dataframe_by_success(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df.loc[_build_success_mask(df)].copy()


def _filter_dataframe_by_nrmse(
    df: pd.DataFrame,
    nrmse_threshold: float | None = None,
) -> pd.DataFrame:
    if df.empty or nrmse_threshold is None:
        return df

    if "nrmse" not in df.columns:
        print("Warning: NRMSE threshold requested but 'nrmse' column is not available.")
        return df

    nrmse_values = pd.to_numeric(df["nrmse"], errors="coerce")
    mask = np.isfinite(nrmse_values) & (nrmse_values <= float(nrmse_threshold))
    return df.loc[mask].copy()


def _filter_dataframe_by_numeric_metric(
    df: pd.DataFrame,
    column_name: str,
    option_name: str,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
    abs_max: float | None = None,
) -> pd.DataFrame:
    if df.empty:
        return df

    if min_value is None and max_value is None and abs_max is None:
        return df

    if column_name not in df.columns:
        print(
            f"Warning: {option_name} requested but column {column_name!r} is not available."
        )
        return df

    values = pd.to_numeric(df[column_name], errors="coerce")
    finite_mask = np.isfinite(values)
    finite_count = int(np.sum(finite_mask))
    total = len(df)
    if finite_count == 0:
        print(
            f"Warning: {option_name} requested but column {column_name!r} has no finite values."
        )
        return df

    if finite_count < total:
        print(
            f"Warning: {option_name} requested but column {column_name!r} is only available for "
            f"{finite_count} / {total} galaxies; skipping this filter to remain backward-compatible."
        )
        return df

    mask = finite_mask.copy()
    if min_value is not None:
        mask &= values >= float(min_value)
    if max_value is not None:
        mask &= values <= float(max_value)
    if abs_max is not None:
        mask &= np.abs(values) <= float(abs_max)

    kept = int(np.sum(mask))
    descriptor_parts = []
    if min_value is not None:
        descriptor_parts.append(f">={float(min_value):.3f}")
    if max_value is not None:
        descriptor_parts.append(f"<={float(max_value):.3f}")
    if abs_max is not None:
        descriptor_parts.append(f"|x|<={float(abs_max):.3f}")
    descriptor = ", ".join(descriptor_parts)
    print(
        f"Applying {option_name} on {column_name} ({descriptor}): kept {kept} / {total} galaxies."
    )
    return df.loc[mask].copy()


def _resolve_quality_filter_thresholds(
    quality_cut: str | None = None,
    max_redchi: float | None = None,
    ppc_p_min: float | None = None,
    ppc_p_max: float | None = None,
    ppc_value_coverage_min: float | None = None,
    ppc_overlap_min: float | None = None,
    max_abs_c_m200_corr: float | None = None,
) -> dict[str, float]:
    thresholds = dict(QUALITY_FILTER_PRESETS.get(quality_cut, {}))

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


def _filter_dataframe_by_quality_metrics(
    df: pd.DataFrame,
    quality_cut: str | None = None,
    max_redchi: float | None = None,
    ppc_p_min: float | None = None,
    ppc_p_max: float | None = None,
    ppc_value_coverage_min: float | None = None,
    ppc_overlap_min: float | None = None,
    max_abs_c_m200_corr: float | None = None,
) -> pd.DataFrame:
    thresholds = _resolve_quality_filter_thresholds(
        quality_cut=quality_cut,
        max_redchi=max_redchi,
        ppc_p_min=ppc_p_min,
        ppc_p_max=ppc_p_max,
        ppc_value_coverage_min=ppc_value_coverage_min,
        ppc_overlap_min=ppc_overlap_min,
        max_abs_c_m200_corr=max_abs_c_m200_corr,
    )
    if not thresholds:
        return df

    if quality_cut is not None:
        print(f"Applying quality preset --quality-cut={quality_cut}.")

    filtered = df.copy()
    filtered = _filter_dataframe_by_numeric_metric(
        filtered,
        "redchi",
        "--max-redchi",
        max_value=thresholds.get("max_redchi"),
    )
    filtered = _filter_dataframe_by_numeric_metric(
        filtered,
        "dev_ppc_p",
        "--ppc-p-min/--ppc-p-max",
        min_value=thresholds.get("ppc_p_min"),
        max_value=thresholds.get("ppc_p_max"),
    )
    filtered = _filter_dataframe_by_numeric_metric(
        filtered,
        "PPC_ETI_VALUE_COVERAGE",
        "--ppc-coverage-min",
        min_value=thresholds.get("ppc_value_coverage_min"),
    )
    filtered = _filter_dataframe_by_numeric_metric(
        filtered,
        "PPC_ETI_OVERLAP",
        "--ppc-overlap-min",
        min_value=thresholds.get("ppc_overlap_min"),
    )
    filtered = _filter_dataframe_by_numeric_metric(
        filtered,
        "c_M200_corr",
        "--max-abs-cm-corr",
        abs_max=thresholds.get("max_abs_c_m200_corr"),
    )
    return filtered


def _normalize_tertile_label(value: str | None, option_name: str) -> str | None:
    if value is None:
        return None

    normalized = str(value).strip().lower()
    alias_map = {
        "mid": "median",
        "middle": "median",
    }
    normalized = alias_map.get(normalized, normalized)
    allowed = {"low", "median", "high"}
    if normalized not in allowed:
        raise ValueError(
            f"Invalid value for {option_name}: {value!r}. Expected one of: low, median, high."
        )
    return normalized


def _attach_result_catalog_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    catalog = _build_base_sample_catalog().reindex(df.index)
    enriched = df.copy()

    for column in ["log10_mstar", "sersic_n"]:
        catalog_series = pd.to_numeric(catalog[column], errors="coerce")
        if column in enriched.columns:
            result_series = pd.to_numeric(enriched[column], errors="coerce")
            enriched.loc[:, column] = result_series.where(result_series.notna(), catalog_series)
        else:
            enriched.loc[:, column] = catalog_series

    return enriched


def _filter_dataframe_by_tertile_group(
    df: pd.DataFrame,
    column: str,
    tertile_label: str | None,
    option_name: str,
    display_name: str,
) -> pd.DataFrame:
    normalized_label = _normalize_tertile_label(tertile_label, option_name)
    if normalized_label is None:
        return df

    if column not in df.columns:
        raise ValueError(f"Column {column!r} is not available for {option_name} filtering.")

    values = pd.to_numeric(df[column], errors="coerce")
    valid = values[np.isfinite(values)]
    if len(valid) < 3:
        raise ValueError(
            f"Not enough valid {display_name} values to apply {option_name}."
        )

    ordered = valid.sort_values(kind="mergesort")
    tertile_groups = np.array_split(ordered.index.to_numpy(dtype=str), 3)
    label_to_index = {"low": 0, "median": 1, "high": 2}
    selected_ifus = tertile_groups[label_to_index[normalized_label]]

    if len(selected_ifus) == 0:
        raise ValueError(
            f"No galaxies matched {option_name}={normalized_label} in the current result set."
        )

    selected_values = ordered.loc[selected_ifus]
    print(
        f"Applying {option_name}={normalized_label}: selected {len(selected_ifus)} / {len(valid)} galaxies "
        f"from the {display_name} tertile range "
        f"[{float(np.nanmin(selected_values)):.4f}, {float(np.nanmax(selected_values)):.4f}]"
    )
    return df.reindex(pd.Index(selected_ifus, dtype=str)).dropna(how="all")


def merge_posterior_samples_file(
    filename: str = NFW_PARAM_CM200_SAMPLE_FILENAME,
    result_dir_override: str | Path | None = None,
) -> Path | None:
    output_file = _resolve_result_dir(result_dir_override) / filename
    per_ifu_files = _collect_per_ifu_sample_files(output_file)

    if not per_ifu_files:
        print(
            f"No per-IFU posterior sample files found in {output_file.parent} "
            f"matching '*_{output_file.name}'."
        )
        return None

    merged_rows: list[tuple[str, np.ndarray, np.ndarray]] = []
    for sample_path in per_ifu_files:
        loaded = _load_single_posterior_sample_file(sample_path, output_file.name)
        if loaded is None:
            continue

        plate_ifu, log10_m200, log10_c = loaded
        if not _is_plate_ifu_like(plate_ifu):
            continue

        merged_rows.append((plate_ifu, log10_m200, log10_c))

    if not merged_rows:
        print("No valid per-IFU posterior sample files could be merged.")
        return None

    merged_rows.sort(key=lambda row: row[0])
    max_sample_count = max(len(row[1]) for row in merged_rows)
    plate_ifus = [row[0] for row in merged_rows]
    sample_counts = np.array([len(row[1]) for row in merged_rows], dtype=np.int32)

    log10_m200_values = np.full(
        (len(merged_rows), max_sample_count), np.nan, dtype=np.float64
    )
    log10_c_values = np.full(
        (len(merged_rows), max_sample_count), np.nan, dtype=np.float64
    )

    for idx, (_, log10_m200, log10_c) in enumerate(merged_rows):
        sample_count = len(log10_m200)
        log10_m200_values[idx, :sample_count] = log10_m200
        log10_c_values[idx, :sample_count] = log10_c

    dataset = xr.Dataset(
        data_vars={
            "log10_M200_samples": (
                ("plate_ifu", "sample"),
                log10_m200_values,
            ),
            "log10_c_samples": (
                ("plate_ifu", "sample"),
                log10_c_values,
            ),
            "sample_count": (("plate_ifu",), sample_counts),
        },
        coords={
            "plate_ifu": np.asarray(plate_ifus, dtype=str),
            "sample": np.arange(max_sample_count, dtype=np.int32),
        },
        attrs={
            "description": "Merged posterior log10-samples for NFW M200 and c across PLATE_IFU files",
            "storage_format": "merged_per_ifu_netcdf",
            "source_file_count": int(len(merged_rows)),
        },
    )

    temp_output_file = output_file.with_name(
        f"{output_file.stem}.tmp{output_file.suffix}"
    )
    dataset.to_netcdf(temp_output_file)
    dataset.close()
    os.replace(temp_output_file, output_file)

    print(
        f"Merged {len(merged_rows)} per-IFU posterior sample files into {output_file}."
    )
    return output_file


def _extract_log10_sample_arrays(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    if "log10_M200_samples" in ds and "log10_c_samples" in ds:
        log10_m200 = np.asarray(ds["log10_M200_samples"].values, dtype=float).reshape(-1)
        log10_c = np.asarray(ds["log10_c_samples"].values, dtype=float).reshape(-1)
        return log10_m200, log10_c

    if "M200_samples" in ds and "c_samples" in ds:
        m200_array = np.asarray(ds["M200_samples"].values, dtype=float).reshape(-1)
        c_array = np.asarray(ds["c_samples"].values, dtype=float).reshape(-1)
        return np.log10(m200_array), np.log10(c_array)

    raise KeyError(
        "Posterior sample variables not found. Expected either "
        "('log10_M200_samples', 'log10_c_samples') or ('M200_samples', 'c_samples')."
    )


def _load_single_posterior_sample_file(
    sample_path: Path, sample_file_name: str
) -> tuple[str | None, np.ndarray, np.ndarray] | None:
    try:
        ds = xr.load_dataset(sample_path)
        plate_ifu = str(
            ds.attrs.get("plate_ifu")
            or _infer_plate_ifu_from_sample_file(sample_path, sample_file_name)
            or ""
        )
        if not plate_ifu:
            ds.close()
            return None

        log10_m200, log10_c = _extract_log10_sample_arrays(ds)
        ds.close()

        if len(log10_m200) == 0 or len(log10_c) == 0:
            return None

        return plate_ifu, log10_m200, log10_c
    except Exception as e:
        print(f"Warning: Could not parse posterior sample file {sample_path}: {e}")
        return None


def _load_posterior_sample_map(
    sample_file: Path, plate_ifus: list[str] | None = None
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if not sample_file.exists():
        print(f"Warning: Merged posterior sample file {sample_file} not found.")
        return {}

    if sample_file.suffix.lower() != ".nc":
        print(
            f"Warning: Unsupported merged posterior sample file format: {sample_file}"
        )
        return {}

    try:
        ds = xr.load_dataset(sample_file)
        sample_plate_ifus = ds.coords["plate_ifu"].astype(str).values.tolist()

        if "log10_M200_samples" in ds and "log10_c_samples" in ds:
            log10_m200_values = np.asarray(ds["log10_M200_samples"].values, dtype=float)
            log10_c_values = np.asarray(ds["log10_c_samples"].values, dtype=float)
        elif "M200_samples" in ds and "c_samples" in ds:
            log10_m200_values = np.log10(
                np.asarray(ds["M200_samples"].values, dtype=float)
            )
            log10_c_values = np.log10(np.asarray(ds["c_samples"].values, dtype=float))
        else:
            ds.close()
            raise KeyError("Posterior sample variables not found in merged sample file.")

        if log10_m200_values.ndim == 1:
            log10_m200_values = log10_m200_values[None, :]
        if log10_c_values.ndim == 1:
            log10_c_values = log10_c_values[None, :]

        if "sample_count" in ds:
            sample_counts = np.asarray(ds["sample_count"].values, dtype=int)
        else:
            finite_mask = np.isfinite(log10_m200_values) & np.isfinite(log10_c_values)
            sample_counts = np.sum(finite_mask, axis=1, dtype=int)

        requested_plate_ifus = None
        if plate_ifus is not None:
            requested_plate_ifus = {str(plate_ifu) for plate_ifu in plate_ifus}

        sample_map: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for idx, plate_ifu in enumerate(sample_plate_ifus):
            plate_ifu = str(plate_ifu)
            if requested_plate_ifus is not None and plate_ifu not in requested_plate_ifus:
                continue

            sample_count = int(sample_counts[idx])
            if sample_count <= 0:
                continue

            m200_array = log10_m200_values[idx, :sample_count].astype(float, copy=False)
            c_array = log10_c_values[idx, :sample_count].astype(float, copy=False)
            sample_map[plate_ifu] = (m200_array, c_array)

        ds.close()
        return sample_map
    except Exception as e:
        print(f"Warning: Could not parse merged posterior sample file {sample_file}: {e}")

    return {}


def get_summary_eti_columns(summary: pd.DataFrame) -> tuple[str, str]:
    return get_summary_interval_columns(summary)


def _format_interval_value(value: float, decimals: int = 2) -> str:
    value = float(value)
    if not np.isfinite(value):
        return "n/a"
    if value == 0:
        return f"{0:.{decimals}f}"
    if abs(value) >= 1000:
        return f"{value:.1e}"
    return f"{value:.{decimals}f}"


def _format_interval_supsub(
    median: float,
    eti_low: float | None,
    eti_high: float | None,
    decimals: int = 2,
) -> str:
    if eti_low is None or eti_high is None:
        return _format_interval_value(median, decimals=decimals)

    median = float(median)
    eti_low = float(eti_low)
    eti_high = float(eti_high)
    if not (np.isfinite(median) and np.isfinite(eti_low) and np.isfinite(eti_high)):
        return _format_interval_value(median, decimals=decimals)

    lower_err = median - eti_low
    upper_err = eti_high - median
    if lower_err < 0 or upper_err < 0:
        return _format_interval_value(median, decimals=decimals)

    median_text = _format_interval_value(median, decimals=decimals)
    lower_text = _format_interval_value(lower_err, decimals=decimals)
    upper_text = _format_interval_value(upper_err, decimals=decimals)
    return rf"{median_text}_{{-{lower_text}}}^{{+{upper_text}}}"


def _get_pair_plot_unit_label(var_name: str) -> str:
    unit_map = {
        "M200_mu": r"\log_{10} M_\odot",
        "M200_sigma": r"dex",
        "log10_c0": r"dex",
        "alpha": "",
        "sigma_int": r"dex",
    }
    return unit_map.get(var_name, "")


def _annotate_pair_marginals_m200(
    pair_axes,
    posterior,
    plotted_var_names: list[str],
    title_fontsize: float = 9,
    plot_median_line: bool = True,
) -> None:
    axes = np.asarray(pair_axes, dtype=object)
    if axes.ndim != 2:
        return

    diagonal_count = min(len(plotted_var_names), axes.shape[0], axes.shape[1])
    for idx in range(diagonal_count):
        ax = axes[idx, idx]
        if ax is None:
            continue

        var_name = plotted_var_names[idx]
        if var_name not in posterior:
            continue

        samples = np.asarray(posterior[var_name].values, dtype=float).reshape(-1)
        samples = samples[np.isfinite(samples)]
        if samples.size < 2:
            continue

        eti_low, median, eti_high = np.percentile(samples, [2.5, 50.0, 97.5])

        if plot_median_line:
            title_line_color = ax.title.get_color()
            if title_line_color in (None, "auto"):
                title_line_color = "0.35"

            ax.axvline(median, color=title_line_color, linestyle="--", linewidth=1.0, alpha=0.85, zorder=3)
            for bound in (eti_low, eti_high):
                ax.axvline(bound, color=title_line_color, linestyle="--", linewidth=1.0, alpha=0.85, zorder=3)

        unit_label = _get_pair_plot_unit_label(var_name)
        title_text = _format_interval_supsub(median, eti_low, eti_high, decimals=3)
        if unit_label:
            title_text = rf"{title_text}\,{unit_label}"
        ax.set_title(rf"${title_text}$", fontsize=title_fontsize, pad=4)


def _build_prior_posterior_density_data(
    posterior_samples: dict[str, np.ndarray],
    prior_draw_count: int,
    random_seed: int = 42,
) -> tuple[object, object]:
    az_api = _get_az()
    rng = np.random.default_rng(random_seed)
    prior_samples = {
        "log10_c0": rng.normal(
            loc=LOG10_C0_PRIOR_MEAN,
            scale=LOG10_C0_PRIOR_SIGMA,
            size=prior_draw_count,
        ),
        "alpha": rng.normal(
            loc=ALPHA_PRIOR_MEAN,
            scale=ALPHA_PRIOR_SIGMA,
            size=prior_draw_count,
        ),
    }

    prior_idata = az_api.from_dict(
        posterior={name: values[None, :] for name, values in prior_samples.items()}
    )
    posterior_idata = az_api.from_dict(
        posterior={name: values[None, :] for name, values in posterior_samples.items()}
    )
    return prior_idata, posterior_idata


# concentration–mass (c–M) relation
# c = 10^log10_c0 * (M200 / M_pivot)^alpha
# log10_c = log10_c0 + alpha * (log10_M200 - log10_M_pivot)
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


def _draw_posterior_cloud(
    ax: plt.Axes,
    M200_summary: np.ndarray,
    c_summary: np.ndarray,
    sample_m200,
    sample_c,
    valid_mask: np.ndarray,
    sample_plot: int = 20,
    log_axes: bool = False,
) -> tuple[bool, int | None]:
    """Draw stacked posterior samples (up to ``sample_plot`` per galaxy).

    Each galaxy contributes at most ``sample_plot`` randomly-drawn posterior
    sample points, plotted as a transparent scatter cloud.
    Falls back to summary-median scatter when no sample arrays are available.
    Returns whether the cloud was drawn and the per-galaxy sample cap used.
    """
    pooled_log10_M200: list[np.ndarray] = []
    pooled_log10_c: list[np.ndarray] = []

    if sample_m200 is not None and sample_c is not None:
        rng = np.random.default_rng(0)
        samples_m200_masked = np.asarray(sample_m200, dtype=object)[valid_mask]
        samples_c_masked = np.asarray(sample_c, dtype=object)[valid_mask]

        for gal_m200, gal_c in zip(samples_m200_masked, samples_c_masked):
            if gal_m200 is None or gal_c is None:
                continue
            arr_m = np.asarray(gal_m200, dtype=float)
            arr_c = np.asarray(gal_c, dtype=float)
            finite = np.isfinite(arr_m) & np.isfinite(arr_c)
            if not np.any(finite):
                continue
            arr_m, arr_c = arr_m[finite], arr_c[finite]
            n = len(arr_m)
            if n > sample_plot:
                idx = rng.choice(n, size=sample_plot, replace=False)
                arr_m, arr_c = arr_m[idx], arr_c[idx]
            pooled_log10_M200.append(arr_m)
            pooled_log10_c.append(arr_c)

    if pooled_log10_M200:
        _pool_m = np.concatenate(pooled_log10_M200)
        _pool_c = np.concatenate(pooled_log10_c)
        all_M200 = _pool_m if log_axes else 10.0 ** _pool_m
        all_c = _pool_c if log_axes else 10.0 ** _pool_c

        ax.scatter(
            all_M200,
            all_c,
            color=COLOR_DATA_POINTS,
            alpha=0.15,
            s=6,
            edgecolors="none",
            rasterized=True,
            zorder=1,
        )
        return True, int(sample_plot)

    # Fallback: summary-median scatter
    _M200_fb = np.log10(M200_summary) if log_axes else M200_summary
    _c_fb = np.log10(c_summary) if log_axes else c_summary
    ax.scatter(
        _M200_fb,
        _c_fb,
        color=COLOR_DATA_POINTS,
        alpha=0.7,
        label="Data Points",
        s=20,
        edgecolors="none",
        zorder=2,
    )
    return False, None


def plot_m200_c_relation_all(
    M200: np.ndarray,
    c: np.ndarray,
    fit_results: dict = None,
    n_boot: int = 50,
    plot_suffix: str = "",
    sample_m200=None,
    sample_c=None,
    sample_plot: int = 20,
):
    """
    Spaghetti plot of c-M200 relation for all data (no Sersic n split).

    If sample_m200 and sample_c are provided (object arrays of per-galaxy
    log10 posterior samples), the individual-galaxy posterior distributions are
    stacked and rendered as a 2-D density cloud (hexbin) instead of gray
    scatter points of summary medians.
    """
    valid_mask = (M200 > 0) & (c > 0) & np.isfinite(M200) & np.isfinite(c)
    if not np.any(valid_mask):
        return

    M200 = M200[valid_mask]
    c = c[valid_mask]

    m_plot = np.logspace(np.log10(np.min(M200)), np.log10(np.max(M200)), 50)
    log10_m_plot = np.log10(m_plot)
    log10_M_pivot = np.log10(M_PIVOT_H_INV / H_0)
    ref_log10_c0 = LOG10_C0_DM14
    ref_alpha = ALPHA_DM14
    ref_log10_sigma = LOG10_C_SIGMA_DM14

    fig, ax_top = plt.subplots(1, 1, figsize=(10, 6))

    # --- Stacked posterior cloud ---
    # Collect all per-galaxy posterior samples when available; fall back to
    # summary-median scatter points when they are not.
    _cloud_drawn, _cloud_sample_count = _draw_posterior_cloud(
        ax_top,
        M200,
        c,
        sample_m200,
        sample_c,
        valid_mask,
        sample_plot=sample_plot,
        log_axes=True,
    )

    c_reference, c_reference_low, c_reference_high = reference_log10_c_band(
        m_plot,
        ref_log10_c0,
        ref_alpha,
        ref_log10_sigma,
        h=H_0,
    )
    ax_top.plot(
        log10_m_plot,
        np.log10(c_reference),
        color=COLOR_DM14,
        linewidth=2,
        linestyle="--",
        label="Dutton & Maccio 2014",
    )
    ax_top.fill_between(
        log10_m_plot,
        np.log10(c_reference_low),
        np.log10(c_reference_high),
        color=COLOR_DM14,
        alpha=0.18,
        label=rf"Dutton & Macciò (2014) $\pm 2\sigma$",
    )

    # Li et al. 2020 (SPARC) reference line
    c_li20, c_li20_low, c_li20_high = reference_log10_c_band(
        m_plot,
        LOG10_C0_LI20,
        ALPHA_LI20,
        LOG10_C_SCATTER_LI20,
        log10_c0_sigma=LOG10_C0_SIGMA_LI20,
        alpha_sigma=ALPHA_SIGMA_LI20,
        h=H_0,
    )
    ax_top.plot(
        log10_m_plot,
        np.log10(c_li20),
        color=COLOR_LI20,
        linewidth=2,
        linestyle="--",
        label="Li et al. 2020 (SPARC)",
    )
    # ax_top.fill_between(
    #     log10_m_plot,
    #     np.log10(c_li20_low),
    #     np.log10(c_li20_high),
    #     color=COLOR_LI20,
    #     alpha=0.18,
    #     label=rf"Li et al. (2020) 2$\sigma$ band",
    # )

    # Yasin et al. 2023 (HI) reference line
    c_yasin23, c_yasin23_low, c_yasin23_high = reference_log10_c_band(
        m_plot,
        LOG10_C0_YASIN23,
        ALPHA_YASIN23,
        LOG10_C_SCATTER_YASIN23,
        log10_c0_sigma=LOG10_C0_SIGMA_YASIN23,
        alpha_sigma=ALPHA_SIGMA_YASIN23,
        h=H_0,
    )
    ax_top.plot(
        log10_m_plot,
        np.log10(c_yasin23),
        color=COLOR_YASIN23,
        linewidth=2,
        linestyle="--",
        label="Yasin et al. 2023 (HI)",
    )
    # ax_top.fill_between(
    #     log10_m_plot,
    #     np.log10(c_yasin23_low),
    #     np.log10(c_yasin23_high),
    #     color=COLOR_YASIN23,
    #     alpha=0.18,
    #     label=rf"Yasin et al. (2023) 2$\sigma$ band",
    # )

    log10_c0_fit = fit_results.get("log10_c0_median") if fit_results else None
    alpha_fit = fit_results.get("alpha_median") if fit_results else None
    log10_c0_fit_eti_low = fit_results.get("log10_c0_eti_low") if fit_results else None
    log10_c0_fit_eti_high = fit_results.get("log10_c0_eti_high") if fit_results else None
    alpha_fit_eti_low = fit_results.get("alpha_eti_low") if fit_results else None
    alpha_fit_eti_high = fit_results.get("alpha_eti_high") if fit_results else None
    sigma_int = fit_results.get("sigma_int_median") if fit_results else None
    sigma_int_eti_low = fit_results.get("sigma_int_eti_low") if fit_results else None
    sigma_int_eti_high = fit_results.get("sigma_int_eti_high") if fit_results else None

    if log10_c0_fit is None or alpha_fit is None:
        log10_c0_fit, alpha_fit = _compute_linear_fit(M200, c, log10_M_pivot)

    if log10_c0_fit is not None and alpha_fit is not None:
        c_median = log10_c_m200_relation_profile(m_plot, log10_c0_fit, alpha_fit, h=H_0)
        ax_top.plot(log10_m_plot, np.log10(c_median), color=COLOR_POSTERIOR_MEDIAN, linewidth=2, label="Posterior Median")

        sigma_band = max(float(sigma_int), 1e-6) if sigma_int is not None else 0.0

        if all(
            value is not None
            for value in (
                log10_c0_fit_eti_low,
                log10_c0_fit_eti_high,
                alpha_fit_eti_low,
                alpha_fit_eti_high,
            )
        ):
            eti_curves = np.vstack(
                [
                    log10_c_m200_relation_profile(m_plot, log10_c0_value, alpha_value, h=H_0)
                    for log10_c0_value in (log10_c0_fit_eti_low, log10_c0_fit_eti_high)
                    for alpha_value in (alpha_fit_eti_low, alpha_fit_eti_high)
                ]
            )
            log10_eti_low_curve = np.nanmin(np.log10(eti_curves), axis=0) - sigma_band
            log10_eti_high_curve = np.nanmax(np.log10(eti_curves), axis=0) + sigma_band
            c_eti_low_curve = 10 ** log10_eti_low_curve
            c_eti_high_curve = 10 ** log10_eti_high_curve
            ax_top.fill_between(
                log10_m_plot,
                log10_eti_low_curve,
                log10_eti_high_curve,
                color=COLOR_POSTERIOR_MEDIAN,
                alpha=0.18,
                label=(
                    rf"{HDI_PROB2:.0%} ETI"
                    + (r" $\oplus\ \sigma_{int}$" if sigma_band > 0 else "")
                ),
            )
        else:
            if sigma_band > 0:
                c_eti_low_curve = 10 ** (np.log10(c_median) - sigma_band)
                c_eti_high_curve = 10 ** (np.log10(c_median) + sigma_band)
                ax_top.fill_between(
                    log10_m_plot,
                    np.log10(c_median) - sigma_band,
                    np.log10(c_median) + sigma_band,
                    color=COLOR_POSTERIOR_MEDIAN,
                    alpha=0.18,
                    label=r"Median $\pm\ \sigma_{int}$",
                )
            else:
                c_eti_low_curve = None
                c_eti_high_curve = None
    else:
        c_median = None
        c_eti_low_curve = None
        c_eti_high_curve = None

    title = "Dark Matter: Halo Concentration-Mass (c-M) Relation\n"

    log10_c_data = np.log10(c)
    _y_lo = np.percentile(log10_c_data, 2) - 0.2
    _y_hi = np.percentile(log10_c_data, 98) + 0.2
    ax_top.set_xlim(log10_m_plot[0] - 0.05, log10_m_plot[-1] + 0.05)
    ax_top.set_ylim(_y_lo, _y_hi)
    ax_top.set_xlabel(r"$\log_{10}(M_{200}/M_\odot)$", fontsize=12)
    ax_top.set_ylabel(r"$\log_{10}\,c_{200}$", fontsize=12)
    # ax_top.set_title(title, fontsize=13)
    if _cloud_drawn:
        from matplotlib.lines import Line2D as _Line2D
        _cloud_handle = _Line2D(
            [],
            [],
            linestyle="None",
            marker="o",
            markersize=3,
            markerfacecolor=COLOR_DATA_POINTS,
            markeredgecolor="none",
            alpha=0.4,
            label=f"Posterior samples ({_cloud_sample_count} per galaxy)",
        )
        handles, labels = ax_top.get_legend_handles_labels()
        ax_top.legend(
            [_cloud_handle] + handles,
            [_cloud_handle.get_label()] + labels,
            fontsize=10,
            loc="lower left",
        )
    else:
        ax_top.legend(fontsize=10, loc="lower left")
    # ax_top.grid(True, which="both", ls="--", alpha=0.5)

    c0_text = (
        _format_interval_supsub(log10_c0_fit, log10_c0_fit_eti_low, log10_c0_fit_eti_high, decimals=3)
        if log10_c0_fit is not None
        else "n/a"
    )
    alpha_text = (
        _format_interval_supsub(
            alpha_fit,
            alpha_fit_eti_low,
            alpha_fit_eti_high,
            decimals=3,
        )
        if alpha_fit is not None
        else "n/a"
    )
    sigma_text = (
        _format_interval_supsub(
            sigma_int,
            sigma_int_eti_low,
            sigma_int_eti_high,
            decimals=3,
        )
        if sigma_int is not None
        else "n/a"
    )
    infer_text = (
        rf"Posterior Median (95% ETI): " "\n"
        rf"$\log_{{10}}c_0 = {c0_text}$, $\alpha = {alpha_text}$" "\n"
        rf"$\sigma_{{int}} = {sigma_text}$"
        # rf"\n\n"
        # rf"Dutton \& Macciò (2014) $c-M$ relation:" "\n"
        # rf"$\log_{{10}}c_0 = {ref_log10_c0:.3f}$, $\alpha = {ref_alpha:.3f}$, $\sigma_{{\log c}} = {ref_log10_sigma:.2f}$ dex" "\n\n"
        # rf"Li et al. (2020) $c-M$ relation:" "\n"
        # rf"$\log_{{10}}c_0 = {LOG10_C0_LI20:.3f} \pm {LOG10_C0_SIGMA_LI20:.3f}$, $\alpha = {ALPHA_LI20:.3f} \pm {ALPHA_SIGMA_LI20:.3f}$, $\sigma_{{\log c}} = {LOG10_C_SCATTER_LI20:.2f}$ dex" "\n\n"
        # rf"Yasin et al. (2023) $c-M$ relation:" "\n"
        # rf"$\log_{{10}}c_0 = {LOG10_C0_YASIN23:.3f} \pm {LOG10_C0_SIGMA_YASIN23:.3f}$, $\alpha = {ALPHA_YASIN23:.3f} \pm {ALPHA_SIGMA_YASIN23:.3f}$, $\sigma_{{\log c}} = {LOG10_C_SCATTER_YASIN23:.2f}$ dex"
    )
    ax_top.text(
        0.98,
        0.02,
        infer_text,
        transform=ax_top.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    result_dir.mkdir(parents=True, exist_ok=True)
    plot_path = result_dir / f"c-M_relation_all.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.savefig(plot_path.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    print(f"Spaghetti plot saved to {plot_path}")


def _gpdfit(z_sorted: np.ndarray) -> float:
    """Estimate the Pareto shape parameter k of the GPD via profile log-likelihood.

    Implements the estimator from Vehtari et al. (2019) / Zhang & Stephens (2009).

    Args:
        z_sorted: sorted, non-negative 1-D array of upper-tail raw importance weights
                  shifted so the minimum is 0. Must be sorted ascending.

    Returns:
        k: Pareto shape parameter.
            k < 0.5  → IS estimate reliable
            0.5 ≤ k < 0.7 → marginal; apply caution
            k ≥ 0.7  → IS estimate unreliable (weights dominated by rare draws)
    """
    M = len(z_sorted)
    if M < 5 or z_sorted[-1] <= 0.0:
        return 0.0

    # Build a grid of candidate b = -k/σ values (must be < 0 for Pareto tail).
    # Grid: b_j = (1 - sqrt(m_grid / (j - 0.5))) / z_max
    # For j small, sqrt(...) > 1 → b_j < 0.  We keep only b_j < 0.
    m_grid = 30 + int(np.ceil(np.sqrt(M)))
    j = np.arange(1, m_grid + 1, dtype=float)
    b_ary = (1.0 - np.sqrt(m_grid / (j - 0.5))) / z_sorted[-1]
    b_ary = b_ary[b_ary < 0.0]
    if len(b_ary) == 0:
        return 0.0

    # Profile log-likelihood: L*(b) = M*(log(-b) - 1) + Σ_j log(1 - b*z_j)
    # (derived by profiling out σ; equivalent to the Zhang & Stephens (2009) score)
    logml = M * (np.log(-b_ary) - 1.0) + np.array(
        [np.sum(np.log1p(-b * z_sorted)) for b in b_ary]
    )

    # Bayesian model averaging: weight candidate b values by profile likelihood.
    logml -= logml.max()
    w = np.exp(logml)
    w /= w.sum()
    b_hat = float(w @ b_ary)  # weighted mean of candidate b values

    # k_hat = mean_j log(1 - b_hat * z_j); for b_hat < 0 and z_j ≥ 0, this is ≥ 0.
    k = float(np.mean(np.log1p(-b_hat * z_sorted)))
    return max(k, 0.0)


def _is_ess_from_log_weights(log_w: np.ndarray) -> float:
    """Compute the effective sample size (ESS) for a set of importance weights.

    ESS = (Σ w_s)² / Σ w_s²  — equivalently, 1 / Σ w̃_s² for normalised weights.
    Range: (0, S] where S = len(log_w).  ESS/S close to 1 means all weights equal.
    """
    log_w = log_w - log_w.max()  # numerical stability
    w = np.exp(log_w)
    ess = (w.sum() ** 2) / np.sum(w ** 2)
    return float(ess)


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
    """Compute Pareto-Smoothed Importance Sampling (PSIS) diagnostics.

    For each galaxy i, evaluates per-sample importance log-weights at the
    posterior-median population hyperparameters Φ*:

        log w_is = log p_pop(θ_is | Φ*) - log p_stage1(θ_is)

    where p_pop factorises as N(m_is; μ_M, σ_M) × N(ℓ_is; c₀ + α m_is, σ_int)
    and p_stage1 is the single-galaxy prior (TruncNorm × Normal).

    Reports per-galaxy ESS and the Pareto shape k̂ from a GPD fit to the upper tail
    of the weight distribution. Saves two standalone diagnostic figures.

    Reference:
        Vehtari et al. (2019), "Pareto Smoothed Importance Sampling", JMLR 25(72).

    Args:
        ...same sample/prior inputs as fit_m200_c_mcmc...
        fit_results:  dict returned by fit_m200_c_mcmc (must contain likelihood_mode
                      == "samples", M200_mu_median, M200_sigma_median, log10_c0_median,
                      alpha_median, sigma_int_median).
        plot_suffix:  appended to output filename.
        sample_cap:   cap on per-galaxy samples (identical to fit_m200_c_mcmc).

    Returns:
        dict with keys "k_hat" (array), "ess" (array), "ess_frac" (array),
        "n_bad_k" (int: count with k̂ ≥ 0.7), "n_warn_k" (int: count with 0.5 ≤ k̂ < 0.7).
        Returns None if inputs are insufficient.
    """
    if fit_results is None or fit_results.get("likelihood_mode") != "samples":
        return None

    # Retrieve posterior-median population hyperparameters.
    try:
        log10_c0 = float(fit_results["log10_c0_median"])
        alpha = float(fit_results["alpha_median"])
        sigma_int = float(fit_results["sigma_int_median"])
        M200_mu = float(fit_results["M200_mu_median"])
        M200_sigma = float(fit_results["M200_sigma_median"])
        nu_pop = float(fit_results.get("nu_pop_median", 30.0))  # default ≈ Normal if absent
    except (KeyError, TypeError, ValueError) as exc:
        print(f"PSIS diagnostic: missing fit_results keys: {exc}")
        return None

    if sigma_int <= 0 or M200_sigma <= 0:
        return None

    log10_M_pivot = np.log10(M_PIVOT_H_INV / H_0)

    # Build per-galaxy sample arrays (same as _prepare_sample_posterior_tensors).
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

    N_gal, S_max, _ = sample_points.shape

    # Evaluate log p_pop(m, ℓ | Φ*) using factored Student-t:
    #   m       ~ t_nu(M200_mu, M200_sigma)
    #   ell | m ~ t_nu(c₀ + α·m, σ_int)
    m = sample_points[:, :, 0]   # (N_gal, S_max) — log10(M200/Mpivot)
    ell = sample_points[:, :, 1]  # (N_gal, S_max) — log10(c)

    log_p_pop_m = t_dist.logpdf(m, df=nu_pop, loc=M200_mu, scale=M200_sigma)
    log_p_pop_c = t_dist.logpdf(ell, df=nu_pop, loc=log10_c0 + alpha * m, scale=sigma_int)
    log_p_pop = log_p_pop_m + log_p_pop_c  # (N_gal, S_max)

    # PSIS diagnostic MUST use plain IS weights log(p_pop / p_stage1).
    # _gpdfit is calibrated for unbounded Pareto tails of plain IS weights.
    # Defensive IS weights are bounded (≤ 1/(1-ε)), causing all top-M_tail
    # weights to pile up within a tiny span (δ ≈ 0.01); the GPD grid then
    # operates in a numerically degenerate regime and produces spurious k̂ >> 0.7.
    log_w_all = log_p_pop - sample_log_prior  # (N_gal, S_max)

    k_hat = np.full(N_gal, np.nan)
    ess = np.full(N_gal, np.nan)

    for i in range(N_gal):
        mask_i = sample_mask[i]  # (S_max,) bool
        if not np.any(mask_i):
            continue
        lw = log_w_all[i, mask_i]  # valid log weights for galaxy i
        S_i = len(lw)

        # ESS
        ess[i] = _is_ess_from_log_weights(lw)

        # PSIS: fit GPD to the top M = min(S//5, ceil(3*sqrt(S))) raw weights.
        M_tail = min(S_i // 5, int(np.ceil(3.0 * np.sqrt(S_i))))
        M_tail = max(M_tail, 5)  # need at least 5 tail points
        if M_tail >= S_i:
            k_hat[i] = 0.0
            continue
        lw_sorted = np.sort(lw)
        tail = lw_sorted[-M_tail:]
        raw_tail = np.exp(tail - tail[-1])
        z = raw_tail - raw_tail[0]  # shift minimum to 0 after mapping back to raw weights
        k_hat[i] = _gpdfit(z)

    # Compute ESS fraction relative to raw sample count.
    raw_counts = np.array([int(np.sum(sample_mask[i])) for i in range(N_gal)], dtype=float)
    ess_frac = np.where(raw_counts > 0, ess / raw_counts, np.nan)

    valid = np.isfinite(k_hat)
    n_bad = int(np.sum(k_hat[valid] >= 0.7))
    n_warn = int(np.sum((k_hat[valid] >= 0.5) & (k_hat[valid] < 0.7)))
    n_good = int(np.sum(k_hat[valid] < 0.5))

    print(
        f"\nPSIS diagnostic ({np.sum(valid)} galaxies): "
        f"k̂<0.5: {n_good}  0.5≤k̂<0.7: {n_warn}  k̂≥0.7: {n_bad}"
    )
    if n_bad > 0:
        print(
            f"  WARNING: {n_bad} galaxies have k̂ ≥ 0.7; "
            "their IS contribution to the population likelihood may be unreliable."
        )

    if save_plots and np.any(valid):
        k_valid = k_hat[valid]
        ess_frac_valid = ess_frac[valid]

        suffix = plot_suffix if plot_suffix else "_all"
        result_dir.mkdir(parents=True, exist_ok=True)

        bins_k = np.linspace(0.0, max(1.0, float(np.nanmax(k_valid)) * 1.1), 30)
        fig_k, ax_k = plt.subplots(1, 1, figsize=(6.2, 4.6))
        ax_k.hist(k_valid, bins=bins_k, color=COLOR_LOW_N, edgecolor="white", linewidth=0.4, alpha=0.85)
        ax_k.axvline(0.5, color="#E69F00", linewidth=1.0, linestyle="--", label=r"Caution threshold: $\hat{k}=0.5$")
        ax_k.axvline(0.7, color="#D55E00", linewidth=1.0, linestyle="--", label=r"Unreliable threshold: $\hat{k}=0.7$")
        ax_k.set_xlabel(r"Pareto shape $\hat{k}$")
        ax_k.set_ylabel("Number of galaxies")
        ax_k.set_title(r"PSIS Pareto-tail diagnostic per galaxy", fontsize=11)
        ax_k.legend(fontsize=8)
        ax_k.text(
            0.97, 0.97,
            "\n".join([
                f"Reliable (<0.5): {n_good}",
                f"Caution (0.5-0.7): {n_warn}",
                f"Unreliable (>=0.7): {n_bad}",
            ]),
            transform=ax_k.transAxes,
            ha="right", va="top", multialignment="left", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
        fig_k.tight_layout()
        out_k_path = result_dir / f"psis_importance_diagnostics_khat{suffix}.png"
        fig_k.savefig(out_k_path, dpi=300, bbox_inches="tight")
        fig_k.savefig(out_k_path.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
        plt.close(fig_k)

        order = np.argsort(ess_frac_valid)
        fig_ess, ax_ess = plt.subplots(1, 1, figsize=(6.2, 4.6))
        ax_ess.scatter(
            np.arange(len(order)),
            ess_frac_valid[order],
            s=4, c=COLOR_DATA_POINTS, alpha=0.5, linewidths=0,
        )
        ax_ess.axhline(0.1, color="#D55E00", linewidth=1.2, linestyle="--", label="Reference threshold: ESS / S = 0.1")
        ax_ess.set_xlabel("Galaxy rank (sorted by ESS/S)")
        ax_ess.set_ylabel("Effective sample fraction (ESS / S)")
        ax_ess.set_title("Importance-sampling efficiency per galaxy", fontsize=11)
        ax_ess.set_ylim(0, 1.05)
        ax_ess.legend(fontsize=8)
        fig_ess.tight_layout()
        out_ess_path = result_dir / f"psis_importance_diagnostics_ess{suffix}.png"
        fig_ess.savefig(out_ess_path, dpi=300, bbox_inches="tight")
        fig_ess.savefig(out_ess_path.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
        plt.close(fig_ess)
        print(f"PSIS diagnostic figures saved to {out_k_path} and {out_ess_path}")

    return {
        "k_hat": k_hat,
        "ess": ess,
        "ess_frac": ess_frac,
        "n_bad_k": n_bad,
        "n_warn_k": n_warn,
        "n_good_k": n_good,
    }


def plot_population_inference_diagnostics(
    M200: np.ndarray,
    c: np.ndarray,
    fit_results: dict | None = None,
    plot_suffix: str = "",
    sample_m200=None,
    sample_c=None,
    sample_plot: int = 20,
) -> Path | None:
    """Create a compact residual-diagnostics figure for the population-level c-M fit."""
    if not fit_results:
        return None

    log10_c0_fit = fit_results.get("log10_c0_median")
    alpha_fit = fit_results.get("alpha_median")
    sigma_int = fit_results.get("sigma_int_median")
    if log10_c0_fit is None or alpha_fit is None or sigma_int is None:
        return None

    valid_mask = (M200 > 0) & (c > 0) & np.isfinite(M200) & np.isfinite(c)
    if not np.any(valid_mask):
        return None

    M200 = np.asarray(M200, dtype=float)[valid_mask]
    c = np.asarray(c, dtype=float)[valid_mask]
    c_pred = log10_c_m200_relation_profile(M200, float(log10_c0_fit), float(alpha_fit), h=H_0)

    log10_residuals = np.log10(c) - np.log10(c_pred)
    sigma_int = max(float(sigma_int), 1e-6)
    standardized_residuals = log10_residuals / sigma_int
    rmse_linear = np.sqrt(np.mean((c - c_pred) ** 2))
    nrmse_median = rmse_linear / (np.mean(c) if np.mean(c) > 0 else 1.0)
    log10_rmse = np.sqrt(np.mean(log10_residuals ** 2))
    dof = int(max(len(M200) - 2, 1))
    redchi_median = float(np.sum(standardized_residuals ** 2) / dof)

    within_1sigma = float(np.mean(np.abs(standardized_residuals) <= 1.0))
    within_2sigma = float(np.mean(np.abs(standardized_residuals) <= 2.0))

    log10_sigma_band = sigma_int
    hist_max = max(4.0, np.nanmax(np.abs(standardized_residuals)) * 1.1)
    bins = np.linspace(-hist_max, hist_max, 28)
    x_pdf = np.linspace(-hist_max, hist_max, 400)
    result_dir.mkdir(parents=True, exist_ok=True)
    suffix = plot_suffix if plot_suffix else "_all"
    plot_path = result_dir / f"c-M_relation_diagnostics{suffix}.png"

    fig_resid = plt.figure(figsize=(6.0, 5.6))
    ax_resid = fig_resid.add_subplot(1, 1, 1)
    ax_resid.axhspan(-2 * log10_sigma_band, 2 * log10_sigma_band, color=COLOR_HDI_BAND, alpha=0.18)
    ax_resid.axhspan(-log10_sigma_band, log10_sigma_band, color=COLOR_LOW_N, alpha=0.10)
    ax_resid.axhline(0.0, color="0.2", linewidth=1.1)

    # --- Posterior sample residuals ---
    _drew_samples = False
    if sample_m200 is not None and sample_c is not None:
        rng = np.random.default_rng(0)
        samples_m_masked = np.asarray(sample_m200, dtype=object)[valid_mask]
        samples_c_masked = np.asarray(sample_c, dtype=object)[valid_mask]
        pool_M200_s: list[np.ndarray] = []
        pool_resid_s: list[np.ndarray] = []
        for gal_m200, gal_c in zip(samples_m_masked, samples_c_masked):
            if gal_m200 is None or gal_c is None:
                continue
            arr_m = np.asarray(gal_m200, dtype=float)
            arr_c = np.asarray(gal_c, dtype=float)
            finite = np.isfinite(arr_m) & np.isfinite(arr_c)
            if not np.any(finite):
                continue
            arr_m, arr_c = arr_m[finite], arr_c[finite]
            n = len(arr_m)
            if n > sample_plot:
                idx = rng.choice(n, size=sample_plot, replace=False)
                arr_m, arr_c = arr_m[idx], arr_c[idx]
            M200_s = 10.0 ** arr_m
            c_pred_s = log10_c_m200_relation_profile(M200_s, float(log10_c0_fit), float(alpha_fit), h=H_0)
            resid_s = arr_c - np.log10(c_pred_s)
            pool_M200_s.append(M200_s)
            pool_resid_s.append(resid_s)
        if pool_M200_s:
            ax_resid.scatter(
                np.concatenate(pool_M200_s),
                np.concatenate(pool_resid_s),
                s=6, color=COLOR_DATA_POINTS, alpha=0.15, edgecolors="none", rasterized=True, zorder=1,
            )
            _drew_samples = True
    if not _drew_samples:
        ax_resid.scatter(M200, log10_residuals, s=18, color=COLOR_DATA_POINTS, alpha=0.55, edgecolors="none")
    ax_resid.set_xscale("log")
    ax_resid.set_xlabel(r"$M_{200} \ [M_\odot]$")
    ax_resid.set_ylabel(r"$\Delta \log_{10} c$")
    ax_resid.set_title("Residuals vs. Halo Mass", fontsize=11)
    fig_resid.savefig(plot_path.with_name(f"{plot_path.stem}_residuals{plot_path.suffix}"), dpi=300, bbox_inches="tight")
    fig_resid.savefig(plot_path.with_name(f"{plot_path.stem}_residuals.pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig_resid)

    fig_hist = plt.figure(figsize=(6.0, 5.6))
    ax_hist = fig_hist.add_subplot(1, 1, 1)
    ax_hist.hist(standardized_residuals, bins=bins, density=True, histtype="stepfilled", color=COLOR_LOW_N, alpha=0.22, edgecolor=COLOR_LOW_N, linewidth=1.2)
    ax_hist.plot(x_pdf, norm.pdf(x_pdf, loc=0.0, scale=1.0), color=COLOR_HIGH_N, linewidth=1.8)
    ax_hist.axvline(0.0, color="0.2", linewidth=1.1)
    ax_hist.axvline(-1.0, color="0.5", linewidth=1.0, linestyle="--")
    ax_hist.axvline(1.0, color="0.5", linewidth=1.0, linestyle="--")
    ax_hist.set_xlabel(r"$\Delta \log_{10} c / \sigma_{\mathrm{int}}$")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title("Standardized Residual Distribution", fontsize=11)
    coverage_text = (
        rf"$|\Delta|/\sigma_{{\mathrm{{int}}}}\leq1$: {within_1sigma:.1%}" "\n"
        rf"$|\Delta|/\sigma_{{\mathrm{{int}}}}\leq2$: {within_2sigma:.1%}"
    )
    ax_hist.text(
        0.97, 0.97, coverage_text,
        transform=ax_hist.transAxes,
        ha="right", va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.7", alpha=0.85),
    )
    fig_hist.savefig(plot_path.with_name(f"{plot_path.stem}_hist{plot_path.suffix}"), dpi=300, bbox_inches="tight")
    fig_hist.savefig(plot_path.with_name(f"{plot_path.stem}_hist.pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig_hist)

    summary_lines = [
        r"Population-fit diagnostics",
        rf"$N = {len(M200)}$, ",
        rf"$\log_{{10}}c_0 = {float(log10_c0_fit):.3f}$, $\alpha = {float(alpha_fit):.3f}$, $\sigma_{{\mathrm{{int}}}} = {sigma_int:.3f}$",
        rf"$|r|/\sigma_{{\mathrm{{int}}}} \leq 1$: {within_1sigma:.1%}, $|r|/\sigma_{{\mathrm{{int}}}} \leq 2$: {within_2sigma:.1%}",
    ]
    print(f"Population diagnostic plot saved to {plot_path}")
    return plot_path


def _fit_all_data_if_requested(
    M200: np.ndarray,
    c: np.ndarray,
    sample_m200=None,
    sample_c=None,
    sample_m200_prior_mu=None,
    sample_m200_prior_sigma=None,
    sample_m200_prior_lower=None,
    sample_m200_prior_upper=None,
    sample_c_prior_mu=None,
    sample_c_prior_sigma=None,
    log10_gmm_weights=None,
    log10_gmm_means=None,
    log10_gmm_covariances=None,
    sample_cap: int | None = None,
):
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
        dataset_label="all",
    )


def _slice_pipeline_inputs(mask: np.ndarray, **arrays) -> dict:
    sliced = {}
    for key, value in arrays.items():
        if value is None:
            sliced[key] = None
            continue

        if key in {
            "sample_m200",
            "sample_c",
            "gmm_weights",
            "gmm_means",
            "gmm_covariances",
        }:
            sliced[key] = np.array(value[mask], dtype=object)
        else:
            sliced[key] = np.asarray(value, dtype=float)[mask]

    return sliced


def _run_pipeline(
    mask: np.ndarray,
    sersic_n_raw: np.ndarray,
    M200_raw: np.ndarray,
    c_raw: np.ndarray,
    plot_suffix: str,
    label: str,
    sample_m200_raw=None,
    sample_c_raw=None,
    sample_m200_prior_mu_raw=None,
    sample_m200_prior_sigma_raw=None,
    sample_m200_prior_lower_raw=None,
    sample_m200_prior_upper_raw=None,
    sample_c_prior_mu_raw=None,
    sample_c_prior_sigma_raw=None,
    gmm_weights_raw=None,
    gmm_means_raw=None,
    gmm_covariances_raw=None,
    sample_cap: int | None = None,
    sample_plot: int = 20,
):
    print(f"\n=== Running HBM pipeline for {label} data ===")
    if M200_raw is None or c_raw is None:
        print(f"Warning: Missing M200/c summary data for {label}; skipping.")
        return None

    M200 = np.asarray(M200_raw, dtype=float)[mask]
    c = np.asarray(c_raw, dtype=float)[mask]
    sersic_n = sersic_n_raw[mask]

    sliced = _slice_pipeline_inputs(
        mask,
        sample_m200=sample_m200_raw,
        sample_c=sample_c_raw,
        sample_m200_prior_mu=sample_m200_prior_mu_raw,
        sample_m200_prior_sigma=sample_m200_prior_sigma_raw,
        sample_m200_prior_lower=sample_m200_prior_lower_raw,
        sample_m200_prior_upper=sample_m200_prior_upper_raw,
        sample_c_prior_mu=sample_c_prior_mu_raw,
        sample_c_prior_sigma=sample_c_prior_sigma_raw,
        gmm_weights=gmm_weights_raw,
        gmm_means=gmm_means_raw,
        gmm_covariances=gmm_covariances_raw,
    )

    print(f"Data points after filtering: {len(M200)} (dropped {len(sersic_n_raw) - len(M200)})")

    if len(M200) < 3:
        print("Not enough valid data points for HBM fitting.")
        return None

    fit_results_all = _fit_all_data_if_requested(
        M200,
        c,
        sample_m200=sliced["sample_m200"],
        sample_c=sliced["sample_c"],
        sample_m200_prior_mu=sliced["sample_m200_prior_mu"],
        sample_m200_prior_sigma=sliced["sample_m200_prior_sigma"],
        sample_m200_prior_lower=sliced["sample_m200_prior_lower"],
        sample_m200_prior_upper=sliced["sample_m200_prior_upper"],
        sample_c_prior_mu=sliced["sample_c_prior_mu"],
        sample_c_prior_sigma=sliced["sample_c_prior_sigma"],
        log10_gmm_weights=sliced["gmm_weights"],
        log10_gmm_means=sliced["gmm_means"],
        log10_gmm_covariances=sliced["gmm_covariances"],
        sample_cap=sample_cap,
    )
    plot_m200_c_relation_all(
        M200,
        c,
        fit_results=fit_results_all,
        plot_suffix=plot_suffix,
        sample_m200=sliced["sample_m200"],
        sample_c=sliced["sample_c"],
        sample_plot=sample_plot,
    )
    plot_population_inference_diagnostics(
        M200,
        c,
        fit_results=fit_results_all,
        plot_suffix=plot_suffix,
        sample_m200=sliced["sample_m200"],
        sample_c=sliced["sample_c"],
        sample_plot=sample_plot,
    )
    if fit_results_all and fit_results_all.get("likelihood_mode") == "samples":
        compute_psis_importance_diagnostics(
            log10_M200_posterior_samples=sliced["sample_m200"],
            log10_c_posterior_samples=sliced["sample_c"],
            log10_M200_prior_mu=sliced["sample_m200_prior_mu"],
            log10_M200_prior_sigma=sliced["sample_m200_prior_sigma"],
            log10_M200_prior_lower=sliced["sample_m200_prior_lower"],
            log10_M200_prior_upper=sliced["sample_m200_prior_upper"],
            log10_c_prior_mu=sliced["sample_c_prior_mu"],
            log10_c_prior_sigma=sliced["sample_c_prior_sigma"],
            fit_results=fit_results_all,
            plot_suffix=plot_suffix,
            sample_cap=sample_cap,
        )
    plt.show()
    return fit_results_all


def main(
    use_gmm: bool = False,
    use_samples: bool = False,
    sample_cap: int | None = None,
    nrmse_threshold: float | None = None,
    sample_plot: int = 20,
    result_dir_override: str | Path | None = None,
    ifu_ids: list[str] | None = None,
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
    Main execution function.
    """
    data = get_m200_c_data(
        result_dir_override=result_dir_override,
        ifu_ids=ifu_ids,
        nrmse_threshold=nrmse_threshold,
        quality_cut=quality_cut,
        max_redchi=max_redchi,
        ppc_p_min=ppc_p_min,
        ppc_p_max=ppc_p_max,
        ppc_value_coverage_min=ppc_value_coverage_min,
        ppc_overlap_min=ppc_overlap_min,
        max_abs_c_m200_corr=max_abs_c_m200_corr,
        filter_mass=filter_mass,
        filter_n=filter_n,
    )
    if not data:
        print("Failed to load data. Exiting.")
        return

    active_result_dir = _resolve_result_dir(result_dir_override)
    filtered_ifu_file = active_result_dir / "plateifus-filtered.txt"
    with open(filtered_ifu_file, "w") as _f:
        _f.write("\n".join(data["plate_ifu"]) + "\n")
    print(f"Saved {len(data['plate_ifu'])} filtered IFU IDs to {filtered_ifu_file}")

    sersic_n_raw = np.array(data["sersic_n"], dtype=float)
    M200_raw = np.array(data["M200"], dtype=float) if data.get("M200") is not None else None
    c_raw = np.array(data["c"], dtype=float) if data.get("c") is not None else None
    sersic_n_raw = np.where(np.isfinite(sersic_n_raw), sersic_n_raw, 0.0)

    if not use_samples and not use_gmm:
        use_gmm = True
        print("No likelihood mode selected; defaulting to GMM likelihood.")

    log10_M200_posterior_samples_raw = (
        data.get("log10_M200_posterior_samples") if use_samples else None
    )
    log10_c_posterior_samples_raw = data.get("log10_c_posterior_samples") if use_samples else None
    log10_M200_prior_mu_raw = data.get("log10_M200_prior_mu") if use_samples else None
    log10_M200_prior_sigma_raw = data.get("log10_M200_prior_sigma") if use_samples else None
    log10_M200_prior_lower_raw = data.get("log10_M200_prior_lower") if use_samples else None
    log10_M200_prior_upper_raw = data.get("log10_M200_prior_upper") if use_samples else None
    log10_c_prior_mu_raw = data.get("log10_c_prior_mu") if use_samples else None
    log10_c_prior_sigma_raw = data.get("log10_c_prior_sigma") if use_samples else None
    log10_gmm_weights_raw = data.get("log10_gmm_weights") if use_gmm else None
    log10_gmm_means_raw = data.get("log10_gmm_means") if use_gmm else None
    log10_gmm_covariances_raw = data.get("log10_gmm_covariances") if use_gmm else None

    print(f"Posterior-sample likelihood enabled: {use_samples}")
    print(f"GMM likelihood enabled: {use_gmm}")
    if sample_cap is not None:
        print(f"Posterior sample cap: {sample_cap}")
    print(f"Posterior samples per galaxy in plot: {sample_plot}")

    mask = np.ones_like(sersic_n_raw, dtype=bool)

    _run_pipeline(
        mask=mask,
        sersic_n_raw=sersic_n_raw,
        M200_raw=M200_raw,
        c_raw=c_raw,
        plot_suffix="",
        label="summary",
        sample_m200_raw=log10_M200_posterior_samples_raw,
        sample_c_raw=log10_c_posterior_samples_raw,
        sample_m200_prior_mu_raw=log10_M200_prior_mu_raw,
        sample_m200_prior_sigma_raw=log10_M200_prior_sigma_raw,
        sample_m200_prior_lower_raw=log10_M200_prior_lower_raw,
        sample_m200_prior_upper_raw=log10_M200_prior_upper_raw,
        sample_c_prior_mu_raw=log10_c_prior_mu_raw,
        sample_c_prior_sigma_raw=log10_c_prior_sigma_raw,
        gmm_weights_raw=log10_gmm_weights_raw,
        gmm_means_raw=log10_gmm_means_raw,
        gmm_covariances_raw=log10_gmm_covariances_raw,
        sample_cap=sample_cap,
        sample_plot=sample_plot,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fit c-M200 relation.")
    parser.add_argument(
        "--merge-samples",
        action="store_true",
        help=(
            "Merge per-IFU posterior sample files in the results directory into "
            "the aggregate NetCDF file and keep the original files"
        ),
    )
    parser.add_argument(
        "--gmm",
        action="store_true",
        help="Use GMM likelihood when GMM parameters are available",
    )
    parser.add_argument(
        "--samples",
        action="store_true",
        help="Use saved posterior samples with first-stage prior correction when sample files are available",
    )
    parser.add_argument(
        "--sample-cap",
        type=int,
        help="Maximum number of posterior samples to use per galaxy when sample-based likelihoods are enabled, including --gen-sample-mode=psis-khat.",
    )
    parser.add_argument(
        "--sample-plot",
        type=int,
        default=20,
        help="Number of posterior samples to plot per galaxy in the c-M relation figure (default: 20)",
    )
    parser.add_argument(
        "--nrmse",
        type=float,
        help="Filter rows by nrmse, keeping only entries with nrmse smaller than this value",
    )
    parser.add_argument(
        "--quality-cut",
        type=str,
        choices=sorted(QUALITY_FILTER_PRESETS.keys()),
        default=None,
        help=(
            "Apply a recommended stage-one quality-cut preset based on single-galaxy "
            "diagnostics saved by dm.py. These cuts use redchi, PPC metrics, and the "
            "|corr(M200, c)| identifiability proxy, but never cut directly on M200 or c."
        ),
    )
    parser.add_argument(
        "--max-redchi",
        type=float,
        default=None,
        help="Keep only galaxies with stage-one reduced chi-squared <= this value.",
    )
    parser.add_argument(
        "--ppc-p-min",
        type=float,
        default=None,
        help="Keep only galaxies with weighted Student-t PPC p-value >= this value.",
    )
    parser.add_argument(
        "--ppc-p-max",
        type=float,
        default=None,
        help="Keep only galaxies with weighted Student-t PPC p-value <= this value.",
    )
    parser.add_argument(
        "--ppc-coverage-min",
        type=float,
        default=None,
        help="Keep only galaxies with PPC_ETI_VALUE_COVERAGE >= this value.",
    )
    parser.add_argument(
        "--ppc-overlap-min",
        type=float,
        default=None,
        help="Keep only galaxies with PPC_ETI_OVERLAP >= this value.",
    )
    parser.add_argument(
        "--max-abs-cm-corr",
        type=float,
        default=None,
        help="Keep only galaxies with |corr(M200, c)| <= this value in the stage-one posterior.",
    )
    parser.add_argument(
        "--plot-attrition",
        nargs=4,
        default=None,
        metavar=("FILE_A", "LABEL_A", "FILE_B", "LABEL_B"),
        help=(
            "Compare the distributions of two galaxy-ID text files. Provide four arguments: "
            "FILE_A LABEL_A FILE_B LABEL_B. Example: "
            "--plot-attrition plateifus.txt All plateifus-620.txt Selected"
        ),
    )
    parser.add_argument(
        "--attrition-output",
        type=Path,
        default=result_dir / "galaxy_select_compare.png",
        help="Output path for the attrition summary figure. A PDF with the same stem is also written.",
    )
    parser.add_argument(
        "--gen-sample",
        type=int,
        metavar="N",
        help=(
            "Draw N galaxies from the --ifu-file pool and write them to data/. "
            "Requires --ifu-file. Use --gen-sample-mode to select the sampling strategy. "
            "Recommended value: 60."
        ),
    )
    parser.add_argument(
        "--gen-sample-mode",
        type=str,
        default="mass-parent",
        choices=["random", "mass-quintile", "mass-parent", "sersic-quintile", "sersic-parent", "psis-khat"],
        metavar="MODE",
        help=(
            "Sampling strategy for --gen-sample. "
            "random: uniformly draw galaxies from the candidate IFU pool without replacement. "
            "mass-quintile: uniform draw across 5 stellar-mass quintiles of the pool. "
            "mass-parent: weight pool to match stellar-mass distribution of data/plateifus.txt. "
            "sersic-quintile: uniform draw across 5 Sersic-n quintiles of the pool. "
            "sersic-parent: weight pool to match Sersic-n distribution of data/plateifus.txt. "
            "psis-khat: fit the population model on the candidate pool and take the galaxies with the smallest PSIS k-hat. "
            "Default: mass-parent."
        ),
    )
    parser.add_argument(
        "--gen-sample-output",
        type=str,
        default=None,
        metavar="FILENAME",
        help="Output filename for --gen-sample (filename only, no directory). Default: robustness_sample_N.txt.",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default=None,
        help=(
            "Override the result directory used to load nfw_param_cm200.csv and related "
            "sample files. Relative paths are resolved from the workspace root."
        ),
    )
    parser.add_argument(
        "--filter-mass",
        "-filter-mass",
        type=str,
        choices=["low", "median", "high"],
        default=None,
        help=(
            "Restrict the run to the low, median, or high equal-count stellar-mass tertile "
            "computed from all successful galaxies in the selected result directory."
        ),
    )
    parser.add_argument(
        "--filter-n",
        "-filter-n",
        type=str,
        choices=["low", "median", "high"],
        default=None,
        help=(
            "Restrict the run to the low, median, or high equal-count Sersic-n tertile "
            "computed from all successful galaxies in the selected result directory."
        ),
    )
    parser.add_argument(
        "--ifu-file",
        type=str,
        default=None,
        help=(
            "Path to a text file containing one IFU ID per line. When provided, only those "
            "galaxies are used in the current Stage 2 run."
        ),
    )
    args = parser.parse_args()

    default_attrition_output = result_dir / "galaxy_select_compare.png"
    active_result_dir = _set_result_dir(args.result_dir)

    ifu_ids = _load_ifu_id_list(args.ifu_file) if args.ifu_file is not None else None

    if args.merge_samples:
        merge_posterior_samples_file(result_dir_override=args.result_dir)
        raise SystemExit(0)

    if args.sample_cap is not None and args.sample_cap < 1:
        print("--sample-cap must be at least 1.")
        raise SystemExit(1)

    if args.sample_plot < 1:
        print("--sample-plot must be at least 1.")
        raise SystemExit(1)

    if args.nrmse is not None and args.nrmse < 0:
        print("--nrmse must be non-negative.")
        raise SystemExit(1)

    if args.max_redchi is not None and args.max_redchi <= 0:
        print("--max-redchi must be positive.")
        raise SystemExit(1)

    if args.ppc_p_min is not None and not (0.0 <= args.ppc_p_min <= 1.0):
        print("--ppc-p-min must be between 0 and 1.")
        raise SystemExit(1)

    if args.ppc_p_max is not None and not (0.0 <= args.ppc_p_max <= 1.0):
        print("--ppc-p-max must be between 0 and 1.")
        raise SystemExit(1)

    if (
        args.ppc_p_min is not None
        and args.ppc_p_max is not None
        and args.ppc_p_min > args.ppc_p_max
    ):
        print("--ppc-p-min cannot be larger than --ppc-p-max.")
        raise SystemExit(1)

    if args.ppc_coverage_min is not None and not (0.0 <= args.ppc_coverage_min <= 1.0):
        print("--ppc-coverage-min must be between 0 and 1.")
        raise SystemExit(1)

    if args.ppc_overlap_min is not None and not (0.0 <= args.ppc_overlap_min <= 1.0):
        print("--ppc-overlap-min must be between 0 and 1.")
        raise SystemExit(1)

    if args.max_abs_cm_corr is not None and not (0.0 <= args.max_abs_cm_corr <= 1.0):
        print("--max-abs-cm-corr must be between 0 and 1.")
        raise SystemExit(1)

    if args.plot_attrition:
        attrition_output = args.attrition_output
        if attrition_output == default_attrition_output:
            attrition_output = active_result_dir / "galaxy_select_compare.png"
        attrition_sample_specs = _build_attrition_sample_specs(args.plot_attrition)
        plot_path = plot_sample_attrition_pipeline(
            sample_specs=attrition_sample_specs,
            output_path=attrition_output,
        )
        print(f"Sample attrition figure saved to {plot_path}")
        raise SystemExit(0)

    if args.gen_sample is not None:
        if args.gen_sample < 10:
            print("--gen-sample must be at least 10.")
            raise SystemExit(1)
        if ifu_ids is None:
            print("--ifu-file is required when --gen-sample is used.")
            raise SystemExit(1)
        out_path = generate_robustness_sample(
            n_sample=args.gen_sample,
            nrmse_threshold=args.nrmse,
            output_filename=args.gen_sample_output,
            result_dir_override=args.result_dir,
            ifu_ids=ifu_ids,
            sample_cap=args.sample_cap,
            mode=args.gen_sample_mode,
        )
        print(f"Robustness subsample file: {out_path}")
        raise SystemExit(0)

    main(
        use_gmm=args.gmm,
        use_samples=args.samples,
        sample_cap=args.sample_cap,
        nrmse_threshold=args.nrmse,
        sample_plot=args.sample_plot,
        result_dir_override=args.result_dir,
        ifu_ids=ifu_ids,
        quality_cut=args.quality_cut,
        max_redchi=args.max_redchi,
        ppc_p_min=args.ppc_p_min,
        ppc_p_max=args.ppc_p_max,
        ppc_value_coverage_min=args.ppc_coverage_min,
        ppc_overlap_min=args.ppc_overlap_min,
        max_abs_c_m200_corr=args.max_abs_cm_corr,
        filter_mass=args.filter_mass,
        filter_n=args.filter_n,
    )
