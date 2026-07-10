"""Result-file I/O — CSV parameter storage, NetCDF posterior samples, and merging.

Functions extracted from ``src-orig/main.py`` and ``src-orig/m200.py``.

The functions no longer rely on module-level globals; paths are passed
explicitly.  Callers should obtain the result directory from
``src.config.settings.result_dir``.
"""

from __future__ import annotations

import ast
import os
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from src.config.settings import settings


# ═══════════════════════════════════════════════════════════════════════════
#  CSV parameter file I/O  (from main.py)
# ═══════════════════════════════════════════════════════════════════════════

def store_params_file(
    plate_ifu: str,
    fit_parameters: dict,
    filename: str,
    result_dir: Path | str,
    *,
    write_lock=None,
) -> None:
    """Store *fit_parameters* dict for *plate_ifu* in a CSV file.

    Parameters
    ----------
    plate_ifu : str
        Plate-IFU identifier.
    fit_parameters : dict
        Flat dict of parameter name → value.
    filename : str
        CSV filename (placed in *result_dir*).
    result_dir : Path or str
        Directory where the CSV is written.
    """
    output_file = Path(result_dir) / filename

    with write_lock if write_lock is not None else nullcontext():
        if output_file.exists():
            try:
                all_fit_parameters = pd.read_csv(
                    output_file, index_col=0
                ).to_dict(orient="index")
            except pd.errors.EmptyDataError:
                all_fit_parameters = {}
        else:
            all_fit_parameters = {}

        if plate_ifu in all_fit_parameters:
            del all_fit_parameters[plate_ifu]

        all_fit_parameters[plate_ifu] = fit_parameters

        df = pd.DataFrame.from_dict(all_fit_parameters, orient="index")
        df.rename_axis("PLATE_IFU", inplace=True)
        temp_output_file = output_file.with_name(
            f".{output_file.name}.{os.getpid()}.tmp"
        )
        try:
            df.to_csv(temp_output_file)
            os.replace(temp_output_file, output_file)
        finally:
            temp_output_file.unlink(missing_ok=True)


def get_params_file(
    plate_ifu: str,
    filename: str,
    result_dir: Path | str,
) -> dict | None:
    """Retrieve the stored parameter dict for *plate_ifu*, or None."""
    output_file = Path(result_dir) / filename
    if not output_file.exists():
        return None

    try:
        all_fit_parameters = pd.read_csv(
            output_file, index_col=0
        ).to_dict(orient="index")
    except pd.errors.EmptyDataError:
        return None

    return all_fit_parameters.get(plate_ifu)


def get_processed_plate_ifus(
    filename: str,
    result_dir: Path | str,
    *,
    successful_only: bool = False,
    required_sample_filename: str | None = None,
) -> set[str]:
    """Return the set of plate-IFU ids already present in the CSV."""
    output_file = Path(result_dir) / filename
    if not output_file.exists():
        return set()

    try:
        df = pd.read_csv(output_file, index_col=0)
    except pd.errors.EmptyDataError:
        return set()

    if successful_only:
        if "result" not in df.columns:
            return set()
        df = df[df["result"].astype(str).str.lower() == "success"]

    processed = {str(idx) for idx in df.index.tolist()}
    if required_sample_filename is not None:
        sample_output = Path(result_dir) / required_sample_filename
        processed = {
            plate_ifu
            for plate_ifu in processed
            if _get_posterior_sample_output_path(
                sample_output, plate_ifu
            ).exists()
        }
    return processed


def load_m200_c_result_table(
    result_dir: Path | str | None = None,
    filename: str | None = None,
) -> pd.DataFrame:
    """Load the Stage 1 NFW parameter table used as Stage 2 input."""
    active_result_dir = settings.resolve_result_dir(result_dir)
    table_filename = filename or settings.nfw_param_cm200_filename
    nfw_param_file = active_result_dir / table_filename
    if not nfw_param_file.exists():
        raise FileNotFoundError(f"Data file not found: {nfw_param_file}")

    df = pd.read_csv(nfw_param_file, index_col=0)
    df.index = df.index.map(str)
    return df


def _parse_object_column(df: pd.DataFrame, column_name: str) -> np.ndarray | None:
    if column_name not in df.columns:
        return None

    try:
        return np.array(
            [
                ast.literal_eval(value) if isinstance(value, str) else value
                for value in df[column_name]
            ],
            dtype=object,
        )
    except Exception as exc:
        print(f"Warning: Could not parse {column_name}: {exc}")
        return None


def attach_posterior_samples_to_m200_c_data(
    data: dict,
    plate_ifus: list[str],
    result_dir: Path | str | None = None,
    filename: str | None = None,
) -> dict:
    """Attach merged per-galaxy posterior samples to a Stage 2 data dict."""
    active_result_dir = settings.resolve_result_dir(result_dir)
    sample_filename = filename or settings.nfw_param_cm200_sample_filename
    sample_file = active_result_dir / sample_filename

    log10_m200_samples = np.array([None] * len(plate_ifus), dtype=object)
    log10_c_samples = np.array([None] * len(plate_ifus), dtype=object)
    sample_map = load_posterior_sample_map(sample_file, plate_ifus=plate_ifus)
    for idx, plate_ifu in enumerate(plate_ifus):
        samples = sample_map.get(str(plate_ifu))
        if samples is None:
            continue
        log10_m200_samples[idx] = samples[0]
        log10_c_samples[idx] = samples[1]

    data["log10_M200_posterior_samples"] = log10_m200_samples
    data["log10_c_posterior_samples"] = log10_c_samples
    return data


def m200_c_table_to_data_dict(
    df: pd.DataFrame,
    result_dir: Path | str | None = None,
    *,
    include_posterior_samples: bool = True,
) -> dict:
    """Convert a prepared Stage 2 table into model-input arrays."""
    plate_ifus = df.index.astype(str).tolist()
    data = {
        "plate_ifu": df.index.to_numpy(dtype=str),
        "log10_mstar": (
            df["log10_mstar"].values
            if "log10_mstar" in df.columns
            else np.full(len(df), np.nan)
        ),
        "sersic_n": (
            df["sersic_n"].values if "sersic_n" in df.columns else np.zeros(len(df))
        ),
        "nrmse": df["nrmse"].values if "nrmse" in df.columns else np.zeros(len(df)),
        "M200": df["M200"].values if "M200" in df.columns else None,
        "c": df["c"].values if "c" in df.columns else None,
        "log10_gmm_source": (
            df["log10_gmm_source"].values
            if "log10_gmm_source" in df.columns
            else None
        ),
        "log10_gmm_n_components": (
            df["log10_gmm_n_components"].values
            if "log10_gmm_n_components" in df.columns
            else None
        ),
        "log10_gmm_weights": _parse_object_column(df, "log10_gmm_weights"),
        "log10_gmm_means": _parse_object_column(df, "log10_gmm_means"),
        "log10_gmm_covariances": _parse_object_column(
            df,
            "log10_gmm_covariances",
        ),
        "log10_gmm_bic": (
            df["log10_gmm_bic"].values if "log10_gmm_bic" in df.columns else None
        ),
        "log10_gmm_bic_by_n": _parse_object_column(df, "log10_gmm_bic_by_n"),
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

    if include_posterior_samples:
        attach_posterior_samples_to_m200_c_data(
            data,
            plate_ifus,
            result_dir=result_dir,
        )
    return data


def get_m200_c_data(
    result_dir: Path | str | None = None,
    dataframe: pd.DataFrame | None = None,
    *,
    include_posterior_samples: bool = True,
) -> dict | None:
    """Load Stage 2 c-M200 model inputs from an already prepared table or CSV."""
    df = dataframe if dataframe is not None else load_m200_c_result_table(result_dir)
    if df.empty:
        print("Warning: No successful fits found in data.")
        return None
    return m200_c_table_to_data_dict(
        df,
        result_dir=result_dir,
        include_posterior_samples=include_posterior_samples,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Per-IFU posterior-sample NetCDF I/O  (from main.py)
# ═══════════════════════════════════════════════════════════════════════════

def _get_posterior_sample_output_path(
    output_file: Path, plate_ifu: str
) -> Path:
    """Return the per-IFU path: ``<plate_ifu>_<output_file.name>``."""
    return output_file.with_name(f"{plate_ifu}_{output_file.name}")


def store_posterior_samples_file(
    plate_ifu: str,
    posterior_samples: dict,
    filename: str,
    result_dir: Path | str,
) -> None:
    """Store posterior log10(M200) & log10(c) samples for a single IFU.

    Parameters
    ----------
    plate_ifu : str
    posterior_samples : dict
        Keys ``log10_M200_samples`` and ``log10_c_samples``, or
        legacy ``M200_samples`` / ``c_samples`` (which are log10-converted).
    filename : str
        Merged NetCDF filename; per-IFU file is ``<plate_ifu>_<filename>``.
    result_dir : Path or str
    """
    output_file = Path(result_dir) / filename
    per_ifu_output_file = _get_posterior_sample_output_path(
        output_file, str(plate_ifu)
    )

    log10_m200_raw = posterior_samples.get("log10_M200_samples")
    log10_c_raw = posterior_samples.get("log10_c_samples")

    if log10_m200_raw is None or log10_c_raw is None:
        legacy_m200 = posterior_samples.get("M200_samples")
        legacy_c = posterior_samples.get("c_samples")
        if legacy_m200 is not None and legacy_c is not None:
            log10_m200_raw = np.log10(np.asarray(legacy_m200, dtype=float))
            log10_c_raw = np.log10(np.asarray(legacy_c, dtype=float))
        else:
            log10_m200_raw = []
            log10_c_raw = []

    log10_m200 = np.asarray(log10_m200_raw, dtype=float).reshape(-1)
    log10_c = np.asarray(log10_c_raw, dtype=float).reshape(-1)

    if len(log10_m200) != len(log10_c):
        raise ValueError(
            f"Posterior sample length mismatch for {plate_ifu}: "
            f"{len(log10_m200)} != {len(log10_c)}"
        )

    dataset = xr.Dataset(
        data_vars={
            "log10_M200_samples": (
                ("sample",),
                log10_m200.astype(np.float64, copy=False),
            ),
            "log10_c_samples": (
                ("sample",),
                log10_c.astype(np.float64, copy=False),
            ),
            "sample_count": np.array(len(log10_m200), dtype=np.int32),
        },
        coords={
            "sample": np.arange(len(log10_m200), dtype=np.int32),
        },
        attrs={
            "description": (
                "Posterior log10-samples for NFW M200 and c "
                "for a single PLATE_IFU"
            ),
            "plate_ifu": str(plate_ifu),
            "storage_format": "per_ifu_netcdf",
        },
    )

    temp_output_file = per_ifu_output_file.with_name(
        f"{per_ifu_output_file.stem}.tmp{per_ifu_output_file.suffix}"
    )
    dataset.to_netcdf(temp_output_file)
    dataset.close()
    os.replace(temp_output_file, per_ifu_output_file)


# ═══════════════════════════════════════════════════════════════════════════
#  Merged posterior-sample NetCDF I/O  (from m200.py)
# ═══════════════════════════════════════════════════════════════════════════

def _is_plate_ifu_like(value: str) -> bool:
    """Return True if *value* looks like ``"PPPP-MMMM"``."""
    parts = str(value).split("-", 1)
    return len(parts) == 2 and all(p.isdigit() for p in parts)


def _infer_plate_ifu_from_sample_file(
    sample_path: Path, sample_file_name: str
) -> str | None:
    """Strip the known suffix from the filename to recover the plate-IFU."""
    suffix = f"_{sample_file_name}"
    if sample_path.name.endswith(suffix):
        return sample_path.name[: -len(suffix)]
    return None


def _collect_per_ifu_sample_files(sample_file: Path) -> list[Path]:
    """List per-IFU sample files matching ``*_<sample_file.name>``."""
    per_ifu_files: list[Path] = []
    per_ifu_pattern = f"*_{sample_file.name}"
    for candidate in sorted(sample_file.parent.glob(per_ifu_pattern)):
        plate_ifu = _infer_plate_ifu_from_sample_file(
            candidate, sample_file.name
        )
        if plate_ifu and _is_plate_ifu_like(plate_ifu):
            per_ifu_files.append(candidate)
    return per_ifu_files


def _extract_log10_sample_arrays(
    ds: xr.Dataset,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (log10_m200, log10_c) from a Dataset, supporting legacy keys."""
    if "log10_M200_samples" in ds and "log10_c_samples" in ds:
        return (
            np.asarray(ds["log10_M200_samples"].values, dtype=float).reshape(-1),
            np.asarray(ds["log10_c_samples"].values, dtype=float).reshape(-1),
        )
    if "M200_samples" in ds and "c_samples" in ds:
        return (
            np.log10(np.asarray(ds["M200_samples"].values, dtype=float).reshape(-1)),
            np.log10(np.asarray(ds["c_samples"].values, dtype=float).reshape(-1)),
        )
    raise KeyError(
        "Posterior sample variables not found. Expected either "
        "('log10_M200_samples', 'log10_c_samples') or "
        "('M200_samples', 'c_samples')."
    )


def _load_single_posterior_sample_file(
    sample_path: Path, sample_file_name: str
) -> tuple[str, np.ndarray, np.ndarray] | None:
    """Load a per-IFU NetCDF and return (plate_ifu, log10_M200, log10_c)."""
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
        print(
            f"Warning: Could not parse posterior sample file {sample_path}: {e}"
        )
        return None


def load_posterior_sample_map(
    sample_file: Path,
    plate_ifus: list[str] | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load merged posterior samples and return {plate_ifu: (log10_M200, log10_c)}.

    Parameters
    ----------
    sample_file : Path
        Path to the merged NetCDF (``*.nc``).
    plate_ifus : list[str] | None
        Optional filter — only return entries for these plate-ifus.

    Returns
    -------
    dict
        Keys are plate-ifu strings; values are tuples of
        ``(log10_M200_array, log10_c_array)``.
    """
    if not sample_file.exists():
        print(f"Warning: Merged posterior sample file {sample_file} not found.")
        return {}

    if sample_file.suffix.lower() != ".nc":
        print(
            f"Warning: Unsupported merged posterior sample file format: "
            f"{sample_file}"
        )
        return {}

    try:
        ds = xr.load_dataset(sample_file)
        sample_plate_ifus = ds.coords["plate_ifu"].astype(str).values.tolist()

        log10_m200, log10_c = _extract_log10_sample_arrays(ds)

        if log10_m200.ndim == 1:
            log10_m200 = log10_m200[None, :]
        if log10_c.ndim == 1:
            log10_c = log10_c[None, :]

        if "sample_count" in ds:
            sample_counts = np.asarray(
                ds["sample_count"].values, dtype=int
            )
        else:
            finite_mask = np.isfinite(log10_m200) & np.isfinite(log10_c)
            sample_counts = np.sum(finite_mask, axis=1, dtype=int)

        requested_plate_ifus: set[str] | None = None
        if plate_ifus is not None:
            requested_plate_ifus = {str(p) for p in plate_ifus}

        sample_map: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for idx, ifu in enumerate(sample_plate_ifus):
            ifu = str(ifu)
            if requested_plate_ifus is not None and ifu not in requested_plate_ifus:
                continue

            count = int(sample_counts[idx])
            if count <= 0:
                continue

            sample_map[ifu] = (
                log10_m200[idx, :count].astype(float, copy=False),
                log10_c[idx, :count].astype(float, copy=False),
            )

        ds.close()
        return sample_map
    except Exception as e:
        print(
            f"Warning: Could not parse merged posterior sample file "
            f"{sample_file}: {e}"
        )
    return {}


def merge_posterior_samples_file(
    filename: str,
    result_dir: Path | str,
    plate_ifus: set[str] | None = None,
) -> Path | None:
    """Merge selected per-IFU sample files into a single NetCDF.

    Parameters
    ----------
    filename : str
        NetCDF filename (placed in *result_dir*).
    result_dir : Path or str
        Directory containing the per-IFU ``<plate>_<filename>`` files and
        where the merged result is written.
    plate_ifus : set of str, optional
        If provided, merge only these plate-IFU identifiers.

    Returns
    -------
    Path or None
        Path to the merged file, or None if nothing to merge.
    """
    output_file = Path(result_dir) / filename
    per_ifu_files = _collect_per_ifu_sample_files(output_file)

    if not per_ifu_files:
        print(
            f"No per-IFU posterior sample files found in "
            f"{output_file.parent} matching '*_{output_file.name}'."
        )
        return None

    merged_rows: list[tuple[str, np.ndarray, np.ndarray]] = []
    for sample_path in per_ifu_files:
        loaded = _load_single_posterior_sample_file(
            sample_path, output_file.name
        )
        if loaded is None:
            continue
        plate_ifu, log10_m200, log10_c = loaded
        if not _is_plate_ifu_like(plate_ifu):
            continue
        if plate_ifus is not None and plate_ifu not in plate_ifus:
            continue
        merged_rows.append((plate_ifu, log10_m200, log10_c))

    if not merged_rows:
        print("No valid per-IFU posterior sample files could be merged.")
        return None

    merged_rows.sort(key=lambda row: row[0])
    max_sample_count = max(len(row[1]) for row in merged_rows)
    plate_ifus = [row[0] for row in merged_rows]
    sample_counts = np.array(
        [len(row[1]) for row in merged_rows], dtype=np.int32
    )

    log10_m200_values = np.full(
        (len(merged_rows), max_sample_count), np.nan, dtype=np.float64
    )
    log10_c_values = np.full(
        (len(merged_rows), max_sample_count), np.nan, dtype=np.float64
    )

    for idx, (_, lm200, lc) in enumerate(merged_rows):
        n = len(lm200)
        log10_m200_values[idx, :n] = lm200
        log10_c_values[idx, :n] = lc

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
            "description": (
                "Merged posterior log10-samples for NFW M200 and c "
                "across PLATE_IFU files"
            ),
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
        f"Merged {len(merged_rows)} per-IFU posterior sample files "
        f"into {output_file}."
    )
    return output_file
