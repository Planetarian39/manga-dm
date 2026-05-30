"""Stage 2 pipeline: population-level c-M200 relation inference.

Orchestration logic extracted from ``src-orig/m200.py``.
"""

from __future__ import annotations

from pathlib import Path

from src.config.settings import settings
from src.data.results import merge_posterior_samples_file


def run_stage2(
    fit: bool = False,
    quality_cut: str = "recommended",
    diagnose: bool = False,
    result_dir_override: str | Path | None = None,
    n_cores: int | None = None,
) -> None:
    """Orchestrate Stage 2 population-model inference.

    Parameters
    ----------
    fit : bool
        Run the full population MCMC fit.
    quality_cut : str
        Quality-filter preset name (``"recommended"`` or ``"strict"``).
    diagnose : bool
        Run PSIS diagnostics only (requires prior fit results).
    n_cores : int or None
        Number of chains to run in parallel.
    """
    result_dir = settings.resolve_result_dir(result_dir_override)
    result_dir.mkdir(parents=True, exist_ok=True)

    if fit:
        from src.models.population import fit_m200_c_population

        print(f"Running Stage 2 population MCMC fit (quality cut: {quality_cut})...")
        fit_m200_c_population(
            quality_cut=quality_cut,
            result_dir_override=result_dir,
        )
    elif diagnose:
        from src.models.population import run_m200_c_psis_diagnostics

        print("Running PSIS diagnostics...")
        run_m200_c_psis_diagnostics(
            quality_cut=quality_cut,
            result_dir_override=result_dir,
        )
    else:
        print("No action specified for Stage 2. Use --fit or --diagnose.")

    print("Stage 2 complete.")


def merge_samples(
    ifu_file: str | Path | None = None,
    result_dir_override: str | Path | None = None,
) -> None:
    """Merge per-IFU posterior sample files into a single NetCDF."""
    result_dir = settings.resolve_result_dir(result_dir_override)
    filename = settings.nfw_param_cm200_sample_filename
    merge_posterior_samples_file(filename=filename, result_dir=result_dir)
