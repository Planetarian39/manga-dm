"""Stage 2 pipeline: population-level c-M200 relation inference.

Orchestration logic extracted from ``src-orig/m200.py``.
"""

from __future__ import annotations

from pathlib import Path

from src.config.constants import QUALITY_FILTER_PRESETS
from src.config.settings import settings
from src.data.results import load_posterior_sample_map, merge_posterior_samples_file


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
    from pathlib import Path as _Path
    import sys as _sys

    # Add src-orig to path for legacy module access
    _old_root = _Path(__file__).resolve().parent.parent.parent / "src-orig"
    if str(_old_root) not in _sys.path:
        _sys.path.insert(0, str(_old_root))
    import m200 as _m200

    result_dir = settings.resolve_result_dir(result_dir_override)
    result_dir.mkdir(parents=True, exist_ok=True)

    # Configure legacy globals
    _m200._set_result_dir(result_dir)

    if n_cores is None:
        n_cores = getattr(settings, "n_cores", None)

    preset = QUALITY_FILTER_PRESETS.get(quality_cut, QUALITY_FILTER_PRESETS["recommended"])

    if fit:
        print(f"Running Stage 2 population MCMC fit (quality cut: {quality_cut})...")
        _m200.fit_m200_c_mcmc(
            # quality_thresholds=preset,
            n_cores=n_cores,
        )
    elif diagnose:
        print("Running PSIS diagnostics...")
        # _m200.compute_psis_importance_diagnostics(...)
        print("PSIS diagnostics not yet wired — use legacy m200.py directly.")
    else:
        print("No action specified for Stage 2. Use --fit or --diagnose.")

    print("Stage 2 complete.")


def merge_samples(
    ifu_file: str | Path | None = None,
    result_dir_override: str | Path | None = None,
) -> None:
    """Merge per-IFU posterior sample files into a single NetCDF."""
    from pathlib import Path as _Path
    import sys as _sys
    _old_root = _Path(__file__).resolve().parent.parent.parent / "src-orig"
    if str(_old_root) not in _sys.path:
        _sys.path.insert(0, str(_old_root))
    import m200 as _m200

    result_dir = settings.resolve_result_dir(result_dir_override)
    _m200._set_result_dir(result_dir)
    filename = settings.nfw_param_cm200_sample_filename
    _m200.merge_posterior_samples_file(filename=filename, result_dir_override=result_dir)
