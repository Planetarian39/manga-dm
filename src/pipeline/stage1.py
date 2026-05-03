"""Stage 1 pipeline: single-galaxy rotation-curve and NFW DM fitting.

Orchestration logic extracted from ``src-orig/main.py``.
"""

from __future__ import annotations

import gc
import multiprocessing
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.config.constants import TEST_PLATE_IFUS, PLATES_FILENAME
from src.config.settings import settings
from src.data.catalog import get_plateifu_list
from src.data.results import get_params_file, get_processed_plate_ifus, store_params_file


def _is_plate_ifu_id(value: str) -> bool:
    """Return True if *value* matches ``PPPP-MMMM`` format."""
    parts = value.strip().split("-", 1)
    return len(parts) == 2 and all(p.isdigit() for p in parts)


def process_plate_ifu(
    plate_ifu: str,
    process_nfw: bool = True,
    debug: bool = False,
    result_dir_override: str | Path | None = None,
    *,
    fits_util=None,
) -> None:
    """Run Stage 1 for a single plate-IFU: RC fit + optional NFW DM fit.

    Currently delegates to the legacy ``src-orig/main.py`` implementation.
    When the old scripts are retired, this will host the full pipeline logic.
    """
    # Import legacy implementation lazily to avoid hard dependency
    from pathlib import Path as _Path
    import sys as _sys
    _old_root = _Path(__file__).resolve().parent.parent.parent / "src-orig"
    if str(_old_root) not in _sys.path:
        _sys.path.insert(0, str(_old_root))
    import main as _main

    result_dir = settings.resolve_result_dir(result_dir_override)
    result_dir.mkdir(parents=True, exist_ok=True)

    # Configure legacy globals
    _main._set_result_dir(result_dir)
    return _main.process_plate_ifu(plate_ifu, process_nfw=process_nfw, debug=debug)


def process_plate_ifu_worker(
    plate_ifu: str,
    run_nfw: bool,
    debug: bool,
    result_dir_override: str | None = None,
    r0_frac: float | None = None,
    m200_prior_dex: float | None = None,
    inc_prior_enable: bool | None = None,
) -> None:
    """Multiprocessing worker wrapper."""
    import sys as _sys
    from pathlib import Path as _Path
    _old_root = _Path(__file__).resolve().parent.parent.parent / "src-orig"
    if str(_old_root) not in _sys.path:
        _sys.path.insert(0, str(_old_root))
    import main as _main

    result_dir = settings.resolve_result_dir(result_dir_override)
    result_dir.mkdir(parents=True, exist_ok=True)

    _main._set_result_dir(result_dir)
    _main._set_r0_frac(r0_frac)
    _main._set_m200_prior_dex(m200_prior_dex)
    _main._set_inc_prior_enable(inc_prior_enable)
    _main.process_plate_ifu(plate_ifu, process_nfw=run_nfw, debug=debug)


def run_stage1(
    ifu: str = "test",
    nfw: bool = False,
    n_cores: int | None = None,
    result_dir_override: str | Path | None = None,
    debug: bool = False,
    r0_frac: float | None = None,
    m200_prior_dex: float | None = None,
    inc_prior_enable: bool | None = None,
) -> None:
    """Orchestrate Stage 1 processing.

    Parameters
    ----------
    ifu : str
        ``"test"`` for 8 test galaxies, ``"all"`` for all in plateifu list,
        or a specific plate-ifu string.
    nfw : bool
        Whether to also run NFW DM fitting.
    n_cores : int or None
        Number of parallel workers (default: ``settings.n_cores`` or 1).
    """
    # Determine plate-IFU list
    if ifu.lower() == "test":
        plate_ifus = list(TEST_PLATE_IFUS)
    elif _is_plate_ifu_id(ifu):
        plate_ifus = [ifu]
    elif ifu.lower() == "all":
        plate_ifus = get_plateifu_list(filepath=PLATES_FILENAME)
        if not plate_ifus:
            print("No plate-IFUs found. Use 'manga select --download' first.")
            return
    else:
        plate_ifus = [ifu]

    result_dir = settings.resolve_result_dir(result_dir_override)
    result_dir.mkdir(parents=True, exist_ok=True)

    if n_cores is None:
        n_cores = getattr(settings, "n_cores", None) or 1

    # Filter already-processed
    processed = get_processed_plate_ifus(settings.rc_param_filename, result_dir)
    todo = [p for p in plate_ifus if p not in processed]
    if not todo:
        print("All plate-IFUs already processed.")
        return

    print(f"Processing {len(todo)} plate-IFUs with {n_cores} workers...")

    if n_cores > 1:
        with multiprocessing.Pool(processes=n_cores) as pool:
            args = [
                (p, nfw, debug, result_dir_override, r0_frac, m200_prior_dex, inc_prior_enable)
                for p in todo
            ]
            list(tqdm(
                pool.starmap(process_plate_ifu_worker, args),
                total=len(todo),
                desc="Stage 1",
            ))
    else:
        for plate_ifu in tqdm(todo, desc="Stage 1"):
            process_plate_ifu(plate_ifu, process_nfw=nfw, debug=debug, result_dir_override=result_dir_override)

    gc.collect()
    print("Stage 1 complete.")
