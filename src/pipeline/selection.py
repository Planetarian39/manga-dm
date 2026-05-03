"""Sample selection and quality-filtering logic.

Consolidated from ``src-orig/plates.py`` and ``src-orig/m200.py``.
"""

from __future__ import annotations

from pathlib import Path

from src.config.constants import TEST_PLATE_IFUS, PLATES_FILENAME
from src.config.settings import settings


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

    # Use legacy implementation via sys.path hook
    from pathlib import Path as _Path
    import sys as _sys
    _old_root = _Path(__file__).resolve().parent.parent.parent / "src-orig"
    if str(_old_root) not in _sys.path:
        _sys.path.insert(0, str(_old_root))

    import plates as _plates
    return _plates.main()


def generate_robustness_sample(
    n: int = 10,
    result_dir_override: str | Path | None = None,
) -> None:
    """Generate *n* robustness sub-samples from the posterior pool.

    Delegates to the legacy ``m200.py`` implementation.
    """
    from pathlib import Path as _Path
    import sys as _sys
    _old_root = _Path(__file__).resolve().parent.parent.parent / "src-orig"
    if str(_old_root) not in _sys.path:
        _sys.path.insert(0, str(_old_root))
    import m200 as _m200

    result_dir = settings.resolve_result_dir(result_dir_override)
    _m200._set_result_dir(result_dir)

    print(f"Generating {n} robustness sub-samples...")
    _m200.generate_robustness_sample()
    print("Done.")
