"""Sample selection and quality-filtering logic.

Consolidated from ``src-orig/plates.py`` and ``src-orig/m200.py``.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from src.config.constants import PLATES_FILENAME
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
    """Generate *n* robustness sub-samples from the posterior pool.

    Delegates to the current population-model helper.  Additional CLI options
    will be wired in a later pass; this preserves the existing public surface.
    """
    from src.models.population import generate_robustness_sample as _generate

    print(f"Generating {n} robustness sub-samples...")
    _generate(n_sample=n, result_dir_override=result_dir_override)
    print("Done.")
