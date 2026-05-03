"""Plate-IFU catalog and target-selection utilities.

Migrated from ``src-orig/util/drpall_util.py``.  Also consolidates plate-IFU
list helpers that were duplicated in ``main.py`` and ``rc.py``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from astropy.io import fits
from astropy.table import Table

from src.config.constants import BIT_MASK_3_EXCLUDE, BIT_MASK_DRP_FAIL, TEST_PLATE_IFUS, PLATES_FILENAME  # noqa: F401

log = logging.getLogger(__name__)
if not log.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(handler)
log.setLevel(logging.INFO)
log.propagate = False


class DrpallUtil:
    # Bit masks — class-level for backward-compat with self.BIT_MASK_*
    BIT_MASK_3_EXCLUDE: int = BIT_MASK_3_EXCLUDE
    BIT_MASK_DRP_FAIL: int = BIT_MASK_DRP_FAIL

    def __init__(self, drpall_file: str):
        self.drpall_file = Path(drpall_file)
        if not self.drpall_file.exists():
            raise FileNotFoundError(f"FITS file not found: {self.drpall_file}")

    @staticmethod
    def _load_table(fits_path: Path, ext: int = 1) -> Table:
        """Load FITS table from the specified extension and return an astropy Table."""
        with fits.open(fits_path) as hdul:
            return Table(hdul[ext].data)

    @staticmethod
    def _find_colname(table: Table, candidates: Iterable[str]) -> Optional[str]:
        """Return the first matching column name from candidates or None if none found."""
        colset = set(table.colnames)
        for name in candidates:
            if name in colset:
                return name
        return None

    def _get_col_as_int(self, table: Table, candidates: Iterable[str], length_fallback: int, dtype=np.int64):
        """Return a numpy integer array for the first matching column name.

        If no column is found, return an array of zeros with length length_fallback.
        Masked values are filled with 0 before casting.
        """
        name = self._find_colname(table, candidates)
        if name is None:
            return np.zeros(length_fallback, dtype=dtype)
        col = table[name]
        try:
            arr = np.asarray(col.filled(0))
        except Exception:
            arr = np.asarray(col)
        return arr.astype(dtype, copy=False)

    def _get_col_as_str(self, table: Table, candidates: Iterable[str], length_fallback: int):
        """Return a numpy string array for the first matching column name.

        If no column is found, return an empty string array.
        """
        name = self._find_colname(table, candidates)
        if name is None:
            return np.array([""] * length_fallback, dtype=str)
        col = table[name]
        try:
            arr = np.asarray(col.filled(""))
        except Exception:
            arr = np.asarray(col)
        return arr.astype(str)

    def select_target_galaxies(self, drpall: Table) -> Table:
        """Select galaxies that are MANGATARG (mngtarg1 or mngtarg3) AND do not have excluded bits."""
        nrows = len(drpall)
        mngtarg1 = self._get_col_as_int(drpall, ['MNGTARG1', 'mngtarg1', 'MNGTARG_1'], nrows)
        mngtarg3 = self._get_col_as_int(drpall, ['MNGTARG3', 'mngtarg3', 'MNGTARG_3'], nrows)

        cond_a = (mngtarg1 != 0) | (mngtarg3 != 0)
        cond_b = (np.bitwise_and(mngtarg3, self.BIT_MASK_3_EXCLUDE) == 0)

        sel = cond_a & cond_b
        return drpall[sel]

    def select_high_quality(self, galaxies: Table) -> Table:
        """Filter out galaxies with DRP3QUAL failure bits set."""
        nrows = len(galaxies)
        drp3qual = self._get_col_as_int(galaxies, ['DRP3QUAL', 'drp3qual', 'DRP_3_QUAL'], nrows)
        sel = (np.bitwise_and(drp3qual, self.BIT_MASK_DRP_FAIL) == 0)
        return galaxies[sel]

    def unique_by_id(self, table: Table, id_candidates: Iterable[str]) -> Table:
        """Return a table with unique entries by the first existing id column.

        Keeps the first occurrence of each id (stable).
        If no id column exists, returns the input table.
        """
        id_col = self._find_colname(table, id_candidates)
        if id_col is None:
            log.warning("Warning: 'MANGAID' column not found, skipping deduplication.")
            return table

        ids = np.asarray(table[id_col])
        _, idx = np.unique(ids, return_index=True)
        idx_sorted = np.sort(idx)
        return table[idx_sorted]

    def get_all_fits(self):
        """Apply target selection, quality filter, and deduplication; return the final table."""
        drpall = self._load_table(self.drpall_file)
        galaxies = self.select_target_galaxies(drpall)
        highqual = self.select_high_quality(galaxies)
        uniquegals = self.unique_by_id(highqual, ['MANGAID', 'mangaid', 'MANGA_ID', 'MANGA_Id'])
        log.info("--- Selection completed ---")
        return uniquegals

    def _fetch_scalar_column_value(self, plateifu: str, candidates: Iterable[str]):
        """Open drpall, find row matching plateifu and return first available candidate column value or None."""
        with fits.open(self.drpall_file) as hdul:
            try:
                orig_names = list(hdul[1].columns.names)
            except Exception:
                orig_names = list(getattr(hdul[1].data, "dtype").names or [])
            lower_names = [n.lower() for n in orig_names]

            if "plateifu" not in lower_names:
                log.info("The 'plateifu' column was not found in the drpall file")
                return None
            plateifu_col = orig_names[lower_names.index("plateifu")]

            data = hdul[1].data
            match = data[plateifu_col] == plateifu
            if not np.any(match):
                log.info(f"No match found for {plateifu} in drpall")
                return None

            for cand in candidates:
                lc = cand.lower()
                if lc in lower_names:
                    colname = orig_names[lower_names.index(lc)]
                    return data[colname][match][0]
            return None

    def get_redshift(self, plateifu: str) -> float | None:
        """Return z_sys for plateifu using available columns (nsa_z, nsa_zdist, z) or None."""
        val = self._fetch_scalar_column_value(plateifu, ["nsa_z", "nsa_zdist", "z"])
        if val is not None:
            log.info(f"Using z sys value: {val}")
        return float(val) if val is not None else None

    def get_phi_ba(self, plateifu: str) -> tuple[float | None, float | None]:
        """Return (position angle in degrees, axis ratio b/a) for plateifu or (None, None)."""
        phi_val = self._fetch_scalar_column_value(plateifu, ["NSA_SERSIC_PHI", "nsa_elpetro_phi"])
        ba_val = self._fetch_scalar_column_value(plateifu, ["NSA_SERSIC_BA", "nsa_elpetro_ba"])
        phi = float(phi_val) if phi_val is not None else None
        ba = float(ba_val) if ba_val is not None else None
        return (phi, ba)

    def get_stellar_mass(self, plateifu: str) -> tuple[float | None, float | None]:
        """Return (elpetro_mass, sersic_mass) for plateifu or (None, None)."""
        sersic_mass = self._fetch_scalar_column_value(plateifu, ["NSA_SERSIC_MASS"])
        elpetro_mass = self._fetch_scalar_column_value(plateifu, ["NSA_ELPETRO_MASS"])
        return elpetro_mass, sersic_mass

    def get_effective_radius(self, plateifu: str) -> float | None:
        """Return effective radius (arcsec) for plateifu or None."""
        return self._fetch_scalar_column_value(plateifu, ["NSA_ELPETRO_TH50_R"])

    def get_sersic_n(self, plateifu: str) -> float | None:
        """Return Sersic index n for plateifu from NSA, or None if unavailable."""
        val = self._fetch_scalar_column_value(plateifu, ["NSA_SERSIC_N", "nsa_sersic_n"])
        return float(val) if val is not None else None

    # ── Galaxy-selection searches ────────────────────────────────────────

    @staticmethod
    def _ba_to_inc(ba: np.ndarray, ba_0: float = 0.2) -> np.ndarray:
        """Convert axis ratio b/a to inclination in radians."""
        ba_sq = ba**2
        BA_0_sq = ba_0**2
        numerator = ba_sq - BA_0_sq
        denominator = 1.0 - BA_0_sq
        cos_i_sq = numerator / denominator
        cos_i_sq_clipped = np.clip(cos_i_sq, 0.0, 1.0)
        inc_rad = np.arccos(np.sqrt(cos_i_sq_clipped))
        return inc_rad

    def search_plateifu_by_inc(self, inc_min: float, inc_max: float) -> tuple[np.ndarray, np.ndarray]:
        """Search galaxies with inclination between inc_min and inc_max (degrees).

        Returns (plateifu_array, inclination_array).
        """
        drpall = self._load_table(self.drpall_file)
        galaxies = self.select_target_galaxies(drpall)
        highqual = self.select_high_quality(galaxies)
        nrows = len(highqual)

        ba = self._get_col_as_int(highqual, ['NSA_SERSIC_BA', 'nsa_elpetro_ba'], nrows, dtype=np.float64)
        inc = np.degrees(self._ba_to_inc(ba))
        sel = (inc >= inc_min) & (inc <= inc_max)

        result = highqual[sel]
        uniquegals = self.unique_by_id(result, ['MANGAID', 'mangaid', 'MANGA_ID', 'MANGA_Id'])
        plateifu_list = self._get_col_as_str(uniquegals, ['PLATEIFU', 'plateifu', 'PLATE_IFU', 'plate_ifu'], len(uniquegals))
        return plateifu_list, inc[sel]

    def search_plateifu_by_stellar_mass(self, mass_min: float, mass_max: float) -> tuple[np.ndarray, np.ndarray]:
        """Search by stellar mass range. Returns (plateifu_array, mass_array)."""
        drpall = self._load_table(self.drpall_file)
        galaxies = self.select_target_galaxies(drpall)
        highqual = self.select_high_quality(galaxies)
        nrows = len(highqual)

        mass = self._get_col_as_int(highqual, ['NSA_ELPETRO_MASS', 'nsa_sersic_mass'], nrows, dtype=np.float64)
        sel = (mass >= mass_min) & (mass <= mass_max)
        result = highqual[sel]
        uniquegals = self.unique_by_id(result, ['MANGAID', 'mangaid', 'MANGA_ID', 'MANGA_Id'])
        plateifu_list = self._get_col_as_str(uniquegals, ['PLATEIFU', 'plateifu', 'PLATE_IFU', 'plate_ifu'], len(uniquegals))
        return plateifu_list, mass[sel]

    def search_plateifu_by_effective_radius(self, r_eff_min: float, r_eff_max: float) -> tuple[np.ndarray, np.ndarray]:
        """Search by effective radius range. Returns (plateifu_array, reff_array)."""
        drall = self._load_table(self.drpall_file)
        galaxies = self.select_target_galaxies(drall)
        highqual = self.select_high_quality(galaxies)
        nrows = len(highqual)

        reff = self._get_col_as_int(highqual, ['NSA_ELPETRO_TH50_R', 'nsa_elpetro_th50_r'], nrows, dtype=np.float64)
        sel = (reff >= r_eff_min) & (reff <= r_eff_max)
        result = highqual[sel]
        uniquegals = self.unique_by_id(result, ['MANGAID', 'mangaid', 'MANGA_ID', 'MANGA_Id'])
        plateifu_list = self._get_col_as_str(uniquegals, ['PLATEIFU', 'plateifu', 'PLATE_IFU', 'plate_ifu'], len(uniquegals))
        return plateifu_list, reff[sel]

    def search_plateifu_by_sersic_n(self, sersic_n_min: float, sersic_n_max: float) -> tuple[np.ndarray, np.ndarray]:
        """Search by Sersic-n range. Returns (plateifu_array, sersic_n_array)."""
        drall = self._load_table(self.drpall_file)
        galaxies = self.select_target_galaxies(drall)
        highqual = self.select_high_quality(galaxies)
        nrows = len(highqual)

        sersic_n = self._get_col_as_int(highqual, ['NSA_SERSIC_N', 'nsa_sersic_n'], nrows, dtype=np.float64)
        sel = (sersic_n >= sersic_n_min) & (sersic_n <= sersic_n_max)
        result = highqual[sel]
        uniquegals = self.unique_by_id(result, ['MANGAID', 'mangaid', 'MANGA_ID', 'MANGA_Id'])
        plateifu_list = self._get_col_as_str(uniquegals, ['PLATEIFU', 'plateifu', 'PLATE_IFU', 'plate_ifu'], len(uniquegals))
        return plateifu_list, sersic_n[sel]


# ── Plate-IFU list utilities (consolidated from main.py and rc.py) ────────

def get_plateifu_list(filepath: str | None = None, test: bool = False) -> list[str]:
    """Return a list of plate-IFU strings.

    If *test* is True, return the hardcoded TEST_PLATE_IFUS.
    Otherwise, read from *filepath* (defaulting to ``PLATES_FILENAME``).
    """
    if test:
        return list(TEST_PLATE_IFUS)
    path = Path(filepath) if filepath else Path(PLATES_FILENAME)
    if not path.exists():
        return list(TEST_PLATE_IFUS)
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]
