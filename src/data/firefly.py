"""Firefly stellar-mass and spatial data access.

Migrated from ``src-orig/util/firefly_util.py``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
from astropy.io import fits
from astropy.table import Table

log = logging.getLogger(__name__)
if not log.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(handler)
log.setLevel(logging.INFO)
log.propagate = False


class FireflyUtil:
    firefly_file: Path
    hdu: fits.HDUList

    def __init__(self, firefly_file: str):
        self.firefly_file = Path(firefly_file)
        if not self.firefly_file.exists():
            raise FileNotFoundError(f"FITS file not found: {self.firefly_file}")

        try:
            self.hdu = fits.open(self.firefly_file)
        except Exception as e:
            raise Exception(f"Error opening FITS file: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if self.hdu:
            self.hdu.close()

    # ── Row lookup ───────────────────────────────────────────────────

    def _find_row_index(self, plateifu: str) -> int:
        """Return the integer row index that matches *plateifu*."""
        table = Table(self.hdu[1].data)
        matches = np.where(table['PLATEIFU'] == plateifu)[0]
        if matches.size == 0:
            raise ValueError(f"plateifu not found: {plateifu}")
        if matches.size > 1:
            log.warning(
                "Multiple matches for plateifu %s; using the first match", plateifu
            )
        return int(matches[0])

    # ── Data accessors ───────────────────────────────────────────────

    def get_stellar_density_cell(self, plateifu: str) -> tuple[np.ndarray, np.ndarray]:
        """Return (linear_density_Msun_kpc2, linear_density_err) for *plateifu*.

        HDU 13: surface mass density (log M⊙/kpc²) → linear M⊙/kpc².
        """
        hdu_index = 13
        row_idx = self._find_row_index(plateifu)
        data = self.hdu[hdu_index].data  # (10735, 2800, 2)
        if not (0 <= row_idx < data.shape[0]):
            raise ValueError(
                f"Row index {row_idx} out of bounds for HDU{hdu_index} with shape {data.shape}"
            )
        data_row = data[row_idx, :, :]  # (2800, 2)
        density = data_row[:, 0]  # log(M⊙/kpc²)
        density_err = data_row[:, 1]
        linear_density = 10**density
        linear_density_err = linear_density * np.log(10) * density_err
        return linear_density, linear_density_err

    def get_stellar_mass_cell(self, plateifu: str) -> tuple[np.ndarray, np.ndarray]:
        """Return (linear_mass_Msun, linear_mass_err) for *plateifu*.

        HDU 11 channels 2–3: total stellar mass of Voronoi cell in log(M⊙).
        """
        hdu_index = 11
        row_idx = self._find_row_index(plateifu)
        data = self.hdu[hdu_index].data  # (10735, 2800, 4)
        if not (0 <= row_idx < data.shape[0]):
            raise ValueError(
                f"Row index {row_idx} out of bounds for HDU{hdu_index} with shape {data.shape}"
            )
        data_row = data[row_idx, :, :]  # (2800, 4)
        mass = data_row[:, 2]  # log(M⊙)
        mass_err = data_row[:, 3]
        linear_mass = 10**mass
        linear_mass_err = linear_mass * np.log(10) * mass_err
        return linear_mass, linear_mass_err

    def get_spatial_info(self, plateifu: str) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        """Return (bin_number, x_pos, y_pos, radius_eff, azimuth) for *plateifu*.

        HDU 4: spatial information for each Voronoi cell.
        """
        hdu_index = 4
        row_idx = self._find_row_index(plateifu)
        data = self.hdu[hdu_index].data  # (10735, 2800, 5)
        if not (0 <= row_idx < data.shape[0]):
            raise ValueError(
                f"Row index {row_idx} out of bounds for HDU{hdu_index} with shape {data.shape}"
            )
        data_row = data[row_idx, :, :]  # (2800, 5)
        return (
            data_row[:, 0],  # bin_number
            data_row[:, 1],  # x_pos
            data_row[:, 2],  # y_pos
            data_row[:, 3],  # radius_eff
            data_row[:, 4],  # azimuth
        )

    def get_binid(self, plateifu: str) -> np.ndarray:
        """Return the bin ID map for *plateifu*."""
        binid, _, _, _, _ = self.get_spatial_info(plateifu)
        return binid

    def get_radius_eff(self, plateifu: str) -> tuple[np.ndarray, np.ndarray]:
        """Return (radius_eff, azimuth) for *plateifu*."""
        _, _, _, radius_eff, azimuth = self.get_spatial_info(plateifu)
        return radius_eff, azimuth
