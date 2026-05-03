"""MaNGA MAPS file reader — velocity maps, dispersion, spatial info.

Migrated from ``src-orig/util/maps_util.py``.  H0 constants are imported
from ``src.config.constants``.
"""

from __future__ import annotations

from warnings import catch_warnings

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning

from src.config.constants import H0_MANGA, H0_PHYS, H_RATIO


class MapsUtil:
    # Class-level constants (kept for backward-compat with self.H0_MANGA etc.)
    H0_MANGA: float = H0_MANGA
    H0_PHYS: float = H0_PHYS
    H_RATIO: float = H_RATIO

    def __init__(self, maps_file_path: str):
        self.maps_file_path = maps_file_path
        try:
            self.hdu = fits.open(self.maps_file_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"File not found: {self.maps_file_path}. Please check the path."
            )
        except Exception as e:
            raise Exception(f"Error opening FITS file: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if self.hdu:
            self.hdu.close()

    # ── Emission-line velocity ───────────────────────────────────────

    def get_eml_vel_map(self, channel_name: str = 'Ha-6564') -> tuple[np.ndarray, str, np.ndarray]:
        """Return (velocity_map_km_s, unit_str, ivar_map) for *channel_name*."""
        gas_vel_hdr = self.hdu['EMLINE_GVEL'].header
        velocity_unit = gas_vel_hdr['BUNIT']

        gas_vel_data = self.hdu['EMLINE_GVEL'].data
        gas_mask_data = self.hdu['EMLINE_GVEL_MASK'].data
        gas_ivar_data = self.hdu['EMLINE_GVEL_IVAR'].data

        channel_index = self._channel_dictionary('EMLINE_GVEL').get(channel_name)
        if channel_index is None:
            raise ValueError(f"Channel {channel_name} not found in MAPS file.")

        gas_vel_channel = gas_vel_data[channel_index, ...]
        gas_mask_channel = gas_mask_data[channel_index, ...]
        masked_velocity_map = np.where(gas_mask_channel == 0, gas_vel_channel, np.nan)

        gas_ivar_channel = gas_ivar_data[channel_index, ...]
        masked_ivar_map = np.where(gas_ivar_channel > 0, gas_ivar_channel, np.nan)

        return masked_velocity_map, velocity_unit, masked_ivar_map

    # ── Emission-line flux ───────────────────────────────────────────

    def get_eml_gflux_map(self, channel_name: str = 'Ha-6564') -> tuple[np.ndarray, str, np.ndarray]:
        """Return (flux_map, unit_str, ivar_map) for *channel_name*."""
        hdr = self.hdu['EMLINE_GFLUX'].header
        flux_unit = hdr.get('BUNIT', '')

        flux_data = self.hdu['EMLINE_GFLUX'].data
        mask_data = self.hdu['EMLINE_GFLUX_MASK'].data
        ivar_data = self.hdu['EMLINE_GFLUX_IVAR'].data

        channel_index = self._channel_dictionary('EMLINE_GFLUX').get(channel_name)
        if channel_index is None:
            raise ValueError(f"Channel {channel_name} not found in MAPS file.")

        flux_channel = flux_data[channel_index, ...]
        mask_channel = mask_data[channel_index, ...]
        ivar_channel = ivar_data[channel_index, ...]

        masked_flux_map = np.where(mask_channel == 0, flux_channel, np.nan)
        masked_ivar_map = np.where(ivar_channel > 0, ivar_channel, np.nan)

        return masked_flux_map, flux_unit, masked_ivar_map

    # ── Stellar velocity ─────────────────────────────────────────────

    def get_stellar_vel_map(self) -> tuple[np.ndarray, str, np.ndarray]:
        """Return (stellar_vel_map, unit_str, ivar_map)."""
        hdr = self.hdu['STELLAR_VEL'].header
        velocity_unit = hdr.get('BUNIT', '')

        vel_data = self.hdu['STELLAR_VEL'].data
        mask_data = self.hdu['STELLAR_VEL_MASK'].data
        ivar_data = self.hdu['STELLAR_VEL_IVAR'].data

        masked_velocity_map = np.where(mask_data == 0, vel_data, np.nan)
        masked_ivar_map = np.where(ivar_data > 0, ivar_data, np.nan)

        return masked_velocity_map, velocity_unit, masked_ivar_map

    # ── Spatial coordinates ──────────────────────────────────────────

    def get_sky_offsets(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (offset_x, offset_y) in arcsec."""
        data_skycoo = self.hdu['SPX_SKYCOO'].data
        return data_skycoo[0, ...], data_skycoo[1, ...]

    def get_snr_map(self) -> np.ndarray:
        """Return the bin SNR map."""
        return self.hdu['BIN_SNR'].data

    def get_pa(self) -> float | None:
        """Return ECOOPA position angle from primary header, or None."""
        return self.hdu['PRIMARY'].header.get('ECOOPA', None)

    def get_ba(self) -> float | None:
        """Return b/a axis ratio (1 - ECOOELL) from primary header, or None."""
        hdr = self.hdu['PRIMARY'].header
        ellip_val = hdr.get('ECOOELL', None)
        if ellip_val is not None:
            return 1 - ellip_val
        return None

    # ── PSF FWHM ─────────────────────────────────────────────────────

    def get_fwhm(self, channel_name: str = 'Ha-6564') -> float | None:
        """Return the reconstructed FWHM in arcsec for the band closest to *channel_name*."""
        hdr = self.hdu['PRIMARY'].header
        try:
            wavelength = float(channel_name.split('-')[-1])
        except (IndexError, ValueError):
            return None

        if wavelength < 5500:
            key = 'GFWHM'
        elif 5500 <= wavelength < 7000:
            key = 'RFWHM'
        elif 7000 <= wavelength < 8500:
            key = 'IFWHM'
        else:
            key = 'ZFWHM'
        return hdr.get(key, None)

    def get_pixel_scale(self) -> float | None:
        """Return pixel scale in arcsec from the PC2_2 header keyword."""
        hdr = self.hdu['EMLINE_GFLUX'].header
        pc2_2 = hdr.get('PC2_2', None)
        if pc2_2 is None:
            return None
        return abs(pc2_2) * 3600  # deg → arcsec

    # ── Radial maps ──────────────────────────────────────────────────

    def get_radius_map(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (elliptical_radius_arcsec, r_kpc, azimuth) from BIN_LWELLCOO.

        The radius *r_kpc* is converted from DAP h⁻¹ kpc to physical kpc
        using H_RATIO = H0_PHYS/H0_MANGA.
        """
        data = self.hdu['BIN_LWELLCOO'].data
        radius = data[0, ...]
        r_h_kpc_raw = data[2, ...]
        r_kpc = r_h_kpc_raw / MapsUtil.H_RATIO
        azimuth = data[3, ...]
        return radius, r_kpc, azimuth

    def get_skycoo_map(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (ra_map, dec_map) from BIN_LWSKYCOO."""
        skycoo_data = self.hdu['BIN_LWSKYCOO'].data
        return skycoo_data[0, ...], skycoo_data[1, ...]

    # ── Bin IDs ──────────────────────────────────────────────────────

    def get_stellar_binid(self) -> np.ndarray:
        """Return stellar continuum bin-ID map (channel 2 of BINID)."""
        return self.hdu['BINID'].data[1, ...]

    def get_emli_binid(self) -> np.ndarray:
        """Return emission-line bin-ID map (channel 4 of BINID, NaN for unassigned)."""
        binid_map = self.hdu['BINID'].data[3, ...]
        return np.where(binid_map >= 0, binid_map, np.nan)

    # ── Velocity dispersion ──────────────────────────────────────────

    def get_eml_gsigma_map(self, channel_name: str = 'Ha-6564') -> tuple[np.ndarray, np.ndarray]:
        """Return (sigma_obs, sigma_inst) for gas velocity dispersion."""
        sigma_data = self.hdu['EMLINE_GSIGMA'].data
        mask_data = self.hdu['EMLINE_GSIGMA_MASK'].data
        sigma_inst_data = self.hdu['EMLINE_INSTSIGMA'].data

        channel_index = self._channel_dictionary('EMLINE_GSIGMA').get(channel_name)
        if channel_index is None:
            raise ValueError(f"Channel {channel_name} not found in MAPS file.")

        sigma_channel = sigma_data[channel_index, ...]
        mask_channel = mask_data[channel_index, ...]
        sigma_inst_channel = sigma_inst_data[channel_index, ...]

        sigma_obs = np.where(mask_channel == 0, sigma_channel, np.nan)
        sigma_inst = np.where(mask_channel == 0, sigma_inst_channel, np.nan)
        return sigma_obs, sigma_inst

    def get_stellar_sigma_map(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (sigma, sigma_corr) for stellar velocity dispersion."""
        sigma_data = self.hdu['STELLAR_SIGMA'].data
        mask_data = self.hdu['STELLAR_SIGMA_MASK'].data
        sigma_corr_data = self.hdu['STELLAR_SIGMACORR'].data[0, ...]

        sigma = np.where(mask_data == 0, sigma_data, np.nan)
        sigma_corr = np.where(mask_data == 0, sigma_corr_data, np.nan)
        return sigma, sigma_corr

    # ── Channel handling ─────────────────────────────────────────────

    def _channel_dictionary(self, ext: str, prefix: str = 'C') -> dict:
        """Construct a dictionary of the channels in a MAPS file."""
        channel_dict = {}
        for k, v in self.hdu[ext].header.items():
            if k[:len(prefix)] == prefix:
                try:
                    i = int(k[len(prefix):]) - 1
                except ValueError:
                    continue
                channel_dict[v] = i
        return channel_dict

    def _channel_units(self, ext: str, prefix: str = 'U') -> np.ndarray:
        """Construct an array with the channel units."""
        cu = {}
        for k, v in self.hdu[ext].header.items():
            if k[:len(prefix)] == prefix:
                try:
                    i = int(k[len(prefix):]) - 1
                except ValueError:
                    continue
                cu[i] = v.strip()
        channel_units = np.empty(max(cu.keys()) + 1, dtype=object)
        for k, v in cu.items():
            channel_units[k] = v
        return channel_units.astype(str)
