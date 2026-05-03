"""Rotation-curve fitting model.

Migrated from ``src-orig/rc.py``.  The ``RotCurve`` class wraps a full
single-galaxy rotation-curve inference pipeline: data loading, quality
filtering, MCMC fitting (PyMC), posterior diagnostics, and summary
output.

The PyMC model code in ``_inf_vel_rot`` is **preserved verbatim**.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning
import astropy.constants as const
from lmfit import Model

import pymc as pm
import pytensor.tensor as pt

# New layered imports
from src.config.constants import (
    COLOR_MODEL, COLOR_DATA_POINTS, COLOR_BOOTSTRAP_LINES,
    H0_MANGA, H0_PHYS, H_RATIO, M_PIVOT_H_INV, ARCSEC_PER_RADIAN,
    MOSTER_M1, MOSTER_N, MOSTER_BETA, MOSTER_GAMMA,
    STRUCT_RD_FACTOR, STRUCT_HERNQUIST_FACTOR,
    VC_BOUNDS, S_OUT_BOUNDS, VSYS_BOUNDS,
    BIT_MASK_3_EXCLUDE, BIT_MASK_DRP_FAIL,
    TEST_PLATE_IFUS, PLATES_FILENAME, VEL_ROT_PARAM_FILE,
)
from src.config.settings import settings
from src.data.catalog import DrpallUtil
from src.data.fits import FitsUtil
from src.data.maps import MapsUtil
from src.data.firefly import FireflyUtil
from src.stats.arviz_compat import (
    ensure_arviz_compat, get_arviz_api,
    get_summary_interval_columns, summary_with_compat,
    set_arviz_ci_defaults, get_posterior_dataset,
)
from src.stats.intervals import calc_interval_overlap_mask

# ArviZ compat
az = ensure_arviz_compat()

# ── Module-level config shims (names preserved for backward-compat) ──
# The RotCurve class references these as bare names; we re-export them
# from the settings singleton and constants module.
RADIUS_MIN_KPC = settings.RADIUS_MIN_KPC
SNR_THRESHOLD = settings.SNR_THRESHOLD
PHI_DEG_THRESHOLD = settings.PHI_DEG_THRESHOLD
IVAR_RATIO_THRESHOLD = settings.IVAR_RATIO_THRESHOLD
GSIGMA_MAX = settings.GSIGMA_MAX
USE_GSIGMA_INST_CORR = settings.USE_GSIGMA_INST_CORR
INC_MIN = settings.INC_MIN
INC_MAX = settings.INC_MAX
VEL_OBS_COUNT_THRESHOLD = settings.VEL_OBS_COUNT_THRESHOLD
RMAX_RT_FACTOR = settings.RMAX_RT_FACTOR
INFER_RHAT_THRESHOLD = settings.INFER_RHAT_THRESHOLD
INFER_ESS_THRESHOLD = settings.INFER_ESS_THRESHOLD
HDI_PROB2 = settings.HDI_PROB2
PPC_HDI_VALUE_COVERAGE_THRESHOLD = settings.PPC_HDI_VALUE_COVERAGE_THRESHOLD
PPC_HDI_OVERLAP_THRESHOLD = settings.PPC_HDI_OVERLAP_THRESHOLD
PPC_MEAS_SIGMA_SCALE = settings.PPC_MEAS_SIGMA_SCALE
BA_0 = settings.BA_0
VEL_SYSTEM_ERROR = settings.VEL_SYSTEM_ERROR
class RotCurve:
    drpall_util = None
    firefly_util = None
    maps_util = None
    plot_util = None
    fit_debug = False
    fit_model = "mcmc"
    n_samples = 0
    pri_inc = True
    pri_phi_delta = True

    def __init__(self, drpall_util: DrpallUtil, firefly_util: FireflyUtil, maps_util: MapsUtil, plot_util: PlotUtil=None) -> None:
        self.drpall_util = drpall_util
        self.firefly_util = firefly_util
        self.maps_util = maps_util
        self.plot_util = plot_util

    ################################################################################
    # calculate functions
    ################################################################################

    # Calculate the galaxy inclination i (in radians)
    # Formula for inclination i
    # The inclination is the angle between the galaxy disk normal and the observer's line of sight.
    # ba: The axis ratio (b/a) of the galaxy, where 'b' is the length of the minor axis and 'a' is the length of the major axis.
    @staticmethod
    def _calc_inc(ba, ba_0=0.2):
        ba_sq = ba**2
        BA_0_sq = ba_0**2

        # Compute the numerator part of cos^2(i)
        numerator = ba_sq - BA_0_sq
        denominator = 1.0 - BA_0_sq

        cos_i_sq = numerator / denominator
        cos_i_sq_clipped = np.clip(cos_i_sq, 0.0, 1.0)

        inc_rad = np.arccos(np.sqrt(cos_i_sq_clipped))
        return inc_rad


    @staticmethod
    def _calc_gsigma_astrophysical(gsigma_map: np.ndarray, gsigma_inst_map: np.ndarray | None=None) -> np.ndarray:
        gsigma_map = np.asarray(gsigma_map, dtype=float)
        if gsigma_inst_map is None:
            return np.where(np.isfinite(gsigma_map) & (gsigma_map > 0), gsigma_map, np.nan)

        gsigma_inst_map = np.asarray(gsigma_inst_map, dtype=float)
        sigma_sq = gsigma_map**2 - gsigma_inst_map**2
        sigma_sq = np.where(np.isfinite(sigma_sq) & (sigma_sq > 0), sigma_sq, np.nan)
        return np.sqrt(sigma_sq)

    def _build_vel_quality_mask(
        self,
        vel_map: np.ndarray,
        snr_map: np.ndarray,
        phi_map: np.ndarray,
        ivar_map: np.ndarray,
        gsigma_map: np.ndarray | None=None,
        gsigma_inst_map: np.ndarray | None=None,
    ) -> np.ndarray:
        phi_limit_rad = np.radians(PHI_DEG_THRESHOLD)
        phi_delta = None if phi_map is None else (phi_map + np.pi/2) % np.pi - np.pi/2

        base_mask = np.isfinite(vel_map)
        base_mask &= np.isfinite(snr_map) & (snr_map >= SNR_THRESHOLD) if snr_map is not None else True
        base_mask &= np.isfinite(phi_map) & (np.abs(phi_delta) <= phi_limit_rad) if phi_map is not None else True
        base_mask &= np.isfinite(ivar_map) & (ivar_map > 0) if ivar_map is not None else True

        if gsigma_map is not None:
            gsigma_eff = self._calc_gsigma_astrophysical(
                gsigma_map,
                gsigma_inst_map if USE_GSIGMA_INST_CORR else None,
            )
            gsigma_mask = np.isfinite(gsigma_eff)
            if GSIGMA_MAX and GSIGMA_MAX > 0:
                gsigma_mask &= (gsigma_eff <= GSIGMA_MAX)
            base_mask &= gsigma_mask

        if IVAR_RATIO_THRESHOLD is None or IVAR_RATIO_THRESHOLD <= 0:
            return base_mask

        ivar_valid = ivar_map[base_mask]
        if ivar_valid.size == 0:
            return base_mask

        ivar_limit = float(np.nanpercentile(ivar_valid, 100 * IVAR_RATIO_THRESHOLD))
        return base_mask & (ivar_map >= ivar_limit)

    # Filter the velocity map with SNR above the threshold and within ±phi_limit of the major axis.
    def _vel_map_filter(
        self,
        vel_map: np.ndarray,
        snr_map: np.ndarray,
        phi_map: np.ndarray,
        ivar_map: np.ndarray,
        gsigma_map: np.ndarray | None=None,
        gsigma_inst_map: np.ndarray | None=None,
    ) -> np.ndarray:
        valid_mask = self._build_vel_quality_mask(
            vel_map,
            snr_map,
            phi_map,
            ivar_map,
            gsigma_map=gsigma_map,
            gsigma_inst_map=gsigma_inst_map,
        )
        vel_map_filtered = np.full_like(vel_map, np.nan, dtype=float)
        vel_map_filtered[valid_mask] = vel_map[valid_mask]
        return vel_map_filtered

    # PA: The position angle of the major axis of the galaxy, measured from north to east.
    # b/a: The axis ratio (b/a) of the galaxy
    def _calc_pa_inc(self) -> float:
        phi = self.maps_util.get_pa()
        ba = self.maps_util.get_ba()
        # print(f"Position Angle PA from MAPS header: {phi:.2f} deg,", f"Inclination b/a from MAPS header: {ba:.3f}")

        inc = self._calc_inc(ba, ba_0=BA_0)
        # print(f"Calculated Inclination i: {np.degrees(inc):.2f} deg")
        # Convert PA from degrees to radians and rotate so North is at +90°
        pa = np.mod(np.radians(phi), 2 * np.pi)
        return pa, inc

    def _get_vel_obs_raw(self, type: str='gas', is_filter: bool=True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        offset_x, offset_y = self.maps_util.get_sky_offsets()
        # print(f"Sky offsets shape: {offset_x.shape}, X offset: [{np.nanmin(offset_x):.3f}, {np.nanmax(offset_x):.3f}] arcsec")

        # R: radial distance map
        radius_map, radius_h_kpc_map, azimuth_map = self.maps_util.get_radius_map()
        # print(f"r_map: [{np.nanmin(radius_map):.3f}, {np.nanmax(radius_map):.3f}] spaxel,", f"shape: {radius_map.shape}")
        # print(f"r_h_kpc_map: [{np.nanmin(radius_h_kpc_map):.3f}, {np.nanmax(radius_h_kpc_map):.3f}] kpc,", f"shape: {radius_h_kpc_map.shape}")
        # print(f"azimuth_map: [{np.nanmin(azimuth_map):.3f}, {np.nanmax(azimuth_map):.3f}] deg,", f"shape: {azimuth_map.shape}")

        # SNR: signal-to-noise ratio map
        snr_map = self.maps_util.get_snr_map()
        # print(f"SNR map shape: {snr_map.shape}, SNR range: [{np.nanmin(snr_map):.3f}, {np.nanmax(snr_map):.3f}]")

        ra_map, dec_map = self.maps_util.get_skycoo_map()
        # print(f"RA map: [{np.nanmin(ra_map):.6f}, {np.nanmax(ra_map):.6f}] deg,", f"Dec map: [{np.nanmin(dec_map):.6f}, {np.nanmax(dec_map):.6f}] deg")

        ## Get the gas velocity map (H-alpha)
        v_obs_gas_map, _gv_unit, _gv_ivar = self.maps_util.get_eml_vel_map()
        # print(f"Gas velocity map shape: {v_obs_gas_map.shape}, Unit: {_gv_unit}, Velocity range: [{np.nanmin(v_obs_gas_map):.3f}, {np.nanmax(v_obs_gas_map):.3f}] {_gv_unit}, size: {np.sum(np.isfinite(v_obs_gas_map))}")
        eml_binid = self.maps_util.get_emli_binid()
        # print(f"Gas Unique indices shape: {eml_binid.shape}, range: [{np.nanmin(eml_binid):.0f}, {np.nanmax(eml_binid):.0f}], size: {len(np.unique(eml_binid))}")

        gsigma_map, gsigma_inst_map = self.maps_util.get_eml_gsigma_map()
        print(f"Gas sigma map shape: {gsigma_map.shape}, range: [{np.nanmin(gsigma_map):.3f}, {np.nanmax(gsigma_map):.3f}] km/s, mean Gas sigma: {np.nanmean(gsigma_map):.3f} km/s")


        ## Get the stellar velocity map
        v_obs_stellar_map, _sv_unit, _sv_ivar = self.maps_util.get_stellar_vel_map()
        stellar_binid = self.maps_util.get_stellar_binid()

        # Velocity correction
        if type == 'gas':
            v_obs_map = v_obs_gas_map
            v_unit = _gv_unit
            v_ivar = _gv_ivar
        else:
            v_obs_map = v_obs_stellar_map
            v_unit = _sv_unit
            v_ivar = _sv_ivar

        azimuth_rad_map = np.radians(azimuth_map)

        # Filter velocity map
        if not is_filter:
            filtered_vel_map = v_obs_map
        else:
            filtered_vel_map = self._vel_map_filter(v_obs_map, snr_map, azimuth_rad_map, v_ivar, gsigma_map, gsigma_inst_map)

        print(f"vel_obs data count: {np.sum(np.isfinite(v_obs_map))}, after filter: {np.sum(np.isfinite(filtered_vel_map))}  ({100.0 * np.sum(np.isfinite(filtered_vel_map)) / np.sum(np.isfinite(v_obs_map)):.2f}%)")

        mask = np.isfinite(filtered_vel_map)
        r_obs_map = np.where(mask, radius_h_kpc_map, np.nan)
        v_obs_map = np.where(mask, v_obs_map, np.nan)
        ivar_obs_map = np.where(mask, v_ivar, np.nan)
        phi_obs_map = np.where(mask, azimuth_rad_map, np.nan)

        return r_obs_map, v_obs_map, ivar_obs_map, phi_obs_map

    def _get_radius(self) -> np.ndarray:
        _, radius_h_kpc_map, _ = self.maps_util.get_radius_map()
        return radius_h_kpc_map


    # Inclination Angle: The angle between the galaxy's disk and the plane of the sky.
    # Azimuthal Angle: The angle of the dataset within the galaxy's disk relative to the kinematic major axis (i.e., the line where the line-of-sight velocity is zero).

    # Formula: V_obs = Vsys + V_rot * (sin(i) * cos(phi - phi_0))
    # Warning: The sign of the calculated velocity may be different from the observed velocity.
    def _vel_obs_project_profile(self, vel_rot: np.ndarray, vel_sys: float, inc: float, phi_map: np.ndarray) -> np.ndarray:
        phi_delta = (phi_map + np.pi) % (2 * np.pi)  # phi_map is (phi - phi_0)

        correction = np.sin(inc) * np.cos(phi_delta)
        vel_obs = vel_sys + vel_rot * correction
        return vel_obs

    # formula: V_rot = (V_obs - Vsys) / (sin(i) * cos(phi - phi_0))
    def _vel_rot_disproject_profile(self, vel_obs: np.ndarray, vel_sys: float, inc: float, phi_map: np.ndarray) -> np.ndarray:
        phi_delta = (phi_map + np.pi) % (2 * np.pi)  # phi_map is (phi - phi_0)

        correction = np.sin(inc) * np.cos(phi_delta)
        correction = np.where(np.abs(correction) < 1e-3, np.nan, correction)
        vel_disproject = (vel_obs - vel_sys) / correction

        # set the sign of vel_disproject according to the phi_map quadrants
        vel_disproject = np.where((phi_delta >= 0) & (phi_delta < np.pi/2), np.abs(vel_disproject), vel_disproject)
        vel_disproject = np.where((phi_delta >= np.pi/2) & (phi_delta < np.pi), -np.abs(vel_disproject), vel_disproject)
        vel_disproject = np.where((phi_delta >= np.pi) & (phi_delta < 3*np.pi/2), -np.abs(vel_disproject), vel_disproject)
        vel_disproject = np.where((phi_delta >= 3*np.pi/2) & (phi_delta < 2*np.pi), np.abs(vel_disproject), vel_disproject)

        return vel_disproject


    ################################################################################
    # profile
    ################################################################################

    # Formula: V(r) = Vc * tanh(r / Rt) + s_out * r
    # Vc: Vc is the asymptotic circular velocity at large radii
    # Rt: Rt is the turnover radius where the hyperbolic tangent term begins to be flat
    # s_out: sout is the slope of the RC at large radii r >> Rt
    # Negativity: The s_out parameter may have bad standard errors.
    def _vel_rot_tan_sout_profile(self, r: np.ndarray, Vc: float, Rt: float, s_out: float) -> np.ndarray:
        return Vc * np.tanh(r / Rt) + s_out * r

    # Formula: V(r) = Vc * tanh(r / Rt) * (1 + beta * r / Rmax)
    # def _vel_rot_tan_beta_profile(self, r: np.ndarray, Vc: float, Rt: float, beta: float, Rmax: float) -> np.ndarray:
    #     return Vc * np.tanh(r / Rt) * (1 + beta * r / Rmax)

    # Universal Rotation Curve (URC)
    # Positivity: stable model with good Standard Errors of the parameters
    # Negativity: the reduced Chi-Squared is not good enough
    # Formula: V(r) = V0 + (2/pi) * Vc * arctan(r / Rt)
    def _vel_rot_arctan_profile(self, r: np.ndarray, V0: float, Vc: float, Rt: float) -> np.ndarray:
        return V0 + (2 / np.pi) * Vc * np.arctan(r / Rt)

    # Formula: V(r) = V0 * (1 - e^(-r / Rt)) (1 + alpha * r / Rt)
    # Negativity: The alpha parameter parameter may have bad standard errors.
    def _vel_rot_polyex_profile(self, r: np.ndarray, V0: float, Rt: float, alpha: float) -> np.ndarray:
        return V0 * (1 - np.exp(-r / Rt)) * (1 + alpha * r / Rt)


    ################################################################################
    # Error functions
    ################################################################################
    @staticmethod
    def _calc_interval_overlap_mask(values: np.ndarray, sigma: np.ndarray, lower: np.ndarray, upper: np.ndarray, sigma_scale: float=1.0) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        sigma = np.asarray(sigma, dtype=float)
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)

        value_lower = values - sigma_scale * sigma
        value_upper = values + sigma_scale * sigma
        valid = np.isfinite(value_lower) & np.isfinite(value_upper) & np.isfinite(lower) & np.isfinite(upper)
        overlap_mask = valid & (np.maximum(value_lower, lower) <= np.minimum(value_upper, upper))
        return overlap_mask

    @staticmethod
    def evaluate_fit_quality(fit_params: dict, data_count: int) -> dict:
        inc_rad = float(fit_params.get('inc', np.nan))
        inc_deg = float(np.degrees(inc_rad)) if np.isfinite(inc_rad) else np.nan
        r_max = float(fit_params.get('Rmax', np.nan))
        rt = float(fit_params.get('Rt', np.nan))
        rmax_rt_ratio = float(r_max / rt) if np.isfinite(r_max) and np.isfinite(rt) and rt > 0 else np.nan

        ppc_value_coverage = float(fit_params.get('PPC_HDI_VALUE_COVERAGE', 'nan'))
        ppc_overlap = float(fit_params.get('PPC_HDI_OVERLAP', 'nan'))
        has_ppc_metrics = (
            ('PPC_HDI_PROB' in fit_params) or
            np.isfinite(ppc_value_coverage) or
            np.isfinite(ppc_overlap)
        )

        checks = {
            'enough_data': int(data_count) >= VEL_OBS_COUNT_THRESHOLD,
            'valid_geometry': np.isfinite(inc_deg) and (INC_MIN <= inc_deg <= INC_MAX),
            'valid_extent': np.isfinite(rmax_rt_ratio) and (rmax_rt_ratio >= RMAX_RT_FACTOR),
            'valid_ppc_value_coverage': (not has_ppc_metrics) or (
                np.isfinite(ppc_value_coverage) and (ppc_value_coverage >= PPC_HDI_VALUE_COVERAGE_THRESHOLD)
            ),
            'valid_ppc_overlap': (not has_ppc_metrics) or (
                np.isfinite(ppc_overlap) and (ppc_overlap >= PPC_HDI_OVERLAP_THRESHOLD)
            ),
        }

        fail_reasons = []
        if not checks['enough_data']:
            fail_reasons.append(f"low_data_count:{int(data_count)}/{VEL_OBS_COUNT_THRESHOLD}")
        if not checks['valid_geometry']:
            fail_reasons.append(f"inclination_out_of_range:{inc_deg:.2f}deg")
        if not checks['valid_extent']:
            fail_reasons.append(f"insufficient_extent:Rmax_Rt={rmax_rt_ratio:.2f}")
        if not checks['valid_ppc_value_coverage']:
            fail_reasons.append(f"low_ppc_value_coverage:{ppc_value_coverage:.3f}")
        if not checks['valid_ppc_overlap']:
            fail_reasons.append(f"low_ppc_overlap:{ppc_overlap:.3f}")

        passed = len(fail_reasons) == 0
        summary = (
            f"COUNT={int(data_count)}/{VEL_OBS_COUNT_THRESHOLD}, "
            f"inc={inc_deg:.2f} deg [{INC_MIN:.1f}, {INC_MAX:.1f}], "
            f"Rmax/Rt={rmax_rt_ratio:.2f} ({RMAX_RT_FACTOR:.2f}), "
            f"PPC_VALUE={ppc_value_coverage:.3f} ({PPC_HDI_VALUE_COVERAGE_THRESHOLD:.3f}), "
            f"PPC_OVERLAP={ppc_overlap:.3f} ({PPC_HDI_OVERLAP_THRESHOLD:.3f})"
        )

        return {
            'passed': passed,
            'summary': summary,
            'fail_reasons': fail_reasons,
            'data_count': int(data_count),
            'inc_deg': inc_deg,
            'rmax_rt_ratio': rmax_rt_ratio,
            'ppc_value_coverage': ppc_value_coverage,
            'ppc_overlap': ppc_overlap,
            'has_ppc_metrics': has_ppc_metrics,
            **checks,
        }

    ################################################################################
    # Fitting methods
    ################################################################################
    # use tanh profile to fit vel_rot
    def _fit_vel_rot(self, vel_param: dict, radius_fit: np.ndarray=None) -> tuple[bool, dict, dict]:
        radius_map = vel_param["radius_obs"]
        vel_obs_map = vel_param["vel_obs"]
        ivar_map = vel_param["ivar_obs"]
        phi_map = vel_param["phi_map"]

        valid_mask = np.isfinite(vel_obs_map) & np.isfinite(radius_map) & (radius_map > RADIUS_MIN_KPC)
        radius_valid = radius_map[valid_mask]
        vel_obs_valid = vel_obs_map[valid_mask]
        ivar_map_valid = ivar_map[valid_mask]
        phi_map_valid = phi_map[valid_mask]

        inc_act = self.get_inc_rad()
        R_max = np.nanmax(radius_valid)
        SIGMA_OBS_BAR = np.sqrt(1.0 / np.nanmean(ivar_map_valid))

        ######################################
        # normal all fit parameters
        ######################################
        params_range = {
            'Vc': (20.0, 500.0),  # km/s
            'Rt': (R_max * 0.01, R_max * 1.0),  # kpc
            's_out': (-10.0, 10.0),  # km/s/kpc
            'Vsys': (-100.0, 100.0),  # km/s
            'inc': (np.deg2rad(np.rad2deg(inc_act)-10), np.deg2rad(np.rad2deg(inc_act)+10)),  # rad
            'phi_delta': (np.deg2rad(-10), np.deg2rad(10)),  # rad
        }

        def _denormalize_params(params_n):
            Vc_n, Rt_n, s_out_n, Vsys_n, inc_n, phi_delta_n = params_n
            Vc = Vc_n * (params_range['Vc'][1] - params_range['Vc'][0]) + params_range['Vc'][0]
            Rt = Rt_n * (params_range['Rt'][1] - params_range['Rt'][0]) + params_range['Rt'][0]
            s_out = s_out_n * (params_range['s_out'][1] - params_range['s_out'][0]) + params_range['s_out'][0]
            Vsys = Vsys_n * (params_range['Vsys'][1] - params_range['Vsys'][0]) + params_range['Vsys'][0]
            inc = inc_n * (params_range['inc'][1] - params_range['inc'][0]) + params_range['inc'][0]
            phi_delta = phi_delta_n * (params_range['phi_delta'][1] - params_range['phi_delta'][0]) + params_range['phi_delta'][0]
            return [Vc, Rt, s_out, Vsys, inc, phi_delta]

        ######################################
        # Fitting process using lmfit (replace curve_fit)
        ######################################
        def model_func(r, Vc_n, Rt_n, s_out_n, Vsys_n, inc_n, phi_delta_n):
            Vc, Rt, s_out, Vsys, inc, phi_delta = _denormalize_params([Vc_n, Rt_n, s_out_n, Vsys_n, inc_n, phi_delta_n])
            vel_rot_model = self._vel_rot_tan_sout_profile(r, Vc, Rt, s_out)
            vel_obs_model = self._vel_obs_project_profile(vel_rot_model, Vsys, inc, phi_map_valid - phi_delta)
            # vel_obs_model = np.copysign(np.abs(vel_obs_model), vel_obs_valid)
            return vel_obs_model

        # sigma: Standard Deviation of the Errors
        sigma = np.sqrt(1.0 / ivar_map_valid + (VEL_SYSTEM_ERROR)**2)  # adding floor error
        weights = np.where(np.isfinite(sigma) & (sigma > 0), 1.0 / sigma, 0.0)

        lm_model = Model(model_func, independent_vars=["r"])

        # normalized initial guesses
        params = lm_model.make_params(Vc_n=0.5, Rt_n=0.2, s_out_n=0.5, Vsys_n=0.5, inc_n=0.5, phi_delta_n=0.5)

        # Fix some parameters during fitting
        params['inc_n'].set(value=(inc_act - params_range['inc'][0]) / (params_range['inc'][1] - params_range['inc'][0]))
        params['inc_n'].vary = False  # fix inclination during fitting
        params['phi_delta_n'].set(value=0.5)  # start from zero offset
        params['phi_delta_n'].vary = False  # fix phi_delta during fitting

        # normalized bounds [0, 1]
        for name in ("Vc_n", "Rt_n", "s_out_n", "Vsys_n", "inc_n", "phi_delta_n"):
            params[name].set(min=0.0, max=1.0)

        lm_result = lm_model.fit(
            vel_obs_valid,
            params=params,
            r=radius_valid,
            weights=weights,
            method="least_squares",
            max_nfev=10000,
            nan_policy='omit',
            fit_kws={'ftol': 1e-8, 'xtol': 1e-8},
        )

        popt = np.array([
            lm_result.params["Vc_n"].value,
            lm_result.params["Rt_n"].value,
            lm_result.params["s_out_n"].value,
            lm_result.params["Vsys_n"].value,
            lm_result.params["inc_n"].value,
            lm_result.params["phi_delta_n"].value,
        ], dtype=float)

        # Covariance matrix (normalized space); provide a fallback if not available
        if lm_result.covar is not None and np.shape(lm_result.covar) == (6, 6):
            pcov = np.array(lm_result.covar, dtype=float)
        else:
            perr_n = np.array([
            lm_result.params["Vc_n"].stderr,
            lm_result.params["Rt_n"].stderr,
            lm_result.params["s_out_n"].stderr,
            lm_result.params["Vsys_n"].stderr,
            lm_result.params["inc_n"].stderr,
            lm_result.params["phi_delta_n"].stderr,
            ], dtype=float)
            perr_n = np.where(np.isfinite(perr_n), perr_n, np.nan)
            pcov = np.diag(perr_n**2)

        Vc_fit, Rt_fit, s_out_fit, Vsys_fit, inc_fit, phi_delta_fit = _denormalize_params(popt)

        ######################################
        # Error estimation (use lmfit built-ins)
        ######################################
        # Best-fit model from lmfit
        vel_obs_model = lm_result.best_fit
        residuals = np.abs(vel_obs_valid) - np.abs(vel_obs_model)
        # RMSE and NRMSE
        rmse = float(np.sqrt(np.nanmean(residuals**2)))
        nrmse = float(rmse / np.nanmean(np.abs(vel_obs_valid)))

        # Reduced Chi-Squared
        redchi = float(lm_result.redchi)  # reduced chi-squared from lmfit
        # Inflate uncertainties if reduced chi-squared > 1
        F_factor = float(np.maximum(np.sqrt(redchi), 1.0))


        # Parameter standard errors (from lmfit), then scale to physical units + optional inflation
        def _stderr(name: str) -> float:
            v = lm_result.params[name].stderr
            return float(v) if v is not None and np.isfinite(v) else np.nan

        Vc_norm_err = _stderr("Vc_n")
        Rt_norm_err = _stderr("Rt_n")
        s_out_norm_err = _stderr("s_out_n")
        Vsys_norm_err = _stderr("Vsys_n")
        inc_norm_err = _stderr("inc_n")
        phi_delta_norm_err = _stderr("phi_delta_n")

        Vc_err = Vc_norm_err * (params_range["Vc"][1] - params_range["Vc"][0]) * F_factor
        Rt_err = Rt_norm_err * (params_range["Rt"][1] - params_range["Rt"][0]) * F_factor
        s_out_err = s_out_norm_err * (params_range["s_out"][1] - params_range["s_out"][0]) * F_factor
        Vsys_err = Vsys_norm_err * (params_range["Vsys"][1] - params_range["Vsys"][0]) * F_factor
        inc_err = inc_norm_err * (params_range["inc"][1] - params_range["inc"][0]) * F_factor
        phi_delta_err = phi_delta_norm_err * (params_range["phi_delta"][1] - params_range["phi_delta"][0]) * F_factor

        Vc_err_pct = (Vc_err / Vc_fit) * 100 if Vc_fit != 0 else np.nan
        Rt_err_pct = (Rt_err / Rt_fit) * 100 if Rt_fit != 0 else np.nan
        s_out_err_pct = (s_out_err / s_out_fit) * 100 if s_out_fit != 0 else np.nan
        Vsys_err_pct = (Vsys_err / Vsys_fit) * 100 if Vsys_fit != 0 else np.nan
        inc_err_pct = (inc_err / inc_fit) * 100 if inc_fit != 0 else np.nan
        phi_delta_err_pct = (phi_delta_err / phi_delta_fit) * 100 if phi_delta_fit != 0 else np.nan

        if self.fit_debug:
            print(f"\n------------ Fitted Rotational Velocity (tanh + sout lmfit) ------------")
            print(f" IFU                    : {self.PLATE_IFU}")
            print(f" Fit  Vc                : {Vc_fit:.1e} km/s, ± {Vc_err:.0e} km/s", f"({Vc_err_pct:.2f} %)")
            print(f" Fit  Rt                : {Rt_fit:.1e} kpc/h, ± {Rt_err:.0e} kpc/h", f"({Rt_err_pct:.2f} %)")
            print(f" Fit  s_out             : {s_out_fit:.1e} km/s/kpc, ± {s_out_err:.0e} km/s/kpc", f"({s_out_err_pct:.2f} %)")
            print(f" Fit  Vsys              : {Vsys_fit:.1e} km/s, ± {Vsys_err:.0e} km/s", f"({Vsys_err_pct:.2f} %)")
            print(f" Fit  inc               : {inc_fit:.1e} rad, ± {inc_err:.0e} rad", f"({inc_err_pct:.2f} %)")
            print(f" Fit  phi_delta         : {phi_delta_fit:.1e} rad, ± {phi_delta_err:.0e} rad", f"({phi_delta_err_pct:.2f} %)")
            print("--------------")
            print(f" Calc inc from b/a      : {inc_act:.1e} rad, {np.degrees(inc_act):.2f} deg")
            print("--------------")
            print(f" Reduced Chi-Squared    : {redchi:.2f}")
            print(f" NRMSE                  : {nrmse:.3f}")
            print(f"correlation matrix      : \n{lm_result.covar}")
            # print("--------------")
            # print(f" Fit report             : \n{lm_result.fit_report()}")
            print("--------------------------------------------------------------------\n")


        ######################################
        # Return fitted velocity profile
        ######################################
        if radius_fit is None:
            radius_fit = radius_map

        # Evaluate vel_rot(r) using lmfit, and get uncertainties via eval_uncertainty
        def vel_rot_func(r, Vc_n, Rt_n, s_out_n, Vsys_n, inc_n, phi_delta_n):
            Vc, Rt, s_out, _Vsys, inc, phi_delta = _denormalize_params([Vc_n, Rt_n, s_out_n, Vsys_n, inc_n, phi_delta_n])
            return self._vel_rot_tan_sout_profile(r, Vc, Rt, s_out)

        vel_rot_model = Model(vel_rot_func, independent_vars=["r"])
        vel_rot_fitted = vel_rot_model.eval(params=lm_result.params, r=radius_fit)

        # lmfit uncertainty propagation (uses covariance internally);
        try:
            vel_fit_stderr = vel_rot_model.eval_uncertainty(params=lm_result.params, r=radius_fit, sigma=1) * F_factor
        except Exception:
            # calulate stderr manually if eval_uncertainty fails
            print("Exception: lmfit eval_uncertainty failed, calculating stderr manually...")
            # Numerical derivatives for uncertainty propagation
            vel_fit_stderr = rmse / np.sqrt(len(vel_obs_valid)) * np.ones_like(vel_rot_fitted)

        # Apply filter to output maps
        residuals = np.abs(vel_obs_valid) - np.abs(vel_obs_model)
        stderr = np.sqrt(1.0 / ivar_map_valid + (VEL_SYSTEM_ERROR)**2)
        STD_ERROR_RATIO = 3.0
        STD_ERROR_RATIO = rmse / SIGMA_OBS_BAR if SIGMA_OBS_BAR > 0 else np.nan
        print(f"SIGMA_OBS_BAR: {SIGMA_OBS_BAR:.3f} km/s, RMSE: {rmse:.3f} km/s, STD_ERROR_RATIO: {STD_ERROR_RATIO:.3f}")
        clip_mask_1d = np.abs(residuals) <= STD_ERROR_RATIO * stderr

        # Map the 1D mask (for valid points) back to the original map shape
        clip_mask = np.zeros_like(vel_obs_map, dtype=bool)
        clip_mask[valid_mask] = clip_mask_1d
        print(f"length of valid vel_obs: {np.sum(valid_mask)}, after clipping: {np.sum(clip_mask)}")


        radius_mask = np.full_like(radius_map, np.nan, dtype=float)
        radius_mask[clip_mask] = radius_map[clip_mask]
        vel_obs_mask = np.full_like(vel_obs_map, np.nan, dtype=float)
        vel_obs_mask[clip_mask] = vel_obs_map[clip_mask]
        ivar_mask = np.full_like(ivar_map, np.nan, dtype=float)
        ivar_mask[clip_mask] = ivar_map[clip_mask]

        fit_result = {
            'radius_obs': radius_mask,
            'vel_obs': vel_obs_mask,
            'ivar_obs': ivar_mask,
            'radius_rot': radius_fit,
            'vel_rot': vel_rot_fitted,
            'stderr_rot': vel_fit_stderr,
        }

        fit_parameters = {
            'result': 'success',
            'Vc': f"{Vc_fit:.2f}",
            'Rt': f"{Rt_fit:.2f}",
            's_out': f"{s_out_fit:.2f}",
            'Vsys': f"{Vsys_fit:.2f}",
            'inc': f"{inc_fit:.2f}",
            'phi_delta': f"{phi_delta_fit:.3f}",
            'Rmax': f"{R_max:.3f}",
            'RMSE': f"{rmse:.3f}",
            'NRMSE': f"{nrmse:.3f}",
            'CHI_SQ_V': f"{redchi:.2f}",
        }
        return True, fit_result, fit_parameters


    ################################################################################
    # MCMC methods
    ################################################################################
    def _inf_vel_rot(self, vel_param: dict, radius_fit: np.ndarray=None) -> tuple[bool, dict, dict]:
        def _get_arviz_api():
            return get_arviz_api()

        def _set_arviz_ci_defaults():
            try:
                if "stats.ci_prob" in az.rcParams:
                    az.rcParams["stats.ci_prob"] = HDI_PROB2
                if "stats.ci_kind" in az.rcParams:
                    az.rcParams["stats.ci_kind"] = "hdi"
            except Exception:
                pass

        def _get_summary_hdi_columns(summary_df):
            return get_summary_interval_columns(summary_df)

        def _get_posterior_dataset(idata):
            if hasattr(idata, "posterior"):
                return idata.posterior
            posterior = idata["posterior"]
            if hasattr(posterior, "dataset"):
                posterior = posterior.dataset
            return posterior

        def _calc_hdi_from_sample_matrix(samples_2d: np.ndarray) -> np.ndarray:
            # ArviZ will change how it interprets raw 2D arrays, so pass an
            # explicit (chain, draw, shape) tensor to preserve per-radius HDIs.
            sample_cube = np.expand_dims(np.asarray(samples_2d, dtype=float).T, axis=0)
            return np.asarray(az.hdi(sample_cube, hdi_prob=HDI_PROB2), dtype=float)

        radius_map = vel_param["radius_obs"]
        vel_obs_map = vel_param["vel_obs"]
        ivar_map = vel_param["ivar_obs"]
        phi_map = vel_param["phi_map"]

        valid_mask = (
            np.isfinite(vel_obs_map) &
            np.isfinite(radius_map) &
            np.isfinite(ivar_map) &
            np.isfinite(phi_map) &
            (radius_map > RADIUS_MIN_KPC)
        )

        radius_valid = np.asarray(radius_map[valid_mask], dtype=float)
        vel_obs_valid = np.asarray(vel_obs_map[valid_mask], dtype=float)
        ivar_map_valid = np.asarray(ivar_map[valid_mask], dtype=float)
        phi_map_valid = np.asarray(phi_map[valid_mask], dtype=float)

        if radius_valid.size < VEL_OBS_COUNT_THRESHOLD:
            print(f"Insufficient valid data points for MCMC fit: {radius_valid.size} < {VEL_OBS_COUNT_THRESHOLD}")
            return False, {}, {"result": "insufficient_data", "data_count": int(radius_valid.size)}

        if radius_fit is None:
            radius_fit = radius_map

        radius_fit = np.asarray(radius_fit, dtype=float)
        inc_act = float(self.get_inc_rad())
        r_max = float(np.nanmax(radius_valid))
        sigma_meas = np.sqrt(1.0 / ivar_map_valid)
        sigma_obs_bar = float(np.nanmedian(sigma_meas))

        pri_inc = bool(getattr(self, "pri_inc", True))
        pri_phi_delta = bool(getattr(self, "pri_phi_delta", True))
        debug = bool(getattr(self, "fit_debug", False))
        success = True

        vel_abs_scale = float(max(np.nanpercentile(np.abs(vel_obs_valid), 90), 40.0))
        vc_mu = float(np.clip(vel_abs_scale / max(np.sin(inc_act), 0.25), 40.0, 350.0))
        vc_sigma = float(max(0.6 * vc_mu, 40.0))
        rt_mu = float(max(0.25 * r_max, RADIUS_MIN_KPC))
        vsys_mu = float(np.clip(np.nanmedian(vel_obs_valid), -60.0, 60.0))
        vsys_sigma = float(max(2.0 * np.nanmedian(sigma_meas), 12.0))
        sigma_int_scale = float(max(3.0 * np.nanmedian(sigma_meas), 12.0))

        inc_low = max(float(np.deg2rad(np.rad2deg(inc_act) - 10.0)), 1e-3)
        inc_high = min(float(np.deg2rad(np.rad2deg(inc_act) + 10.0)), np.pi / 2 - 1e-3)
        if inc_high <= inc_low:
            inc_low = max(inc_act - np.deg2rad(2.0), 1e-3)
            inc_high = min(inc_act + np.deg2rad(2.0), np.pi / 2 - 1e-3)

        rt_lower = max(r_max * 0.01, RADIUS_MIN_KPC)
        rt_upper = max(r_max * 1.0, rt_lower * 1.1)

        if debug:
            print(f"RC pymc radius valid: range=[{np.min(radius_valid):.3f}, {np.max(radius_valid):.3f}] kpc")
            print(f"RC pymc vel obs valid {len(vel_obs_valid)}: range=[{np.min(vel_obs_valid):.3f}, {np.max(vel_obs_valid):.3f}] km/s")
            print(f"RC pymc sigma_meas: mean={np.nanmean(sigma_meas):.3f} km/s, median={np.nanmedian(sigma_meas):.3f} km/s")

        def vel_rot_tan_sout_profile_t(r, vc, rt, s_out):
            rt_safe = pt.maximum(rt, 1e-6)
            return vc * pt.tanh(r / rt_safe) + s_out * r

        def vel_obs_project_profile_t(vel_rot, vel_sys, inc, phi_term):
            correction = pt.sin(inc) * pt.cos(phi_term + np.pi)
            return vel_sys + vel_rot * correction

        with pm.Model() as model:
            vc_t = pm.TruncatedNormal("Vc", mu=vc_mu, sigma=vc_sigma, lower=20.0, upper=500.0)
            rt_t = pm.LogNormal("Rt", mu=np.log(rt_mu), sigma=0.7)
            s_out_t = pm.TruncatedNormal("s_out", mu=0.0, sigma=5.0, lower=-15.0, upper=15.0)
            vsys_t = pm.TruncatedNormal("Vsys", mu=vsys_mu, sigma=vsys_sigma, lower=-120.0, upper=120.0)

            if pri_inc:
                inc_t = pm.TruncatedNormal("inc", mu=inc_act, sigma=np.deg2rad(5.0), lower=inc_low, upper=inc_high)
            else:
                inc_t = pm.Deterministic("inc", pt.as_tensor_variable(float(inc_act)))

            if pri_phi_delta:
                phi_delta_t = pm.TruncatedNormal(
                    "phi_delta",
                    mu=0.0,
                    sigma=np.deg2rad(7.5),
                    lower=-np.deg2rad(20.0),
                    upper=np.deg2rad(20.0),
                )
            else:
                phi_delta_t = pm.Deterministic("phi_delta", pt.as_tensor_variable(0.0))


            # intrinsic scatter prior
            sigma_int_t = pm.Exponential("sigma_int", lam=1.0 / sigma_int_scale)

            # nu prior
            nu_minus_t = pm.Gamma("nu_minus", alpha=2.0, beta=0.1)
            nu_t = pm.Deterministic("nu", nu_minus_t + 2.0)

            r_valid_t = pt.as_tensor_variable(radius_valid)
            phi_valid_t = pt.as_tensor_variable(phi_map_valid)
            sigma_meas_t = pt.as_tensor_variable(sigma_meas)

            vel_rot_valid_t = pm.Deterministic("vel_rot_valid", vel_rot_tan_sout_profile_t(r_valid_t, vc_t, rt_t, s_out_t))
            vel_obs_model_t = pm.Deterministic(
                "vel_obs_model",
                vel_obs_project_profile_t(vel_rot_valid_t, vsys_t, inc_t, phi_valid_t - phi_delta_t),
            )

            sigma_obs_t = pm.Deterministic("sigma_obs", pt.sqrt(sigma_meas_t**2 + sigma_int_t**2))
            pm.StudentT("vel_obs_like", mu=vel_obs_model_t, sigma=sigma_obs_t, nu=nu_t, observed=vel_obs_valid)

            if debug:
                print(">>> Starting PyMC sampling for rotational velocity inference (NUTS)...")

            chains = min(4, os.cpu_count() or 1)
            sample_kwargs = {
                "init": "jitter+adapt_full",
                "draws": 1000,
                "tune": 500,
                "chains": chains,
                "cores": chains,
                "target_accept": 0.95,
                "progressbar": debug,
                "random_seed": 42,
                "return_inferencedata": True,
                "compute_convergence_checks": debug,
            }

            try:
                trace = pm.sample(nuts_sampler="nutpie", **sample_kwargs)
            except Exception as exc:
                if debug:
                    print(f"Warning: nutpie sampling failed ({exc}), falling back to default PyMC NUTS.")
                trace = pm.sample(**sample_kwargs)

        az_api = _get_arviz_api()
        _set_arviz_ci_defaults()

        var_names = ["Vc", "Rt", "s_out", "Vsys", "sigma_int", "nu"]
        if pri_inc:
            var_names.append("inc")
        if pri_phi_delta:
            var_names.append("phi_delta")

        summary = summary_with_compat(
            az_api.summary,
            trace,
            var_names=var_names,
            round_to=3,
            stat_focus="median",
        )
        hdi_low_col, hdi_high_col = _get_summary_hdi_columns(summary)
        ess_col = "ess_bulk" if "ess_bulk" in summary.columns else "ess_median" if "ess_median" in summary.columns else None

        for var_name in var_names:
            if "r_hat" not in summary.columns:
                break
            r_hat = float(summary.loc[var_name, "r_hat"])
            if np.isfinite(r_hat) and r_hat > INFER_RHAT_THRESHOLD:
                print(f"Warning: R-hat for variable {var_name} is {r_hat:.3f} > {INFER_RHAT_THRESHOLD}.")
                success = False

            if ess_col is not None:
                ess_value = float(summary.loc[var_name, ess_col])
                if np.isfinite(ess_value) and ess_value < INFER_ESS_THRESHOLD:
                    print(f"Warning: {ess_col} for variable {var_name} is {ess_value:.1f} < {INFER_ESS_THRESHOLD}, indicating potential sampling inefficiency.")
                    success = False

        posterior = _get_posterior_dataset(trace)
        flat_trace = posterior.stack(sample=("chain", "draw"))

        vc_samples = np.asarray(flat_trace["Vc"].values, dtype=float)
        rt_samples = np.asarray(flat_trace["Rt"].values, dtype=float)
        s_out_samples = np.asarray(flat_trace["s_out"].values, dtype=float)
        vsys_samples = np.asarray(flat_trace["Vsys"].values, dtype=float)
        sigma_int_samples = np.asarray(flat_trace["sigma_int"].values, dtype=float)

        if pri_inc:
            inc_samples = np.asarray(flat_trace["inc"].values, dtype=float)
        else:
            inc_samples = np.full_like(vc_samples, inc_act, dtype=float)

        if pri_phi_delta:
            phi_delta_samples = np.asarray(flat_trace["phi_delta"].values, dtype=float)
        else:
            phi_delta_samples = np.zeros_like(vc_samples, dtype=float)

        vel_rot_valid_samples = self._vel_rot_tan_sout_profile(
            radius_valid[:, None],
            vc_samples[None, :],
            rt_samples[None, :],
            s_out_samples[None, :],
        )

        vc_median = float(summary.loc["Vc", "median"])
        rt_median = float(summary.loc["Rt", "median"])
        s_out_median = float(summary.loc["s_out", "median"])
        vsys_median = float(summary.loc["Vsys", "median"])
        inc_median = float(summary.loc["inc", "median"]) if "inc" in summary.index else float(inc_act)
        phi_delta_median = float(summary.loc["phi_delta", "median"]) if "phi_delta" in summary.index else 0.0
        sigma_int_median = float(summary.loc["sigma_int", "median"])

        vc_hdi_low = float(summary.loc['Vc', hdi_low_col])
        vc_hdi_high = float(summary.loc['Vc', hdi_high_col])
        rt_hdi_low = float(summary.loc['Rt', hdi_low_col])
        rt_hdi_high = float(summary.loc['Rt', hdi_high_col])
        s_out_hdi_low = float(summary.loc['s_out', hdi_low_col])
        s_out_hdi_high = float(summary.loc['s_out', hdi_high_col])
        vsys_hdi_low = float(summary.loc['Vsys', hdi_low_col])
        vsys_hdi_high = float(summary.loc['Vsys', hdi_high_col])
        inc_hdi_low = float(summary.loc['inc', hdi_low_col]) if 'inc' in summary.index else float(inc_act)
        inc_hdi_high = float(summary.loc['inc', hdi_high_col]) if 'inc' in summary.index else float(inc_act)
        phi_delta_hdi_low = float(summary.loc['phi_delta', hdi_low_col]) if 'phi_delta' in summary.index else 0.0
        phi_delta_hdi_high = float(summary.loc['phi_delta', hdi_high_col]) if 'phi_delta' in summary.index else 0.0
        sigma_int_hdi_low = float(summary.loc['sigma_int', hdi_low_col])
        sigma_int_hdi_high = float(summary.loc['sigma_int', hdi_high_col])

        vel_rot_valid_median = self._vel_rot_tan_sout_profile(radius_valid, vc_median, rt_median, s_out_median)
        vel_obs_model_median = self._vel_obs_project_profile(vel_rot_valid_median, vsys_median, inc_median, phi_map_valid - phi_delta_median)

        radius_fit_samples = np.expand_dims(radius_fit, axis=-1)
        vel_rot_fit_samples = self._vel_rot_tan_sout_profile(radius_fit_samples, vc_samples, rt_samples, s_out_samples)
        vel_rot_fitted = np.nanmedian(vel_rot_fit_samples, axis=-1)
        vel_fit_stderr = np.nanstd(vel_rot_fit_samples, axis=-1)
        sigma_rot_curve_samples = np.sqrt(sigma_int_samples**2 + np.nanmedian(sigma_meas)**2) / np.maximum(np.sin(inc_samples), 0.25)
        rng_pp = np.random.default_rng(314159)
        vel_rot_pp_samples = vel_rot_fit_samples + rng_pp.normal(
            loc=0.0,
            scale=sigma_rot_curve_samples[np.newaxis, :],
            size=vel_rot_fit_samples.shape,
        )
        vel_rot_hdi = _calc_hdi_from_sample_matrix(vel_rot_pp_samples)
        vel_rot_hdi_low = vel_rot_hdi[:, 0]
        vel_rot_hdi_high = vel_rot_hdi[:, 1]

        vel_obs_valid_samples = self._vel_obs_project_profile(
            vel_rot_valid_samples,
            vsys_samples[None, :],
            inc_samples[None, :],
            phi_map_valid[:, None] - phi_delta_samples[None, :],
        )
        sigma_obs_valid_samples = np.sqrt(sigma_meas[:, None]**2 + sigma_int_samples[None, :]**2)
        rng_obs_pp = np.random.default_rng(271828)
        vel_obs_pp_samples = vel_obs_valid_samples + rng_obs_pp.normal(
            loc=0.0,
            scale=sigma_obs_valid_samples,
            size=vel_obs_valid_samples.shape,
        )
        vel_obs_pp_hdi = _calc_hdi_from_sample_matrix(vel_obs_pp_samples)
        vel_obs_hdi_low = vel_obs_pp_hdi[:, 0]
        vel_obs_hdi_high = vel_obs_pp_hdi[:, 1]

        ppc_value_mask = (
            np.isfinite(vel_obs_valid) &
            np.isfinite(vel_obs_hdi_low) &
            np.isfinite(vel_obs_hdi_high) &
            (vel_obs_valid >= vel_obs_hdi_low) &
            (vel_obs_valid <= vel_obs_hdi_high)
        )
        ppc_overlap_mask = self._calc_interval_overlap_mask(
            vel_obs_valid,
            sigma_meas,
            vel_obs_hdi_low,
            vel_obs_hdi_high,
            sigma_scale=PPC_MEAS_SIGMA_SCALE,
        )
        ppc_hdi_value_coverage = float(np.mean(ppc_value_mask.astype(float))) if ppc_value_mask.size > 0 else np.nan
        ppc_hdi_overlap = float(np.mean(ppc_overlap_mask.astype(float))) if ppc_overlap_mask.size > 0 else np.nan

        sample_count = vel_rot_fit_samples.shape[1] if vel_rot_fit_samples.ndim == 2 else 0
        draw_count = min(int(getattr(self, "n_samples", 0)), sample_count)
        if draw_count > 0:
            rng = np.random.default_rng(42)
            sample_indices = rng.choice(sample_count, size=draw_count, replace=sample_count < draw_count)
            vel_rot_samples_plot = vel_rot_fit_samples[:, sample_indices]
        else:
            vel_rot_samples_plot = np.empty((len(radius_fit), 0), dtype=float)

        residuals = np.abs(vel_obs_valid) - np.abs(vel_obs_model_median)
        rmse = float(np.sqrt(np.nanmean(residuals**2)))
        nrmse = float(rmse / np.nanmean(np.abs(vel_obs_valid)))

        sigma_fit = np.sqrt(sigma_meas**2 + sigma_int_median**2)
        params_num = 5
        if pri_inc:
            params_num += 1
        if pri_phi_delta:
            params_num += 1
        dof = max(int(np.sum(np.isfinite(vel_obs_valid) & np.isfinite(sigma_fit)) - params_num), 1)
        redchi = float(np.nansum((residuals / sigma_fit) ** 2) / dof)

        stderr_clip = np.sqrt(sigma_meas**2 + sigma_int_median**2)
        std_error_ratio = rmse / sigma_obs_bar if sigma_obs_bar > 0 else np.nan
        if not np.isfinite(std_error_ratio) or std_error_ratio <= 0:
            std_error_ratio = 3.0
        clip_mask_1d = np.abs(residuals) <= std_error_ratio * stderr_clip

        clip_mask = np.zeros_like(vel_obs_map, dtype=bool)
        clip_mask[valid_mask] = clip_mask_1d

        radius_mask = np.full_like(radius_map, np.nan, dtype=float)
        radius_mask[clip_mask] = radius_map[clip_mask]
        vel_obs_mask = np.full_like(vel_obs_map, np.nan, dtype=float)
        vel_obs_mask[clip_mask] = vel_obs_map[clip_mask]
        ivar_mask = np.full_like(ivar_map, np.nan, dtype=float)
        ivar_mask[clip_mask] = ivar_map[clip_mask]

        ppc_value_map = np.zeros_like(vel_obs_map, dtype=bool)
        ppc_value_map[valid_mask] = ppc_value_mask
        ppc_overlap_map = np.zeros_like(vel_obs_map, dtype=bool)
        ppc_overlap_map[valid_mask] = ppc_overlap_mask

        if debug:
            summary_cols = ["median", hdi_low_col, hdi_high_col, "r_hat"]
            if ess_col is not None:
                summary_cols.append(ess_col)
            print("\n------------ Infer Rotational Velocity (PyMC) ------------")
            print(summary[summary_cols])
            print("--- Median ---")
            print(f" Median Vc          : {vc_median:.3f} [{vc_hdi_low:.3f}, {vc_hdi_high:.3f}] km/s")
            print(f" Median Rt          : {rt_median:.3f} [{rt_hdi_low:.3f}, {rt_hdi_high:.3f}] kpc/h")
            print(f" Median s_out       : {s_out_median:.3f} [{s_out_hdi_low:.3f}, {s_out_hdi_high:.3f}] km/s/kpc")
            print(f" Median Vsys        : {vsys_median:.3f} [{vsys_hdi_low:.3f}, {vsys_hdi_high:.3f}] km/s")
            print(f" Median inc         : {np.degrees(inc_median):.3f} [{np.degrees(inc_hdi_low):.3f}, {np.degrees(inc_hdi_high):.3f}] deg")
            print(f" Median phi_delta   : {np.degrees(phi_delta_median):.3f} [{np.degrees(phi_delta_hdi_low):.3f}, {np.degrees(phi_delta_hdi_high):.3f}] deg")
            print(f" Median sigma_int   : {sigma_int_median:.3f} [{sigma_int_hdi_low:.3f}, {sigma_int_hdi_high:.3f}] km/s")
            print("--- Diagnostics ---")
            print(f" Reduced Chi-Squared: {redchi:.3f}")
            print(f" NRMSE              : {nrmse:.3f}")
            print(f" RMSE               : {rmse:.3f} km/s")
            print(f" PPC Value Coverage : {ppc_hdi_value_coverage:.3f} ({HDI_PROB2:.0%} HDI)")
            print(f" PPC Overlap Rate   : {ppc_hdi_overlap:.3f} (threshold={PPC_HDI_OVERLAP_THRESHOLD:.3f}, obs ±{PPC_MEAS_SIGMA_SCALE:.1f}σ)")
            print(f" Valid count        : {np.sum(valid_mask)} -> {np.sum(clip_mask)} after clipping")
            print("----------------------------------------------------------\n")

        fit_result = {
            'radius_obs': radius_mask,
            'vel_obs': vel_obs_mask,
            'ivar_obs': ivar_mask,
            'radius_rot': radius_fit,
            'vel_rot': vel_rot_fitted,
            'stderr_rot': vel_fit_stderr,
            'vel_rot_hdi_low': vel_rot_hdi_low,
            'vel_rot_hdi_high': vel_rot_hdi_high,
            'vel_rot_samples': vel_rot_samples_plot,
            'ppc_value_mask': ppc_value_map,
            'ppc_overlap_mask': ppc_overlap_map,
        }


        fit_parameters = {
            'result': 'success' if success else 'failure',
            'Vc': f"{vc_median:.2f}",
            'Vc_hdi_low': f"{vc_hdi_low:.2f}",
            'Vc_hdi_high': f"{vc_hdi_high:.2f}",
            'Rt': f"{rt_median:.2f}",
            'Rt_hdi_low': f"{rt_hdi_low:.2f}",
            'Rt_hdi_high': f"{rt_hdi_high:.2f}",
            's_out': f"{s_out_median:.2f}",
            's_out_hdi_low': f"{s_out_hdi_low:.2f}",
            's_out_hdi_high': f"{s_out_hdi_high:.2f}",
            'Vsys': f"{vsys_median:.2f}",
            'Vsys_hdi_low': f"{vsys_hdi_low:.2f}",
            'Vsys_hdi_high': f"{vsys_hdi_high:.2f}",
            'inc': f"{inc_median:.2f}",
            'inc_hdi_low': f"{inc_hdi_low:.4f}",
            'inc_hdi_high': f"{inc_hdi_high:.4f}",
            'phi_delta': f"{phi_delta_median:.3f}",
            'phi_delta_hdi_low': f"{phi_delta_hdi_low:.4f}",
            'phi_delta_hdi_high': f"{phi_delta_hdi_high:.4f}",
            'sigma_int': f"{sigma_int_median:.3f}",
            'sigma_int_hdi_low': f"{sigma_int_hdi_low:.3f}",
            'sigma_int_hdi_high': f"{sigma_int_hdi_high:.3f}",
            'Rmax': f"{r_max:.3f}",
            'RMSE': f"{rmse:.3f}",
            'NRMSE': f"{nrmse:.3f}",
            'CHI_SQ_V': f"{redchi:.2f}",
            'PPC_HDI_PROB': f"{HDI_PROB2:.3f}",
            'PPC_HDI_VALUE_COVERAGE': f"{ppc_hdi_value_coverage:.3f}",
            'PPC_HDI_OVERLAP': f"{ppc_hdi_overlap:.3f}",
        }
        return success, fit_result, fit_parameters

    ################################################################################
    # public methods
    ################################################################################
    def set_PLATE_IFU(self, plate_ifu: str) -> None:
        self.PLATE_IFU = plate_ifu
        return

    def set_fit_debug(self, debug: bool=True) -> None:
        self.fit_debug = debug
        return

    def set_fit_model(self, model_name: str) -> None:
        if model_name not in ("fit", "mcmc"):
            raise ValueError(f"Unsupported fit model: {model_name}")
        self.fit_model = model_name
        return

    def set_n_samples(self, n_samples: int = 0) -> None:
        n_samples = int(n_samples)
        if n_samples < 0:
            raise ValueError(f"n_samples must be >= 0, got {n_samples}")
        self.n_samples = n_samples
        return

    def get_inc_rad(self):
        _, inc_rad = self._calc_pa_inc()
        return inc_rad

    def get_radius_fit(self, radius_max, count: int=200) -> np.ndarray:
        radius_fit = np.linspace(-radius_max, radius_max, num=count)
        return radius_fit

    # observed velocity
    def get_vel_obs(self, is_filter: bool=True):
        return self._get_vel_obs_raw(type='gas', is_filter=is_filter)

    # disprojected velocity
    def get_vel_obs_disp(self, inc_rad:float, vel_sys: float, phi_delta: float=0.0, is_filter: bool=True):
        r_map, v_obs_map, ivar_map, phi_map = self.get_vel_obs(is_filter=is_filter)
        v_rot_map = self._vel_rot_disproject_profile(v_obs_map, vel_sys, inc_rad, phi_map - phi_delta)
        return r_map, v_rot_map, ivar_map

    def fit_vel_rot(self, vel_param: dict, radius_fit=None):
        if self.fit_model == "fit":
            return self._fit_vel_rot(vel_param, radius_fit=radius_fit)
        elif self.fit_model == "mcmc":
            return self._inf_vel_rot(vel_param, radius_fit=radius_fit)


######################################################
# main function for test
######################################################