"""NFW dark-matter halo inference model.

Migrated from ``src-orig/dm.py``.  The ``DmNfw`` class fits a full
NFW dark-matter profile to MaNGA velocity data using PyMC.

The PyMC model code in ``_inf_dm_nfw_pymc`` is **preserved verbatim**.
"""

from __future__ import annotations

import os
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytensor
import pytensor.tensor as pt
from scipy.optimize import brentq
from scipy.special import gammaln, i0, i1, k0, k1
from astropy import constants as const
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
import matplotlib.pyplot as plt

import pymc as pm

# New layered imports
from src.config.constants import (
    H0_PHYS, H_ACTUAL, M_PIVOT_H_INV, STRUCT_RD_FACTOR,
    STRUCT_HERNQUIST_FACTOR, ARCSEC_PER_RADIAN,
    MOSTER_M1, MOSTER_N, MOSTER_BETA, MOSTER_GAMMA,
    OMEGA_M, OMEGA_L, COLOR_M200, COLOR_C,
    LOG10_C0_DM14, ALPHA_DM14,
)
from src.config.settings import settings
from src.data.catalog import DrpallUtil
from src.stats.arviz_compat import (
    ensure_arviz_compat, get_arviz_api,
    get_summary_interval_columns, summary_with_compat,
    set_arviz_ci_defaults, get_posterior_dataset,
    get_prior_dataset,
)
from src.stats.intervals import (
    calc_eti_from_sample_matrix,
    calc_interval_overlap_mask,
    get_interval_value_formatter,
    format_pair_interval_title,
)
from src.stats.gmm import fit_log10_mc_gmm
from src.viz.posterior import annotate_pair_marginals

# ── ArviZ compat ────────────────────────────────────────────────────
az = ensure_arviz_compat()

# ── Config shims (module-level names preserved for backward-compat) ──
HDI_PROB1 = settings.HDI_PROB1
HDI_PROB2 = settings.HDI_PROB2
INFER_RHAT_THRESHOLD = settings.INFER_RHAT_THRESHOLD
INFER_ESS_THRESHOLD = settings.INFER_ESS_THRESHOLD
PPC_HDI_VALUE_COVERAGE_THRESHOLD = settings.PPC_HDI_VALUE_COVERAGE_THRESHOLD
PPC_HDI_OVERLAP_THRESHOLD = settings.PPC_HDI_OVERLAP_THRESHOLD
PPC_MEAS_SIGMA_SCALE = settings.PPC_MEAS_SIGMA_SCALE

# Physical constants (module-level for inline use)
H0 = H0_PHYS
G_kpc_kms_Msun = const.G.to("kpc km^2 / s^2 Msun").value

# ── Thin helper proxies (for code in class that calls bare `_get_arviz_api()` etc.) ──

_get_arviz_api = get_arviz_api
_set_arviz_ci_defaults = lambda: set_arviz_ci_defaults(HDI_PROB1, "eti")
_get_posterior_dataset = get_posterior_dataset
_get_prior_dataset = get_prior_dataset
_get_summary_eti_columns = get_summary_interval_columns
_calc_eti_from_sample_matrix = calc_eti_from_sample_matrix
_calc_interval_overlap_mask = calc_interval_overlap_mask
_get_interval_value_formatter = get_interval_value_formatter
_format_pair_interval_title = format_pair_interval_title
_fit_log10_mc_gmm = fit_log10_mc_gmm
_annotate_pair_marginals = annotate_pair_marginals
class DmNfw:
    drpall_util: DrpallUtil
    PLATE_IFU: str
    plot_enable: bool
    inf_debug: bool = False

    def __init__(self, drpall_util: DrpallUtil):
        self.drpall_util = drpall_util
        self.PLATE_IFU = None
        self.plot_enable = False
        self.r0_frac = 0.3  # inner logistic down-weighting half-weight radius as fraction of r_max
        self.M200_dex = 0.15  # scale factor for M200 prior width (relative to SHMR estimate)
        self.inc_prior_enable = False  # whether to use an informative prior on inclination based on photometry

    ################################################################################
    # Helper calculation methods
    ################################################################################
    def _get_z(self) -> float:
        z = self.drpall_util.get_redshift(self.PLATE_IFU)
        print(f"Redshift z from DRPALL: {z:.5f}")
        return z

    # hubble parameter
    # H(z) = H0 * sqrt( Omega_m*(1 + z)^3 + Omega_Lambda )
    def _calc_Hz_kpc(self, z: float, H0=H0, Om=0.315, Ol=0.685) -> float:
        Hz = H0 * np.sqrt(Om * (1 + z)**3 + Ol)
        Hz = Hz / 1000
        return Hz # in km/s/kpc

    def _calc_r200_from_V200(self, V200: float, z: float) -> float:
        Hz = self._calc_Hz_kpc(z)  # in km/s/kpc
        r200_kpc = V200 / (10 * Hz)  # in kpc
        return r200_kpc # in kpc

   # c = r200 / rss
   # c = 8.5 * ( M200 / (10^12 * h^-1 * Msun) )^(-0.10)
    def _calc_c_from_M200(self, M200: float, h: float) -> float:
        mass_ratio = M200 / (M_PIVOT_H_INV / h)
        return 8.5 * (mass_ratio)**(-0.10)

    # formula: V200^3 = 10 * G * H(z) * M200
    def _calc_V200_from_M200(self, M200: float, z: float) -> float:
        Hz = self._calc_Hz_kpc(z)  # in km/s/kpc
        V200 = (10 * G_kpc_kms_Msun * Hz * M200)**(1/3)  # in km/s
        return V200

    # Moster-like SHMR
    # M1 from Moster+2013 is in h^-1 Msun; convert to physical Msun by dividing by h
    def _calc_Mstar_from_Mhalo(self, M200: float, M1=10**11.59 / H_ACTUAL, N=0.0351, beta=1.376, gamma=0.608):
        x = M200 / M1
        f = 2.0 * N / (x**(-beta) + x**(gamma))
        return f * M200

    def _calc_M200_from_Mstar(self, Mstar: float, Mmin=1e9, Mmax=1e15):
        def f(M):
            return self._calc_Mstar_from_Mhalo(M) - Mstar

        return brentq(f, Mmin, Mmax)

    def _get_Re_kpc(self, z: float) -> float:
        """Return half-light radius in kpc.

        Uses the NSA Petrosian r-band half-light radius from DRPALL (arcsec)
        converted to kpc via the angular diameter distance using astropy.cosmology.
        """
        Re_arcsec = self.drpall_util.get_effective_radius(self.PLATE_IFU)

        cosmo = FlatLambdaCDM(H0=H0, Om0=0.315)
        D_A_kpc = cosmo.angular_diameter_distance(z).to(u.kpc).value
        Re_kpc = float(Re_arcsec) / 206265.0 * D_A_kpc
        print(f"Re from DRPALL: {Re_arcsec:.2f} arcsec -> {Re_kpc:.2f} kpc (D_A={D_A_kpc:.0f} kpc)")
        return Re_kpc

    def _get_mass_star(self) -> float:
        Mstar_elpetro, Mstar_sersic = self.drpall_util.get_stellar_mass(self.PLATE_IFU)

        print (f"Stellar mass from DRPALL: Mstar_elpetro={Mstar_elpetro:.2e} Msun, Mstar_sersic={Mstar_sersic:.2e} Msun")

        Mstar = Mstar_elpetro if Mstar_elpetro is not None else Mstar_sersic

        scale = (1.0 / H_ACTUAL) ** 2
        Mstar_scaled = Mstar * scale
        print(f"Stellar mass scaled by h^-2: Mstar_scaled={Mstar_scaled:.2e} Msun (h={H_ACTUAL})")
        return Mstar_scaled

    ################################################################################
    # MCMC PyMC inference methods
    ################################################################################
    def _inf_dm_nfw_pymc(
        self,
        vel_param: dict,
        radius_fit: np.ndarray=None,
    ):
        radius_obs = np.asarray(vel_param["radius_obs"], dtype=float)
        vel_obs = np.asarray(vel_param["vel_obs"], dtype=float)
        ivar_obs = np.asarray(vel_param["ivar_obs"], dtype=float)
        vel_sys = vel_param["vel_sys"]
        inc_rad = vel_param["inc_rad"]
        phi_map = np.asarray(vel_param["phi_map"], dtype=float)
        # ------------------------------------------
        # data selection / precompute
        # ------------------------------------------
        valid_mask = (np.isfinite(vel_obs) & np.isfinite(radius_obs) & np.isfinite(ivar_obs) & np.isfinite(phi_map) &
                    (radius_obs > 0.01) & (radius_obs < 1.0 * np.nanmax(radius_obs)))
        radius_valid = radius_obs[valid_mask]
        vel_obs_valid = vel_obs[valid_mask]
        ivar_obs_valid = ivar_obs[valid_mask]
        phi_map_valid = phi_map[valid_mask]

        if radius_fit is None:
            radius_fit = radius_obs

        radius_fit = np.asarray(radius_fit, dtype=float).reshape(-1)
        radius_fit = radius_fit[np.isfinite(radius_fit)]
        if radius_fit.size == 0:
            radius_fit = np.asarray(radius_valid, dtype=float).reshape(-1)

        success = True

        print(f"NFW pymc radius valid: range=[{np.min(radius_valid):.2f}, {np.max(radius_valid):.2f}] kpc")
        print(f"NFW pymc vel obs valid {len(vel_obs_valid)}: range=[{np.min(vel_obs_valid):.2f}, {np.max(vel_obs_valid):.2f}] km/s")

        # Convert inverse-variance to 1-sigma error
        stderr_obs_valid = np.sqrt(1.0 / ivar_obs_valid)
        sigma_meas_valid = stderr_obs_valid
        print(f"vel_obs stderr: {np.nanmean(stderr_obs_valid):.2f} km/s")
        print(f"vel_obs sigma_meas: {np.nanmean(sigma_meas_valid):.2f} km/s")

        r_max = np.nanmax(radius_valid)

        # inner logistic down-weighting parameters
        r0_like = r_max * self.r0_frac
        width_like = max(r_max * 0.1, 1e-6)
        w_min_like = 0.3
        w_rc_like_np = w_min_like + (1.0 - w_min_like) / (1.0 + np.exp(-(radius_valid - r0_like) / width_like))
        print(f"inner logistic down-weighting: r0={r0_like:.2f} kpc (r0_frac={self.r0_frac:.2f}), width={width_like:.2f} kpc, w_min={w_min_like:.2f}")

        # stellar mass
        # Mstar is the total stellar mass for the galaxy with infinity radius, which should be larger than the observed Mstar within finite aperture.
        Mstar_obs = self._get_mass_star() * 1.0

        # estimate M200 from Mstar
        Mmin=1e9
        Mmax=1e15
        M200_shmr = self._calc_M200_from_Mstar(Mstar_obs, Mmin=Mmin, Mmax=Mmax)

        z = self._get_z()

        # get H(z) in km/s/kpc
        Hz = self._calc_Hz_kpc(z)

        # Photometric half-light radius (kpc) used to anchor Re and a priors
        Re_ref_kpc = self._get_Re_kpc(z)

        # ------------------------------------------
        # helper functions (closures)
        # Note: use pytensor operations instead of numpy/scipy inside the model
        # ------------------------------------------

        # ------------------------------------------
        # Vstar
        # ------------------------------------------
        # bulge component: Hernquist profile
        def v_star_sq_bulge(r, MB, a):
            v_sq = (G_kpc_kms_Msun * MB * r) / (r + a)**2
            return v_sq


        # V_disk^2(r) = (2 * G * M_baryon / Rd) * y^2 * [I_0(y) K_0(y) - I_1(y) K_1(y)]
        # disk component: Freeman exponential disk
        def v_star_sq_disk(r, M_d, Rd):
            y = r / (2.0 * Rd)

            # PyMC does not expose bessel_i0/i1/k0/k1 in all versions.
            # Use PyTensor's generic modified Bessel functions:
            # I_n(y) = iv(n, y), K_n(y) = kv(n, y)
            I0 = pt.iv(0, y)
            I1 = pt.iv(1, y)
            K0 = pt.kv(0, y)
            K1 = pt.kv(1, y)

            v_sq = (2.0 * G_kpc_kms_Msun * M_d / Rd) * (y**2) * (I0 * K0 - I1 * K1)
            return v_sq


        # M_star: total mass of star
        # Re: Half-mass radius
        # f_bulge: bulge mass fraction
        # a: Hernquist scale radius
        # V_star^2 = (G * MB * r) / (r + a)^2 +(2 * G * M_baryon / Rd) * y^2 * [I_0(y) K_0(y) - I_1(y) K_1(y)]
        def v_star_sq_profile(r, Mstar, Re, f_bulge, a):
            r_safe = pt.where(pt.eq(r, 0), 1e-6, r)
            Rd = Re / 1.678
            MB = f_bulge * Mstar
            MD = (1 - f_bulge) * Mstar

            v_bulge_sq = v_star_sq_bulge(r_safe, MB, a)
            v_disk_sq = v_star_sq_disk(r_safe, MD, Rd)
            v_baryon_sq = v_bulge_sq + v_disk_sq
            return v_baryon_sq


        # ------------------------------------------
        # Vdm
        # ------------------------------------------
        # Moster-like SHMR
        # M1 from Moster+2013 is in h^-1 Msun; convert to physical Msun by dividing by h
        def Mstar_from_M200(M200, M1=10**11.59 / H_ACTUAL, N=0.0351, beta=1.376, gamma=0.608):
            x = M200 / M1
            f = 2.0 * N / (x**(-beta) + x**(gamma))
            return f * M200

        # c = c0 * (M200 / (M_pivot_h_inv / h))^alpha
        # logc = log(c0) + alpha * (log(M200) - log(M_pivot_h_inv / h))
        def c_M200_profile(M200, h, c0, alpha):
            mass_ratio = M200 / (M_PIVOT_H_INV / h)
            return c0 * (mass_ratio)**alpha

        # c = 5.74 * ( M200 / (2 * 10^12 * h^-1 * Msun) )^(-0.097)
        def c_from_M200(M200, h):
            return c_M200_profile(M200, h, 8.5, -0.10)

        # r200 closure (kpc) from M200 and Hz
        def r200_from_M200(M200):
            return (G_kpc_kms_Msun * M200 / (100.0 * Hz ** 2)) ** (1.0 / 3.0)

        # normalized radius x = r / r200
        def x_from_M200(r, M200):
            return r / r200_from_M200(M200)

        def V200_from_M200(M200):
            return (10.0 * G_kpc_kms_Msun * Hz * M200) ** (1.0 / 3.0)


        # numerator/denominator for NFW profile
        def nfw_num_den(x, c):
            cx = c * x
            num = pt.log1p(cx) - (cx) / (1.0 + cx)
            den = pt.maximum(pt.log1p(c) - c / (1.0 + c), 1e-12)
            return num, den

        # Vdm^2(r) = V200^2 / x * (ln(1 + c*x) - (c*x)/(1 + c*x)) / (ln(1 + c) - c/(1 + c))
        def v_dm_sq_profile(r, M200, c):
            x = x_from_M200(r, M200)
            num, den = nfw_num_den(x, c)
            x_safe = pt.maximum(x, 1e-6)
            den_safe = pt.maximum(den, 1e-6)
            V200 = V200_from_M200(M200)
            return (V200**2 / x_safe) * (num / den_safe)

        # ------------------------------------------
        # Vdrift
        # ------------------------------------------
        # Vdrift^2 = 2 * sigma_0^2 * (R / R_d)
        # Re is equivalent to the half-mass radius
        # Re = 1.68 * Rd.
        def v_drift_sq_profile(r, sigma_0, Re):
            R_d = Re / 1.678
            return 2.0 * (sigma_0 ** 2) * (r / R_d)

        # total v_rot^2 closure
        def v_rot_sq_profile(v_dm, v_star, v_drift):
            return v_dm**2 + v_star**2 - v_drift**2

        # Formula: v_obs = v_sys + v_rot * (sin(i) * cos(phi - phi_0))
        # Warning: The sign of the calculated velocity may be different from the observed velocity.
        def v_obs_project_profile(v_rot, v_sys, inc, phi_map):
            # phi_delta = (phi_map + pt.pi) % (2 * pt.pi)  # phi_map is (phi - phi_0)
            phi_delta = phi_map + pt.pi
            correction = pt.sin(inc) * pt.cos(phi_delta)
            v_obs = v_sys + v_rot * correction
            return v_obs

        # PyMC model
        with pm.Model() as model:
            # ------------------------------------------
            # prior distributions
            # ------------------------------------------

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # !!! important note !!!
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Use tight priors for Mstar and M200 (set sigma to a small value).
            # M200 and c are highly degenerate. A restrictive M200 prior is required to recover the M200-c relation,
            # the core focus of this study.
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            # Mstar prior
            # Mstar prior anchored to the observed Mstar, with a small sigma to enable recovery of the M200-c relation.
            Mstar_mu = float(Mstar_obs)
            Mstar_dex = 0.05  # dex
            log10_Mstar_mu = float(np.log10(Mstar_mu))  # log10 of
            log10_Mstar_sigma = Mstar_dex
            log10_Mstar_t = pm.Normal("log10_Mstar", mu=log10_Mstar_mu, sigma=log10_Mstar_sigma)
            Mstar_t = pm.Deterministic("Mstar", pt.pow(10.0, log10_Mstar_t))
            print(f"Mstar prior: mu={Mstar_mu:.2e} Msun, sigma={Mstar_dex:.3f} dex, range=[{10**(log10_Mstar_mu - log10_Mstar_sigma*3):.2e}, {10**(log10_Mstar_mu + log10_Mstar_sigma*3):.2e}] Msun")


            # M200 prior
            # M200 prior anchored to the SHMR estimate from the observed Mstar, with a small sigma to enable recovery of the M200-c relation.
            M200_mu = M200_shmr
            M200_dex = self.M200_dex  # dex
            log10_M200_mu = float(np.log10(M200_mu))
            log10_M200_sigma = M200_dex
            log10_M200_lower = max(float(np.log10(Mmin)), log10_M200_mu - log10_M200_sigma*3)
            log10_M200_upper = min(float(np.log10(Mmax)), log10_M200_mu + log10_M200_sigma*3)
            log10_M200_t = pm.TruncatedNormal("log10_M200",mu=log10_M200_mu,sigma=log10_M200_sigma, lower=log10_M200_lower, upper=log10_M200_upper)
            M200_t = pm.Deterministic("M200", pt.pow(10.0, log10_M200_t))
            print(f"M200 prior: mu={M200_mu:.2e} Msun, sigma={M200_dex:.3f} dex, range=[{10**log10_M200_lower:.2e}, {10**log10_M200_upper:.2e}] Msun")

            # c prior
            # Independent c: decoupled from M200, for empirical c-M200 fitting later.
            c_mu = 9.0  # typical c for M200~1e12 Msun halos at z~0
            c_dex = 0.5  # dex
            log_c_t = pm.Normal("log_c", mu=pt.log(c_mu), sigma=c_dex * pt.log(10))
            c_t = pm.Deterministic("c", pt.exp(log_c_t))
            log10_c_t = pm.Deterministic("log10_c", log_c_t / pt.log(10))
            log10_c_mu = float(np.log10(c_mu))
            log10_c_sigma = c_dex
            print(f"c prior: mu={c_mu:.2f}, sigma={c_dex:.3f} dex, range=[{c_mu * np.exp(-c_dex * np.log(10) * 3):.2f}, {c_mu * np.exp(c_dex * np.log(10) * 3):.2f}]")


            # sigma_0 prior:
            sigma_0_mu = 10.0
            sigma_0_dex = 0.20
            log_sigma_0_mu = float(np.log(sigma_0_mu))
            log_sigma_0_sigma = sigma_0_dex * float(np.log(10))
            sigma_0_t = pm.LogNormal("sigma_0", mu=log_sigma_0_mu, sigma=log_sigma_0_sigma)
            print(
                f"sigma_0 prior: mu={sigma_0_mu:.2f} km/s, sigma={sigma_0_dex:.3f} dex, "
                f"range=[{np.exp(log_sigma_0_mu - log_sigma_0_sigma*3):.2f}, "
                f"{np.exp(log_sigma_0_mu + log_sigma_0_sigma*3):.2f}] km/s"
            )

            # v_sys prior: normal prior around measured value
            v_sys_mu = vel_sys
            v_sys_sigma = 5.0
            v_sys_lower = v_sys_mu - 20
            v_sys_upper = v_sys_mu + 20
            v_sys_t = pm.TruncatedNormal("v_sys", mu=v_sys_mu, sigma=v_sys_sigma, lower=v_sys_lower, upper=v_sys_upper)
            print(f"v_sys prior: mu={v_sys_mu:.2f} km/s, sigma={v_sys_sigma:.2f} km/s, range=[{v_sys_lower:.2f}, {v_sys_upper:.2f}] km/s")

            # inc prior
            if self.inc_prior_enable:
                inc_t = pm.Normal("inc", mu=inc_rad, sigma=np.deg2rad(5.0))
                print(f"inc prior: Normal(mu={inc_rad:.2f} rad, sigma={np.deg2rad(5.0):.2f} rad) -> range=[{inc_rad - np.deg2rad(15):.2f}, {inc_rad + np.deg2rad(15):.2f}] rad")
            else:
                # Default: inc use the fixed photometric inclination
                # Notice: Infer inc may be degenerate with c
                inc_t = pm.Deterministic("inc", pt.as_tensor_variable(inc_rad))
                print(f"inc fixed at photometric value: {inc_rad:.2f} rad ({np.rad2deg(inc_rad):.2f} deg)")

            # Re prior
            Re_t_mu = float(Re_ref_kpc)
            Re_t_dex = 0.05  # dex
            log_Re_t_mu = float(np.log(Re_t_mu))
            log_Re_t_sigma = Re_t_dex * float(np.log(10))
            Re_t = pm.LogNormal('Re', mu=log_Re_t_mu, sigma=log_Re_t_sigma)
            print(f"Re prior: mu={Re_t_mu:.2f} kpc, sigma={Re_t_dex:.3f} dex, range=[{Re_t_mu*np.exp(-log_Re_t_sigma*3):.2f}, {Re_t_mu*np.exp(log_Re_t_sigma*3):.2f}] kpc")

            # Fix the Hernquist scale radius to the projected half-light relation
            # Re ~= 1.8153 * a so the bulge scale follows Re directly.
            a_t = pm.Deterministic("a", Re_t / 1.8153)
            print(f"a fixed by Re relation: a = Re / 1.8153, reference a≈{Re_t_mu / 1.8153:.2f} kpc")

            # f_bulge prior
            # Use NSA Sersic index n (direct morphological measurement of this galaxy)
            # as the prior center for bulge fraction, which is more accurate than a
            # population-level M*-B/T relation.
            #
            # Empirical logit-linear relation (Fisher & Drory 2008; Simard+2011):
            #   logit(f_bulge) ≈ k * (n - n0),  k≈1.2, n0≈2.5
            # n=1 (pure disk)  → f_bulge ≈ 0.16
            # n=2.5 (mixed)    → f_bulge ≈ 0.50
            # n=4 (pure bulge) → f_bulge ≈ 0.84
            sersic_n = self.drpall_util.get_sersic_n(self.PLATE_IFU)
            logit_f_mu = float(1.2 * (sersic_n - 2.5))
            logit_f_sigma = 0.2  # ~0.05 in f_bulge near center; tighter to resist c–f_bulge degeneracy
            print(f"f_bulge prior: Sersic n={sersic_n:.2f} \u2192 logit mu={logit_f_mu:.2f} \u2192 f_bulge\u2248{1/(1+np.exp(-logit_f_mu)):.2f}")

            # latent logit variable with fixed (numpy) prior center — no stochastic dependency,
            # avoids funnel geometry that degrades NUTS sampling efficiency.
            logit_f = pm.Normal("logit_f", mu=logit_f_mu, sigma=logit_f_sigma)
            # transform to (0,1)
            f_bulge_t = pm.Deterministic("f_bulge", pm.math.sigmoid(logit_f))

            # Intrinsic scatter prior:
            sigma_int_scale = float(np.nanmedian(sigma_meas_valid)) * 2
            sigma_int_t = pm.Exponential(
                "sigma_int",
                lam=1.0 / sigma_int_scale,
            )
            print(
                f"sigma_int prior: Exponential(mean={sigma_int_scale:.2f} km/s, "
                f"95% upper~{-sigma_int_scale * np.log(0.05):.2f} km/s)"
            )

            # nu prior for Student-t likelihood to be more robust to potential outliers in velocity measurements,
            #  which can otherwise bias the inference of DM parameters.
            nu_minus_t = pm.Gamma("nu_minus", alpha=2.0, beta=0.1)
            nu_t = pm.Deterministic("nu", nu_minus_t + 2.0)
            print("nu prior: Gamma(nu-2; alpha=2.0, beta=0.1) -> mean=20.0, mode=10.0, nu>2")

            # ------------------------------------------
            # deterministic relations
            # ------------------------------------------
            r = radius_valid  # numpy array
            v_star_t = pm.Deterministic("v_star", pt.sqrt(v_star_sq_profile(r, Mstar_t, Re_t, f_bulge_t, a_t)))
            v_dm_t = pm.Deterministic("v_dm", pt.sqrt(v_dm_sq_profile(r, M200_t, c_t)))
            v_drift_t = pm.Deterministic("v_drift", pt.sqrt(v_drift_sq_profile(r, sigma_0_t, Re_t)))
            v_rot_t = pm.Deterministic("v_rot", pt.sqrt(pt.maximum(1e-9, v_rot_sq_profile(v_dm_t, v_star_t, v_drift_t))))
            v_obs_model_t = pm.Deterministic("v_obs_model", v_obs_project_profile(v_rot_t, v_sys_t, inc_t, phi_map_valid))

            # ------------------------------------------
            # likelihood: observed rotation velocities
            # ------------------------------------------

            # Measurement error model (intrinsic-scatter form):
            # sigma_obs^2 = sigma_meas^2 + sigma_int^2
            sigma_meas_t = pt.as_tensor_variable(sigma_meas_valid)
            sigma_obs_t = pm.Deterministic("sigma_obs", pt.sqrt(sigma_meas_t**2 + sigma_int_t**2))
            # rc_like = pm.Normal("v_obs", mu=v_obs_model_t, sigma=sigma_obs_t, observed=vel_obs_valid)
            # student's t likelihood to be more robust to potential outliers in velocity measurements, which can otherwise bias the inference of DM parameters.
            rc_like = pm.StudentT("v_obs", mu=v_obs_model_t, sigma=sigma_obs_t, nu=nu_t, observed=vel_obs_valid)

            # ------------------------------------------
            # potential
            # ------------------------------------------
            # Down-weight small radii with a smooth logistic ramp in radius.
            w_rc_like = pt.as_tensor_variable(w_rc_like_np)
            pm.Potential("rc_like_weighted", pt.sum((w_rc_like - 1.0) * pm.logp(rc_like, vel_obs_valid)))

            # ------------------------------------------
            # sampling options & run
            # ------------------------------------------
            print(">>> Starting PyMC sampling (NUTS)... this may take time.\n")

            draws = 1000
            tune = 500
            chains = min(4, os.cpu_count())
            target_accept = 0.95

            if self.inf_debug:
                displaybar = True
                checks = True
            else:
                displaybar = False
                checks = False

            # MUST to use nutpie because v_star_sq_disk() uses Bessel functions which are not supported by numpyro/blackjax samplers.
            sampler = 'nutpie' #'nutpie', 'numpyro', 'blackjax'
            init = "jitter+adapt_full" # jitter+adapt_diag, jitter+adapt_full
            random_seed = 42

            if self.plot_enable:
                # sample prior predictive
                prior_var_names = ["Mstar", "M200", "c", "v_sys", "f_bulge", "sigma_0", "Re"]
                prior_sample_count = max(draws * chains, 2000)
                print(f"Sampling prior predictive with {prior_sample_count} samples...")
                prior_trace = pm.sample_prior_predictive(samples=prior_sample_count, var_names=prior_var_names, random_seed=random_seed,return_inferencedata=True)

            # sample posterior
            print(f"Sampling posterior with {sampler} sampler, init={init}, draws={draws}, tune={tune}, chains={chains}, target_accept={target_accept}")
            trace = pm.sample(init=init, draws=draws, tune=tune, chains=chains, cores=chains,
                              nuts_sampler=sampler, target_accept=target_accept,
                              progressbar=displaybar,
                              random_seed=random_seed,
                              return_inferencedata=True, compute_convergence_checks=checks)

            if self.inf_debug:
                print("\n\n")
                print(">>> Sampling completed.\n")

        # ------------------------------------------
        # postprocess
        # ------------------------------------------
        # summary with diagnostics
        # Keep both linear-scale derived parameters and the latent log-scale
        # variables in the summary so downstream analysis can use the log posteriors
        # directly without re-transforming deterministic samples.
        var_names = ["Mstar", "M200", "c", "v_sys", "inc", "f_bulge", "sigma_0", "Re", "sigma_int", "nu"]
        log_var_names = ["log10_Mstar", "log10_M200", "log10_c"]
        summary_var_names = var_names + log_var_names
        az_api = _get_arviz_api()
        _set_arviz_ci_defaults()

        summary = summary_with_compat(
            az_api.summary,
            trace,
            var_names=summary_var_names,
            round_to=3,
            stat_focus="median",
        )
        eti_low_col, eti_high_col = _get_summary_eti_columns(summary)
        ess_col = "ess_bulk" if "ess_bulk" in summary.columns else "ess_median" if "ess_median" in summary.columns else None

        for var in summary_var_names:
            r_hat = float(summary.loc[var, "r_hat"])
            if r_hat > INFER_RHAT_THRESHOLD:
                print(f"Warning: R-hat for variable {var} is {r_hat:.3f} > {INFER_RHAT_THRESHOLD}, indicating potential non-convergence.")
                success = False

            if ess_col is not None:
                ess_value = float(summary.loc[var, ess_col])
                if ess_value < INFER_ESS_THRESHOLD:
                    print(f"Warning: {ess_col} for variable {var} is {ess_value:.1f} < {INFER_ESS_THRESHOLD}, indicating potential sampling inefficiency.")
                    success = False

        # extract posterior samples
        posterior = _get_posterior_dataset(trace)
        flat_trace = posterior.stack(sample=("chain", "draw"))
        v_obs_samples = flat_trace["v_obs_model"].values
        v_dm_samples = flat_trace["v_dm"].values
        v_star_samples = flat_trace["v_star"].values
        v_drift_samples = flat_trace["v_drift"].values
        v_rot_samples = flat_trace["v_rot"].values
        sigma_obs_samples = flat_trace["sigma_obs"].values
        Mstar_samples = np.asarray(flat_trace["Mstar"].values, dtype=float).reshape(-1)
        M200_samples = np.asarray(flat_trace["M200"].values, dtype=float).reshape(-1)
        c_samples = np.asarray(flat_trace["c"].values, dtype=float).reshape(-1)
        Re_samples = np.asarray(flat_trace["Re"].values, dtype=float).reshape(-1)
        f_bulge_samples = np.asarray(flat_trace["f_bulge"].values, dtype=float).reshape(-1)
        sigma_0_samples = np.asarray(flat_trace["sigma_0"].values, dtype=float).reshape(-1)
        sigma_int_samples = np.asarray(flat_trace["sigma_int"].values, dtype=float).reshape(-1)
        log10_M200_samples = np.asarray(flat_trace["log10_M200"].values, dtype=float).reshape(-1)
        log10_c_samples = np.asarray(flat_trace["log10_c"].values, dtype=float).reshape(-1)
        nu_samples = np.asarray(flat_trace["nu"].values, dtype=float).reshape(-1)

        Mstar_median = float(summary.loc["Mstar", "median"])
        M200_median = float(summary.loc["M200", "median"])
        c_median = float(summary.loc["c", "median"])
        log10_Mstar_median = float(summary.loc["log10_Mstar", "median"])
        log10_M200_median = float(summary.loc["log10_M200", "median"])
        log10_c_median = float(summary.loc["log10_c", "median"])
        sigma0_median = float(summary.loc["sigma_0", "median"])
        v_sys_median = float(summary.loc["v_sys", "median"])
        inc_median_rad = float(summary.loc["inc", "median"])
        inc_median_deg = float(np.rad2deg(inc_median_rad))
        Re_median = float(summary.loc["Re", "median"])
        f_bulge_median = float(summary.loc["f_bulge", "median"])
        sigma_int_median = float(summary.loc["sigma_int", "median"])
        nu_median = float(summary.loc["nu", "median"])

        v_obs_median = np.nanmedian(v_obs_samples, axis=1)
        v_rot_median = np.nanmedian(v_rot_samples, axis=1)
        v_star_median = np.nanmedian(v_star_samples, axis=1)
        v_dm_median = np.nanmedian(v_dm_samples, axis=1)
        v_drift_median = np.nanmedian(v_drift_samples, axis=1)
        sigma_obs_median = np.nanmedian(sigma_obs_samples, axis=1)

        V200_calc = self._calc_V200_from_M200(M200_median, z)
        r200_calc = self._calc_r200_from_V200(V200_calc, z)
        c_calc = c_from_M200(M200_median, h=H_ACTUAL)

        samples_2d = np.vstack([log10_M200_samples, log10_c_samples]).T

        # Gaussian Mixture fit for potentially banana-shaped posterior (log10 M200, log10 c)
        gmm_params = _fit_log10_mc_gmm(samples_2d=samples_2d, max_components=3, random_state=42)

        posterior_samples = {
            'log10_M200_samples': log10_M200_samples,
            'log10_c_samples': log10_c_samples,
        }
        # ------------------------------------------
        # posterior predictive checks (dev_ppc_p)
        # ------------------------------------------
        # Prepare model samples (n_samples, n_points)
        v_obs_model = v_obs_samples.T
        sigma_obs = sigma_obs_samples.T
        nu = nu_samples

        rng_ppc = np.random.default_rng(random_seed)
        student_t_df = np.broadcast_to(np.asarray(nu[:, None], dtype=float), v_obs_model.shape)
        v_obs_ppc = v_obs_model + sigma_obs * rng_ppc.standard_t(student_t_df)

        # Align samples
        n_samp = min(v_obs_ppc.shape[0], v_obs_model.shape[0], sigma_obs.shape[0], nu.shape[0])
        y_rep = v_obs_ppc[:n_samp, :]
        mu = v_obs_model[:n_samp, :]
        sigma = sigma_obs[:n_samp, :]
        nu = np.asarray(nu[:n_samp], dtype=float)

        # Mask invalid points
        valid_cols = np.isfinite(vel_obs_valid) & np.all(np.isfinite(sigma) & (sigma > 0), axis=0)
        sigma_meas_valid_cols = sigma_meas_valid[valid_cols]
        weight_valid = w_rc_like_np[valid_cols]

        if n_samp == 0 or not np.any(valid_cols):
            dev_ppc_p = np.nan
            ppc_hdi_value_coverage = np.nan
            ppc_hdi_overlap = np.nan
        else:
            # Calculate weighted Student-t deviance and PPC coverage metrics using the
            # same pointwise likelihood family and radial weights as the fitted model.
            y_obs_valid = vel_obs_valid[valid_cols]
            y_rep_valid = y_rep[:, valid_cols]
            mu_valid = mu[:, valid_cols]
            sigma_valid = sigma[:, valid_cols]
            nu_valid = nu[:, None]

            def _student_t_logpdf(y_values: np.ndarray, mu_values: np.ndarray) -> np.ndarray:
                z = (y_values - mu_values) / sigma_valid
                return (
                    gammaln((nu_valid + 1.0) / 2.0)
                    - gammaln(nu_valid / 2.0)
                    - 0.5 * (np.log(np.pi) + np.log(nu_valid))
                    - np.log(sigma_valid)
                    - ((nu_valid + 1.0) / 2.0) * np.log1p((z ** 2) / nu_valid)
                )

            logp_obs = _student_t_logpdf(y_obs_valid[None, :], mu_valid)
            logp_rep = _student_t_logpdf(y_rep_valid, mu_valid)

            dev_obs = -2.0 * np.sum(weight_valid[None, :] * logp_obs, axis=1)
            dev_rep = -2.0 * np.sum(weight_valid[None, :] * logp_rep, axis=1)
            dev_ppc_p = float(np.mean(dev_rep > dev_obs))

            v_obs_ppc_hdi = _calc_eti_from_sample_matrix(y_rep_valid.T, prob=HDI_PROB2)
            v_obs_hdi_low = v_obs_ppc_hdi[:, 0]
            v_obs_hdi_high = v_obs_ppc_hdi[:, 1]
            ppc_value_mask = (
                np.isfinite(y_obs_valid) &
                np.isfinite(v_obs_hdi_low) &
                np.isfinite(v_obs_hdi_high) &
                (y_obs_valid >= v_obs_hdi_low) &
                (y_obs_valid <= v_obs_hdi_high)
            )
            ppc_overlap_mask = _calc_interval_overlap_mask(
                y_obs_valid,
                sigma_meas_valid_cols,
                v_obs_hdi_low,
                v_obs_hdi_high,
                sigma_scale=PPC_MEAS_SIGMA_SCALE,
            )
            ppc_hdi_value_coverage = float(np.mean(ppc_value_mask.astype(float))) if ppc_value_mask.size > 0 else np.nan
            ppc_hdi_overlap = float(np.mean(ppc_overlap_mask.astype(float))) if ppc_overlap_mask.size > 0 else np.nan

        # ------------------------------------------
        # residuals
        # ------------------------------------------
        mask = np.isfinite(vel_obs_valid) & np.isfinite(v_obs_median)
        res_obs_median = vel_obs_valid - v_obs_median
        res_norm_median = np.full_like(res_obs_median, np.nan, dtype=float)
        res_norm_median[mask] = res_obs_median[mask] / sigma_obs_median[mask]
        rmse_median = float(np.sqrt(np.mean(res_obs_median[mask]**2)))
        nrmse_median = float(rmse_median / np.mean(np.abs(vel_obs_valid[mask])))

        # ------------------------------------------
        # Recalculate Reduced Chi2 for the median fit
        # ------------------------------------------
        # parameters in model: M200, c, sigma_0, v_sys, inc, phi_delta, Re, f_bulge, a, sigma_obs
        params_num = 10
        dof = int(max(np.sum(valid_cols) - params_num, 1))

        # reduced chi2 for the median fit
        chi2_median = np.sum(res_norm_median[mask]**2)
        redchi_median = float(chi2_median / dof)

        # Get the correlation between variables c and M200 from the posterior samples
        c_m200_corr = float(np.corrcoef(c_samples, M200_samples)[0, 1])

        # ---------------------
        # Inference summary info
        # ---------------------
        if self.inf_debug:
            print(f"\n------------ Infer Dark Matter NFW ({self.PLATE_IFU}) ------------")
            print("--- Summary (median + HDI) ---")
            summary_cols = ["median", eti_low_col, eti_high_col, "r_hat"]
            if ess_col is not None:
                summary_cols.append(ess_col)
            summary_display = summary[summary_cols].copy()
            print(summary_display)
            print("--- Expectation ---")
            print(f"Mstar Expect        : {Mstar_obs:.3e} Msun")
            print(f"M200 Expect         : {M200_shmr:.3e} Msun")
            print(f"--- median estimates ---")
            print(f" Median Mstar       : {Mstar_median:.3e} Msun, ETI=[{float(summary.loc['Mstar', eti_low_col]):.3e}, {float(summary.loc['Mstar', eti_high_col]):.3e}]")
            print(f" Median M200        : {M200_median:.3e} Msun, ETI=[{float(summary.loc['M200', eti_low_col]):.3e}, {float(summary.loc['M200', eti_high_col]):.3e}]")
            print(f" Median c           : {c_median:.3f}, ETI=[{float(summary.loc['c', eti_low_col]):.3f}, {float(summary.loc['c', eti_high_col]):.3f}]")
            print(f" Median sigma_0     : {sigma0_median:.3f} km/s, ETI=[{float(summary.loc['sigma_0', eti_low_col]):.3f}, {float(summary.loc['sigma_0', eti_high_col]):.3f}]")
            print(f" Median v_sys       : {v_sys_median:.3f} km/s, ETI=[{float(summary.loc['v_sys', eti_low_col]):.3f}, {float(summary.loc['v_sys', eti_high_col]):.3f}]")
            print(f" Median inc         : {inc_median_deg:.3f} deg, ETI=[{float(np.rad2deg(summary.loc['inc', eti_low_col])):.3f}, {float(np.rad2deg(summary.loc['inc', eti_high_col])):.3f}] (prior: {np.degrees(inc_rad):.3f} deg)")
            print(f" Median Re          : {Re_median:.3f} kpc, ETI=[{float(summary.loc['Re', eti_low_col]):.3f}, {float(summary.loc['Re', eti_high_col]):.3f}]")
            print(f" Median f_bulge     : {f_bulge_median:.3f}, ETI=[{float(summary.loc['f_bulge', eti_low_col]):.3f}, {float(summary.loc['f_bulge', eti_high_col]):.3f}]")
            print(f" Median sigma_int   : {sigma_int_median:.3f} km/s, ETI=[{float(summary.loc['sigma_int', eti_low_col]):.3f}, {float(summary.loc['sigma_int', eti_high_col]):.3f}]")
            print(f" Median nu          : {nu_median:.3f}, ETI=[{float(summary.loc['nu', eti_low_col]):.3f}, {float(summary.loc['nu', eti_high_col]):.3f}]")
            print(f"--- caculate ---")
            print(f" Calc: V200         : {V200_calc:.3f} km/s")
            print(f" Calc: r200         : {r200_calc:.3f} kpc")
            print(f" Calc: c            : {c_calc:.3f}")
            print(f" Calc: v_sys        : {vel_sys:.3f} km/s")
            print(f" Calc: inc (prior)  : {np.degrees(inc_rad):.3f} deg  |  Posterior: {inc_median_deg:.3f} deg (delta={inc_median_deg - np.degrees(inc_rad):.3f} deg)")
            print(f" Stellar Mass       : {Mstar_obs:.3e} Msun")
            print("--- diagnostics ---")
            print(f" Reduced Chi (median): {redchi_median:.3f}")
            print(f" NRMSE (Median)     : {nrmse_median:.3f}")
            print(f" Weighted Student-t PPC p-val : {dev_ppc_p:.3f}")
            print(f" PPC Value Coverage : {ppc_hdi_value_coverage:.3f} ({HDI_PROB2:.0%} ETI)")
            print(f" PPC Overlap Rate   : {ppc_hdi_overlap:.3f} (threshold={PPC_HDI_OVERLAP_THRESHOLD:.3f}, obs ±{PPC_MEAS_SIGMA_SCALE:.1f}σ)")
            print("------------------------------------------------------------\n")
            # diag_samples = [np.asarray(flat_trace[var].values).reshape(-1) for var in summary_var_names]
            # corr = np.corrcoef(diag_samples)
            # print("Correlation Matrix")
            # header = "\t" + " ".join([f"{v:>10s}" for v in summary_var_names])
            # print(header)
            # for i, var in enumerate(summary_var_names):
            #     row = " ".join([f"{corr[i, j]:10.3f}" for j in range(len(summary_var_names))])
            #     print(f"{var:>10s} {row}")
            # print("------------------------------------------------------------\n")


        # ---------------------
        # plot
        # ---------------------
        if self.plot_enable:
            plot_var_names = ["Mstar", "M200", "c", "v_sys", "inc", "f_bulge", "sigma_0", "Re"]
            plt.rcParams['pdf.fonttype'] = 42
            plt.rcParams['ps.fonttype'] = 42
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
            os.makedirs(self.output_dir, exist_ok=True)

            # --------------
            # trace plots
            # --------------
            trace_axes = az.plot_trace(trace, var_names=plot_var_names)
            trace_axes = np.asarray(trace_axes, dtype=object)
            trace_fig = trace_axes.flat[0].figure
            trace_fig.set_size_inches(12, 10)
            trace_fig.savefig(
                os.path.join(self.output_dir, f"{self.PLATE_IFU}_posterior_trace_plot.png"),
                dpi=300,
                bbox_inches='tight',
            )
            trace_fig.savefig(
                os.path.join(self.output_dir, f"{self.PLATE_IFU}_posterior_trace_plot.pdf"),
                format='pdf',
                bbox_inches='tight',
                transparent=True,
            )

            az.rcParams['plot.max_subplots'] = 100

            # --------------
            # corner plot all variables
            # --------------
            pair_axes = az_api.plot_pair(trace, var_names=plot_var_names, kind=['kde'], marginals=True,
                                         marginal_kwargs={"kind": "hist", "hist_kwargs": {"bins": 30, "histtype": "step", "linewidth": 1.5, "density": True}},
                                         kde_kwargs={"hdi_probs": [HDI_PROB1, HDI_PROB2]},
                                         point_estimate=None, textsize=8, divergences=False)
            _annotate_pair_marginals(pair_axes, flat_trace, plot_var_names, title_fontsize=9, plot_median_line=True)
            pair_axes_array = np.asarray(pair_axes, dtype=object)
            for ax in pair_axes_array.flat:
                if ax is not None:
                    ax.set_xticks([])
                    ax.set_yticks([])
            pair_fig = pair_axes_array.flat[0].figure
            pair_fig.set_size_inches(12, 10)
            # pair_fig.suptitle(f"{self.PLATE_IFU} Posterior Pair Plot", fontsize=12, y=0.98)
            pair_fig.savefig(os.path.join(self.output_dir, f"{self.PLATE_IFU}_posterior_pair_plot.png"), dpi=300, bbox_inches='tight')
            pair_fig.savefig(os.path.join(self.output_dir, f"{self.PLATE_IFU}_posterior_pair_plot.pdf"), format='pdf', bbox_inches='tight', transparent=True)



            M200_c_var = ["M200", "c"]
            # --------------
            # corner plot M200 and c
            # --------------
            pair_m200_c_axes = az_api.plot_pair(trace, var_names=M200_c_var, kind=['kde'], marginals=True,
                                                marginal_kwargs={"kind": "hist", "hist_kwargs": {"bins": 30, "histtype": "step", "linewidth": 1.5, "density": True}},
                                                kde_kwargs={"hdi_probs": [HDI_PROB1, HDI_PROB2]},
                                                point_estimate=None, textsize=10, divergences=False)
            pair_m200_c_axes_array = np.asarray(pair_m200_c_axes, dtype=object)
            for ax in pair_m200_c_axes_array.flat:
                if ax is not None:
                    ax.set_xticks([])
                    ax.set_yticks([])
            pair_m200_c_fig = pair_m200_c_axes_array.flat[0].figure
            pair_m200_c_fig.set_size_inches(8, 8)
            pair_m200_c_fig.subplots_adjust(left=0.10, right=0.98, bottom=0.10, top=0.90, wspace=0.05, hspace=0.05)
            # pair_m200_c_fig.suptitle(f"{self.PLATE_IFU} Posterior Pair Plot (M200 & c)", fontsize=12, y=0.98)
            pair_m200_c_fig.savefig(os.path.join(self.output_dir, f"{self.PLATE_IFU}_posterior_m200_c_pair_plot.png"), dpi=300, bbox_inches='tight')
            pair_m200_c_fig.savefig(os.path.join(self.output_dir, f"{self.PLATE_IFU}_posterior_m200_c_pair_plot.pdf"), format='pdf', bbox_inches='tight', transparent=True)

            # --------------
            # custom posterior plot for M200 and c with multiple HDI intervals and legend
            # to instead of plot_posterior
            # --------------
            kde_fig, kde_axes = plt.subplots(1, len(M200_c_var), figsize=(10, 4))
            COLOR_M200 = '#D55E00'
            COLOR_C = '#0072B2'
            posterior_plot_specs = [
                ("M200", M200_samples, COLOR_M200),
                ("c", c_samples, COLOR_C),
            ]
            for ax, (plot_title, plot_samples, plot_color) in zip(np.atleast_1d(kde_axes), posterior_plot_specs):
                plot_posterior_1d_hdi(
                    plot_samples,
                    title=plot_title,
                    base_color=plot_color,
                    ax=ax,
                    hdi_probs=(HDI_PROB1, HDI_PROB2),
                    show_legend=True,
                    show_interval_bars=False,
                )
            # kde_fig.suptitle(f"{self.PLATE_IFU} Posterior KDE", fontsize=12)
            kde_fig.tight_layout(rect=[0, 0, 1, 0.96])
            kde_fig.savefig(os.path.join(self.output_dir, f"{self.PLATE_IFU}_posterior_m200_c_kde_plot.png"), dpi=300, bbox_inches='tight')
            kde_fig.savefig(os.path.join(self.output_dir, f"{self.PLATE_IFU}_posterior_m200_c_kde_plot.pdf"), format='pdf', bbox_inches='tight', transparent=True)

            # --------------
            # prior and posterior comparison plot
            # --------------
            prior_dataset = _get_prior_dataset(prior_trace)
            posterior_dataset = _get_posterior_dataset(trace)
            compare_fig, compare_axes = plt.subplots(1, len(M200_c_var), figsize=(10, 4))
            az_api.plot_density(
                [prior_dataset, posterior_dataset],
                data_labels=["Prior 95% HDI Distribution", "Posterior 95% HDI Distribution"],
                var_names=M200_c_var,
                ax=np.atleast_1d(compare_axes),
                point_estimate=None,
                hdi_prob=HDI_PROB2,
                shade=0.15,
                colors=["#999999", "#0072B2"],
                outline=True,
                textsize=8,
            )
            for axis, var_name in zip(np.atleast_1d(compare_axes), M200_c_var):
                axis.set_title(f"{var_name} Prior vs Posterior")
                axis.set_xlabel(var_name)
            # compare_fig.suptitle(f"{self.PLATE_IFU} Prior vs Posterior", fontsize=12)
            compare_fig.tight_layout(rect=[0, 0, 1, 0.96])
            compare_fig.savefig(os.path.join(self.output_dir, f"{self.PLATE_IFU}_prior_posterior_compare_plot.png"), dpi=300, bbox_inches='tight')
            compare_fig.savefig(os.path.join(self.output_dir, f"{self.PLATE_IFU}_prior_posterior_compare_plot.pdf"), format='pdf', bbox_inches='tight', transparent=True)

            plt.show()
        # ---------------------
        # plot end
        # ---------------------

        # check the inference success
        avg_v_rot = float(np.nanmean(v_rot_median))
        avg_v_dm = float(np.nanmean(v_dm_median))
        avg_v_star = float(np.nanmean(v_star_median))
        avg_v_drift = float(np.nanmean(v_drift_median))
        if avg_v_rot <= avg_v_dm:
            print("Warning: Inferred rotation velocity is less than or equal to dark matter velocity on average. Inference may have failed.")
            success = False
        if avg_v_rot <= avg_v_star:
            print("Warning: Inferred rotation velocity is less than or equal to stellar velocity on average. Inference may have failed.")
            success = False

        if self.inf_debug:
            frac_star = float(np.mean(v_rot_median <= v_star_median))
            frac_dm = float(np.mean(v_rot_median <= v_dm_median))
            print("Diagnostics: v components summary")
            print(f"  avg(v_rot)={avg_v_rot:.3f}, avg(v_star)={avg_v_star:.3f}, avg(v_dm)={avg_v_dm:.3f}, avg(v_drift)={avg_v_drift:.3f}")
            print(f"  frac(v_rot<=v_star)={frac_star:.3f}, frac(v_rot<=v_dm)={frac_dm:.3f}")

        plot_radius = radius_fit

        plot_component_samples = _calc_velocity_component_samples_on_grid(
            plot_radius,
            Hz,
            Mstar_samples,
            M200_samples,
            c_samples,
            Re_samples,
            f_bulge_samples,
            sigma_0_samples,
        )
        v_rot_plot_samples = plot_component_samples["v_rot"]
        v_dm_plot_samples = plot_component_samples["v_dm"]
        v_star_plot_samples = plot_component_samples["v_star"]
        v_drift_plot_samples = plot_component_samples["v_drift"]

        v_rot_plot_median = np.nanmedian(v_rot_plot_samples, axis=1)
        v_dm_plot_median = np.nanmedian(v_dm_plot_samples, axis=1)
        v_star_plot_median = np.nanmedian(v_star_plot_samples, axis=1)
        v_drift_plot_median = np.nanmedian(v_drift_plot_samples, axis=1)

        sigma_meas_rot = float(np.nanmedian(sigma_meas_valid)) if sigma_meas_valid.size > 0 else 0.0
        deproj_scale = max(float(np.sin(abs(inc_rad))), 0.25)
        sigma_rot_curve_samples = np.sqrt(sigma_int_samples**2 + sigma_meas_rot**2) / deproj_scale
        rng_rot_pp = np.random.default_rng(314159)
        v_rot_pp_samples = v_rot_plot_samples + rng_rot_pp.normal(
            loc=0.0,
            scale=sigma_rot_curve_samples[np.newaxis, :],
            size=v_rot_plot_samples.shape,
        )
        v_rot_eti = _calc_eti_from_sample_matrix(v_rot_pp_samples, prob=HDI_PROB2)

        plot_result = {
            'radius': plot_radius,
            'v_rot': v_rot_plot_median,
            'v_rot_eti_low': v_rot_eti[:, 0],
            'v_rot_eti_high': v_rot_eti[:, 1],
            'v_rot_samples': v_rot_plot_samples,
            'v_dm': v_dm_plot_median,
            'v_star': v_star_plot_median,
            'v_drift': v_drift_plot_median,
            'sigma_obs': sigma_obs_median,
        }

        # Compute mean and covariance of the core posterior samples (in log10 space)
        # for hierarchical modeling / inference of M200 and concentration c.
        inf_params = {
            'result': 'success' if success else 'failure',
            'sersic_n': sersic_n,
            'Mstar': Mstar_median,
            'Mstar_eti_low': float(summary.loc['Mstar', eti_low_col]),
            'Mstar_eti_high': float(summary.loc['Mstar', eti_high_col]),
            'log10_Mstar': log10_Mstar_median,
            'log10_Mstar_eti_low': float(summary.loc['log10_Mstar', eti_low_col]),
            'log10_Mstar_eti_high': float(summary.loc['log10_Mstar', eti_high_col]),
            'M200': M200_median,
            'M200_eti_low': float(summary.loc['M200', eti_low_col]),
            'M200_eti_high': float(summary.loc['M200', eti_high_col]),
            'log10_M200': log10_M200_median,
            'log10_M200_eti_low': float(summary.loc['log10_M200', eti_low_col]),
            'log10_M200_eti_high': float(summary.loc['log10_M200', eti_high_col]),
            'c': c_median,
            'c_eti_low': float(summary.loc['c', eti_low_col]),
            'c_eti_high': float(summary.loc['c', eti_high_col]),
            'log10_c': log10_c_median,
            'log10_c_eti_low': float(summary.loc['log10_c', eti_low_col]),
            'log10_c_eti_high': float(summary.loc['log10_c', eti_high_col]),
            'sigma_int': sigma_int_median,
            'sigma_int_eti_low': float(summary.loc['sigma_int', eti_low_col]),
            'sigma_int_eti_high': float(summary.loc['sigma_int', eti_high_col]),
            'c_M200_corr': c_m200_corr,
            'log10_gmm_source': gmm_params['source'],
            'log10_gmm_n_components': int(gmm_params['n_components']),
            'log10_gmm_weights': gmm_params['weights'],
            'log10_gmm_means': gmm_params['means'],
            'log10_gmm_covariances': gmm_params['covariances'],
            'log10_gmm_bic': gmm_params['bic'],
            'log10_gmm_bic_by_n': gmm_params['bic_by_n'],
            'log10_M200_prior_mu': log10_M200_mu,
            'log10_M200_prior_sigma': log10_M200_sigma,
            'log10_M200_prior_lower': log10_M200_lower,
            'log10_M200_prior_upper': log10_M200_upper,
            'log10_c_prior_mu': log10_c_mu,
            'log10_c_prior_sigma': log10_c_sigma,
            'posterior_sample_count': int(len(M200_samples)),
            'nrmse': nrmse_median,
            'redchi': redchi_median,
            'dev_ppc_p': dev_ppc_p,
            'PPC_ETI_PROB': f"{HDI_PROB2:.3f}",
            'PPC_ETI_VALUE_COVERAGE': f"{ppc_hdi_value_coverage:.3f}",
            'PPC_ETI_OVERLAP': f"{ppc_hdi_overlap:.3f}",
        }

        return success, plot_result, inf_params, posterior_samples


    ################################################################################
    # public methods
    ################################################################################
    def set_PLATE_IFU(self, PLATE_IFU: str) -> None:
        self.PLATE_IFU = PLATE_IFU
        return

    def set_plot_enable(self, plot_enable: bool, output_dir: str) -> None:
        self.plot_enable = plot_enable
        self.output_dir = output_dir if plot_enable else None
        return

    def set_inf_debug(self, inf_debug: bool) -> None:
        self.inf_debug = inf_debug
        return

    def set_M200_prior(self, M200_dex: float) -> None:
        self.M200_dex = M200_dex
        return

    def set_inc_prior(self, enable: bool) -> None:
        self.inc_prior_enable = enable
        return

    def set_down_weight_inner(self, r0_frac: float) -> None:
        self.r0_frac = r0_frac
        return

    def inf_dm_nfw(self, vel_param: dict, radius_fit: np.ndarray=None) -> tuple:
        return self._inf_dm_nfw_pymc(vel_param, radius_fit=radius_fit)

def _calc_velocity_component_samples_on_grid(
    radius_grid: np.ndarray,
    Hz: float,
    Mstar_samples: np.ndarray,
    M200_samples: np.ndarray,
    c_samples: np.ndarray,
    Re_samples: np.ndarray,
    f_bulge_samples: np.ndarray,
    sigma_0_samples: np.ndarray,
) -> dict[str, np.ndarray]:
    radius_grid = np.asarray(radius_grid, dtype=float).reshape(-1)
    if radius_grid.size == 0:
        raise ValueError("radius_grid must contain at least one value")

    sample_matrix = np.column_stack([
        np.asarray(Mstar_samples, dtype=float).reshape(-1),
        np.asarray(M200_samples, dtype=float).reshape(-1),
        np.asarray(c_samples, dtype=float).reshape(-1),
        np.asarray(Re_samples, dtype=float).reshape(-1),
        np.asarray(f_bulge_samples, dtype=float).reshape(-1),
        np.asarray(sigma_0_samples, dtype=float).reshape(-1),
    ])
    finite_sample_mask = np.all(np.isfinite(sample_matrix), axis=1)
    if not np.any(finite_sample_mask):
        raise ValueError("No finite posterior samples available for velocity reconstruction")

    Mstar = sample_matrix[finite_sample_mask, 0][None, :]
    M200 = sample_matrix[finite_sample_mask, 1][None, :]
    c = sample_matrix[finite_sample_mask, 2][None, :]
    Re = sample_matrix[finite_sample_mask, 3][None, :]
    f_bulge = sample_matrix[finite_sample_mask, 4][None, :]
    sigma_0 = sample_matrix[finite_sample_mask, 5][None, :]

    radius_sign = np.sign(radius_grid)[:, None]
    radius_abs = np.abs(radius_grid)[:, None]
    radius_safe = np.maximum(radius_abs, 1e-6)
    Re_safe = np.maximum(Re, 1e-6)
    Rd = Re_safe / 1.678
    a = Re_safe / 1.8153
    MB = f_bulge * Mstar
    MD = (1.0 - f_bulge) * Mstar

    with np.errstate(all="ignore"):
        v_bulge_sq = (G_kpc_kms_Msun * MB * radius_safe) / np.maximum((radius_safe + a) ** 2, 1e-12)

        y = np.maximum(radius_safe / (2.0 * Rd), 1e-6)
        bessel_term = np.maximum(i0(y) * k0(y) - i1(y) * k1(y), 0.0)
        v_disk_sq = (2.0 * G_kpc_kms_Msun * MD / Rd) * (y ** 2) * bessel_term
        v_star_sq = np.maximum(v_bulge_sq + v_disk_sq, 0.0)

        r200 = np.maximum((G_kpc_kms_Msun * M200 / (100.0 * Hz ** 2)) ** (1.0 / 3.0), 1e-6)
        x = np.maximum(radius_safe / r200, 1e-6)
        cx = c * x
        num = np.log1p(cx) - cx / (1.0 + cx)
        den = np.maximum(np.log1p(c) - c / (1.0 + c), 1e-12)
        V200 = np.maximum((10.0 * G_kpc_kms_Msun * Hz * M200) ** (1.0 / 3.0), 1e-6)
        v_dm_sq = np.maximum((V200 ** 2 / x) * (num / den), 0.0)

        v_drift_sq = np.maximum(2.0 * (sigma_0 ** 2) * (radius_safe / Rd), 0.0)
        v_rot_sq = np.maximum(v_dm_sq + v_star_sq - v_drift_sq, 1e-9)

    return {
        "v_rot": radius_sign * np.sqrt(v_rot_sq),
        "v_dm": radius_sign * np.sqrt(v_dm_sq),
        "v_star": radius_sign * np.sqrt(v_star_sq),
        "v_drift": radius_sign * np.sqrt(v_drift_sq),
    }

class DmNfw:
    drpall_util: DrpallUtil
