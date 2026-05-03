"""Physical constants, colour palettes, default thresholds, and literature
reference parameters for the manga-dm pipeline.

All config-overridable **defaults** live here.  The active values (possibly
overridden by ``config.toml``) are obtained from ``src.config.settings``.

Usage::

    from src.config.constants import H0_PHYS, COLOR_DATA_POINTS
    from src.config import settings

    # Pure constants — never overridden
    c = H0_PHYS

    # Config-overridable — use settings *or* constants as fallback
    snr_min = settings.SNR_THRESHOLD
"""

from __future__ import annotations

import math

# ═══════════════════════════════════════════════════════════════════════════
# 1.  Physical constants
# ═══════════════════════════════════════════════════════════════════════════

#: Hubble constant (km/s/Mpc) adopted as physical for this project.
H0_PHYS: float = 67.4

#: Dimensionless Hubble parameter h = H0 / 100.
H_ACTUAL: float = 0.674

#: Hubble constant (km/s/Mpc) assumed by the MaNGA DAP for *all* spatial
#: quantities.  DAP radii are stored in h⁻¹ kpc with H0=100 convention,
#: so physical kpc = r_DAP / H_RATIO.
H0_MANGA: float = 100.0

#: Conversion factor from DAP h⁻¹ kpc to physical kpc.
#: r_phys = r_DAP_kpc / H_RATIO  (or equivalently r_DAP_kpc * (H0_MANGA / H0_PHYS)).
H_RATIO: float = H0_PHYS / H0_MANGA  # 0.674

#: Pivot mass (Msun / h) used in the c–M200 relation.
M_PIVOT_H_INV: float = 1.0e12

#: Matter density parameter Ω_m (flat ΛCDM).
OMEGA_M: float = 0.315

#: Dark-energy density parameter Ω_Λ.
OMEGA_L: float = 0.685

#: Arcseconds per radian.
ARCSEC_PER_RADIAN: float = 206265.0

# ── Moster+2013 stellar-to-halo mass relation (SHMR) ───────────────────

#: Characteristic halo mass (Msun) — converted from h⁻¹ Msun using H_ACTUAL.
MOSTER_M1: float = 10.0**11.59 / H_ACTUAL

#: Normalisation of the SHMR.
MOSTER_N: float = 0.0351

#: Low-mass slope of the SHMR.
MOSTER_BETA: float = 1.376

#: High-mass slope of the SHMR.
MOSTER_GAMMA: float = 0.608

# ── Structural conversion ratios ────────────────────────────────────────

#: Ratio of half-light radius Re to exponential-disk scale-length Rd.
#: Rd = Re / STRUCT_RD_FACTOR.
STRUCT_RD_FACTOR: float = 1.678

#: Ratio of half-light radius Re to Hernquist-bulge scale radius a.
#: a = Re / STRUCT_HERNQUIST_FACTOR.
STRUCT_HERNQUIST_FACTOR: float = 1.8153

# ═══════════════════════════════════════════════════════════════════════════
# 2.  c–M200 relation parameters (literature)
# ═══════════════════════════════════════════════════════════════════════════
# All use the pivot mass M_PIVOT_H_INV = 1e12 Msun/h.
#
# Form:  log10(c200) = log10(c0) + α × (log10(M200 / M_pivot))

# ── Dutton & Macciò 2014 (DM14): cosmological simulations ─────────────
LOG10_C0_DM14: float = 0.905
ALPHA_DM14: float = -0.101
LOG10_C_SIGMA_DM14: float = 0.11

# ── Li et al. 2020 (SPARC): late-type galaxies ────────────────────────
LOG10_C0_LI20: float = 0.84
LOG10_C0_SIGMA_LI20: float = 0.03
ALPHA_LI20: float = -0.06
ALPHA_SIGMA_LI20: float = 0.04
LOG10_C_SCATTER_LI20: float = 0.20

# ── Yasin et al. 2023 (HI): cold gas kinematics ───────────────────────
LOG10_C0_YASIN23: float = 0.91
LOG10_C0_SIGMA_YASIN23: float = 0.05
ALPHA_YASIN23: float = -0.11
ALPHA_SIGMA_YASIN23: float = 0.03
LOG10_C_SCATTER_YASIN23: float = 0.15

# ═══════════════════════════════════════════════════════════════════════════
# 3.  Prior defaults (population model)
# ═══════════════════════════════════════════════════════════════════════════

#: Prior mean for log10(c0) — default to DM14 value.
LOG10_C0_PRIOR_MEAN: float = LOG10_C0_DM14
#: Prior sigma for log10(c0).
LOG10_C0_PRIOR_SIGMA: float = 0.5

#: Prior mean for slope α.
ALPHA_PRIOR_MEAN: float = ALPHA_DM14
#: Prior sigma for slope α.
ALPHA_PRIOR_SIGMA: float = 0.3

#: Prior mean for log intrinsic scatter.
LOG_SIGMA_INT_PRIOR_MEAN: float = math.log(0.15)
#: Prior sigma for log intrinsic scatter.
LOG_SIGMA_INT_PRIOR_SIGMA: float = 0.8

#: Gamma prior shape for Student-t degrees-of-freedom ν.
NU_POP_PRIOR_ALPHA: float = 2.0
#: Gamma prior rate for Student-t degrees-of-freedom ν.
NU_POP_PRIOR_BETA: float = 0.1

#: Defensive importance-sampling mixing weight ε (prevents zero weights).
DEFENSIVE_IS_EPSILON: float = 0.1

# ── NFW single-galaxy prior hyper-parameters ────────────────────────────

#: Prior central value for log10(c200).
C_MU: float = 9.0
#: Prior width for log10(c200) in dex.
C_DEX: float = 0.5

#: Prior central value for σ₀ (km/s).
SIGMA_0_MU: float = 10.0
#: Prior width for σ₀ in dex.
SIGMA_0_DEX: float = 0.20

#: Prior sigma for systemic velocity v_sys (km/s).
V_SYS_SIGMA: float = 5.0

#: Prior width for stellar mass in dex.
M_STAR_DEX: float = 0.05

#: Prior width for effective radius Re in dex.
RE_T_DEX: float = 0.05

#: Prior sigma for bulge fraction logit.
LOGIT_F_SIGMA: float = 0.2

#: Inner-logistic half-weight radius fraction for DmNfw.
R0_FRAC: float = 0.3

#: Scale factor for M200 prior width relative to SHMR estimate.
M200_DEX: float = 0.15

# ═══════════════════════════════════════════════════════════════════════════
# 4.  Fit parameter bounds (lmfit)
# ═══════════════════════════════════════════════════════════════════════════

#: Asymptotic velocity Vc bounds (km/s).
VC_BOUNDS: tuple[float, float] = (20.0, 500.0)

#: Outer slope bounds (km/s/kpc).
S_OUT_BOUNDS: tuple[float, float] = (-10.0, 10.0)

#: Systemic velocity bounds (km/s).
VSYS_BOUNDS: tuple[float, float] = (-100.0, 100.0)

# ═══════════════════════════════════════════════════════════════════════════
# 5.  Quality-filter presets
# ═══════════════════════════════════════════════════════════════════════════

QUALITY_FILTER_PRESETS: dict[str, dict[str, float | bool]] = {
    "recommended": {
        "max_redchi": 3.0,
        "ppc_p_min": 0.05,
        "ppc_p_max": 0.95,
        "ppc_overlap_min": 0.5,
        "max_abs_c_m200_corr": 0.95,
    },
    "strict": {
        "max_redchi": 2.0,
        "ppc_p_min": 0.10,
        "ppc_p_max": 0.90,
        "ppc_value_coverage_min": 0.80,
        "ppc_overlap_min": 0.60,
        "max_abs_c_m200_corr": 0.90,
    },
}

# ═══════════════════════════════════════════════════════════════════════════
# 6.  Bit masks for target selection and DRP quality
# ═══════════════════════════════════════════════════════════════════════════

#: MNGTARG3 bits to *exclude* (bits 19, 20, 21, 27).
BIT_MASK_3_EXCLUDE: int = (1 << 19) | (1 << 20) | (1 << 21) | (1 << 27)

#: DRP3QUAL failure bits (bits 14, 30).
BIT_MASK_DRP_FAIL: int = (1 << 14) | (1 << 30)

# ═══════════════════════════════════════════════════════════════════════════
# 7.  SDSS DR17 data URLs
# ═══════════════════════════════════════════════════════════════════════════

MAPS_BASE_URL: str = (
    "https://data.sdss.org/sas/dr17/manga/spectro/analysis/v3_1_1/"
    "3.1.0/HYB10-MILESHC-MASTARHC2"
)
REDUX_BASE_URL: str = "https://data.sdss.org/sas/dr17/manga/spectro/redux/v3_1_1"
FIREFLY_BASE_URL: str = "https://data.sdss.org/sas/dr17/manga/spectro/firefly/v3_1_1"

# ═══════════════════════════════════════════════════════════════════════════
# 8.  Colour palette
# ═══════════════════════════════════════════════════════════════════════════

#: Model / reference curves.
COLOR_MODEL: str = "#000000"

#: Scatter / data points.
COLOR_DATA_POINTS: str = "#4D4D4D"

#: Bootstrap / posterior-sample traces.
COLOR_BOOTSTRAP_LINES: str = "#7F7F7F"

#: Sigma / uncertainty band light fill.
COLOR_SIGMA_BAND: str = "#D9D9D9"

#: HDI band medium fill.
COLOR_HDI_BAND: str = "#BDBDBD"

# ── Velocity component colours ──────────────────────────────────────────

#: Total rotation velocity V_tot.
COLOR_V_TOTAL: str = "#4D4D4D"

#: Stellar contribution V_star.
COLOR_V_STAR: str = "#0072B2"

#: Dark-matter contribution V_DM.
COLOR_V_DM: str = "#7F7F7F"

# ── Sample-class labels (colourblind-safe) ──────────────────────────────

#: High-Sérsic-n (early-type / bulge-dominated).
COLOR_HIGH_N: str = "#D55E00"

#: Low-Sérsic-n (late-type / disk-dominated).
COLOR_LOW_N: str = "#0072B2"

#: Posterior median / point estimate.
COLOR_POSTERIOR_MEDIAN: str = "#0072B2"

#: M200 posterior emphasis colour.
COLOR_M200: str = "#D55E00"

#: Concentration c posterior emphasis colour.
COLOR_C: str = "#0072B2"

# ── Literature-relation colours ─────────────────────────────────────────

#: Dutton & Macciò 2014 (DM14).
COLOR_DM14: str = "#4D4D4D"

#: Li et al. 2020 (LI20 / SPARC).
COLOR_LI20: str = "#009E73"

#: Yasin et al. 2023 (YASIN23 / HI).
COLOR_YASIN23: str = "#D55E00"

# ── Annotation colour ───────────────────────────────────────────────────

#: Dark red for parameter-text annotations.
COLOR_PARAM_TEXT: str = "#7A1E1E"

# ═══════════════════════════════════════════════════════════════════════════
# 9.  Test / default plate-IFU lists
# ═══════════════════════════════════════════════════════════════════════════

TEST_PLATE_IFUS: tuple[str, ...] = (
    "7443-12701",
    "7443-12703",
    "7443-12704",
    "7443-12705",
    "8081-12701",
    "8081-12702",
    "8081-12703",
    "8081-12704",
)

DEFAULT_PLATE_IFUS: tuple[str, ...] = (
    "8994-12701",
    "7977-3704",
    "9493-6101",
    "11743-9102",
)

#: Hard-coded filename for the plate-IFU list.
PLATES_FILENAME: str = "plateifus.txt"

#: Hard-coded filename for the rotation-curve parameter CSV.
VEL_ROT_PARAM_FILE: str = "vel_rot_param_all.csv"
