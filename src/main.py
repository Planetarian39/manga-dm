from pathlib import Path
import os
import multiprocessing
import re
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
import gc
import tomllib

# my imports
from util.maps_util import MapsUtil
from util.drpall_util import DrpallUtil
from util.fits_util import FitsUtil
from util.firefly_util import FireflyUtil
from util.plot_util import PlotUtil
from rc import RotCurve
from dm import DmNfw

COLOR_MODEL = 'black'
COLOR_DATA_POINTS = '#4D4D4D'
COLOR_BOOTSTRAP_LINES = '#7F7F7F'
COLOR_V_TOTAL = '#4D4D4D'
COLOR_V_STAR = '#0072B2'
COLOR_V_DM = '#7F7F7F'

# Load configuration file
with open("config.toml", "rb") as f:
    config = tomllib.load(f)
    if not config:
        raise ValueError("Error: config.toml file is empty")

# get settings from config
data_directory = config.get("file", {}).get("data_directory", "data")
result_directory = config.get("file", {}).get("result_directory", "results")
RC_PARAM_FILENAME = config.get("file", {}).get("rc_param_filename", "rc_param.csv")
NFW_PARAM_FILENAME = config.get("file", {}).get("nfw_param_cm200_filename", "nfw_param_cm200.csv")
NFW_SAMPLE_FILENAME = config.get("file", {}).get("nfw_param_cm200_sample_filename", "nfw_param_cm200_samples.nc")

INC_MIN = config.get("thresholds", {}).get("INC_MIN", 25.0)
INC_MAX = config.get("thresholds", {}).get("INC_MAX", 70.0)
VEL_OBS_COUNT_THRESHOLD = config.get("thresholds", {}).get("VEL_OBS_COUNT_THRESHOLD", 150)
RMAX_RT_FACTOR = config.get("thresholds", {}).get("RMAX_RT_FACTOR", 2)  # factor to determine maximum radius for fitting
PPC_HDI_VALUE_COVERAGE_THRESHOLD = config.get("thresholds", {}).get("PPC_HDI_VALUE_COVERAGE_THRESHOLD", 0.60)
PPC_HDI_OVERLAP_THRESHOLD = config.get("thresholds", {}).get("PPC_HDI_OVERLAP_THRESHOLD", 0.80)

# Set up data and result directories
root_dir = Path(__file__).resolve().parent.parent
data_dir = root_dir / data_directory
result_dir = data_dir / result_directory
data_dir.mkdir(parents=True, exist_ok=True)
result_dir.mkdir(parents=True, exist_ok=True)


fits_util = FitsUtil(data_dir)


def _resolve_result_dir(result_dir_override: str | Path | None = None) -> Path:
    if result_dir_override is None:
        return data_dir / result_directory

    resolved = Path(result_dir_override)
    if not resolved.is_absolute():
        resolved = root_dir / resolved
    return resolved


def _set_result_dir(result_dir_override: str | Path | None = None) -> Path:
    global result_dir

    result_dir = _resolve_result_dir(result_dir_override)
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir


_active_r0_frac: float | None = None
_active_m200_prior_dex: float | None = None
_active_inc_prior_enable: bool | None = None


def _set_r0_frac(r0_frac: float | None) -> float | None:
    global _active_r0_frac
    _active_r0_frac = None if r0_frac is None else float(r0_frac)
    return _active_r0_frac


def _set_m200_prior_dex(m200_prior_dex: float | None) -> float | None:
    global _active_m200_prior_dex
    _active_m200_prior_dex = None if m200_prior_dex is None else float(m200_prior_dex)
    return _active_m200_prior_dex


def _set_inc_prior_enable(enable: bool | None) -> bool | None:
    global _active_inc_prior_enable
    _active_inc_prior_enable = None if enable is None else bool(enable)
    return _active_inc_prior_enable


def _get_posterior_sample_output_path(output_file: Path, plate_ifu: str) -> Path:
    return output_file.with_name(f"{plate_ifu}_{output_file.name}")

# Store RC fit parameters as CSV file
def store_params_file(PLATE_IFU: str, fit_parameters: dict, filename:str):
    output_file = result_dir / filename

    if output_file.exists():
        try:
            all_fit_parameters = pd.read_csv(output_file, index_col=0).to_dict(orient='index')
        except pd.errors.EmptyDataError:
            all_fit_parameters = {}
    else:
        all_fit_parameters = {}

    # clean previous entry
    if PLATE_IFU in all_fit_parameters:
        del all_fit_parameters[PLATE_IFU]

    all_fit_parameters[PLATE_IFU] = fit_parameters

    df = pd.DataFrame.from_dict(all_fit_parameters, orient='index')
    df.rename_axis('PLATE_IFU', inplace=True)
    df.to_csv(output_file)
    return


def store_posterior_samples_file(PLATE_IFU: str, posterior_samples: dict, filename: str):
    output_file = result_dir / filename
    per_ifu_output_file = _get_posterior_sample_output_path(output_file, str(PLATE_IFU))

    log10_m200_samples_raw = posterior_samples.get("log10_M200_samples")
    log10_c_samples_raw = posterior_samples.get("log10_c_samples")

    if log10_m200_samples_raw is None or log10_c_samples_raw is None:
        legacy_m200_samples = posterior_samples.get("M200_samples")
        legacy_c_samples = posterior_samples.get("c_samples")
        if legacy_m200_samples is not None and legacy_c_samples is not None:
            log10_m200_samples_raw = np.log10(np.asarray(legacy_m200_samples, dtype=float))
            log10_c_samples_raw = np.log10(np.asarray(legacy_c_samples, dtype=float))
        else:
            log10_m200_samples_raw = []
            log10_c_samples_raw = []

    log10_m200_samples = np.asarray(log10_m200_samples_raw, dtype=float).reshape(-1)
    log10_c_samples = np.asarray(log10_c_samples_raw, dtype=float).reshape(-1)
    if len(log10_m200_samples) != len(log10_c_samples):
        raise ValueError(
            f"Posterior sample length mismatch for {PLATE_IFU}: "
            f"{len(log10_m200_samples)} != {len(log10_c_samples)}"
        )

    dataset = xr.Dataset(
        data_vars={
            "log10_M200_samples": (("sample",), log10_m200_samples.astype(np.float64, copy=False)),
            "log10_c_samples": (("sample",), log10_c_samples.astype(np.float64, copy=False)),
            "sample_count": np.array(len(log10_m200_samples), dtype=np.int32),
        },
        coords={
            "sample": np.arange(len(log10_m200_samples), dtype=np.int32),
        },
        attrs={
            "description": "Posterior log10-samples for NFW M200 and c for a single PLATE_IFU",
            "plate_ifu": str(PLATE_IFU),
            "storage_format": "per_ifu_netcdf",
        },
    )

    temp_output_file = per_ifu_output_file.with_name(f"{per_ifu_output_file.stem}.tmp{per_ifu_output_file.suffix}")
    dataset.to_netcdf(temp_output_file)
    dataset.close()
    os.replace(temp_output_file, per_ifu_output_file)
    return

def get_params_file(PLATE_IFU: str, filename:str):
    output_file = result_dir / filename

    if not output_file.exists():
        return None

    try:
        all_fit_parameters = pd.read_csv(output_file, index_col=0).to_dict(orient='index')
    except pd.errors.EmptyDataError:
        return None

    if PLATE_IFU in all_fit_parameters:
        return all_fit_parameters[PLATE_IFU]
    else:
        return None


def get_processed_plate_ifus(filename: str):
    output_file = result_dir / filename

    if not output_file.exists():
        return set()

    try:
        df = pd.read_csv(output_file, index_col=0)
    except pd.errors.EmptyDataError:
        return set()

    return {str(plate_ifu) for plate_ifu in df.index.tolist()}


def process_plate_ifu(PLATE_IFU, process_nfw: bool=True, debug: bool=False):
    firefly_util = None
    maps_util = None
    vel_rot = None
    dm_nfw = None

    vel_rot_param = get_params_file(PLATE_IFU, RC_PARAM_FILENAME)
    if not process_nfw and vel_rot_param is not None:
        if vel_rot_param['result'] != 'success':
            print(f"Velocity rotation fit previously failed for {PLATE_IFU}. Skipping processing.")
            return

    drpall_file = fits_util.get_drpall_file()
    firefly_file = fits_util.get_firefly_file()
    maps_file = fits_util.get_maps_file(PLATE_IFU, checksum=False, download=False)
    if maps_file is None:
        print(f"MAPS file for {PLATE_IFU} not found locally. Skipping processing.")
        return

    print(f"DRPALL file: {drpall_file}")
    print(f"FIREFLY file: {firefly_file}")
    print(f"MAPS file: {maps_file}")

    try:
        drpall_util = DrpallUtil(drpall_file)
        firefly_util = FireflyUtil(firefly_file)
        maps_util = MapsUtil(maps_file)
        plot_util = PlotUtil(fits_util)

        vel_rot = RotCurve(drpall_util, firefly_util, maps_util, plot_util=None)
        vel_rot.set_PLATE_IFU(PLATE_IFU)

        r_obs_map, V_obs_map, ivar_map, phi_map = vel_rot.get_vel_obs()
        gflux_map, _, _ = maps_util.get_eml_gflux_map()
        fwhm = maps_util.get_fwhm()
        pixel_scale = maps_util.get_pixel_scale()
        print(f"Pixel scale: {pixel_scale:.3f} arcsec/pixel, FWHM: {fwhm:.3f} arcsec")

        radius_fit = vel_rot.get_radius_fit(np.nanmax(r_obs_map), count=1000)

        vel_rot_filename = RC_PARAM_FILENAME

        #----------------------------------------------------------------------
        # RC fitting
        #----------------------------------------------------------------------
        print(f"## RC fitting {PLATE_IFU} ##")
        vel_param = {
            "radius_obs": r_obs_map,
            "vel_obs": V_obs_map,
            "ivar_obs": ivar_map,
            "phi_map": phi_map,
            "gflux_map": gflux_map,
        }
        success, plot_result, fit_params = vel_rot.fit_vel_rot(vel_param, radius_fit=radius_fit)

        if not success:
            if isinstance(fit_params, dict):
                fit_params['quality_pass'] = False
                fit_params['quality_fail_reasons'] = 'fit_failed'
                fit_params['quality_summary'] = 'fit_failed_before_quality_gate'
                store_params_file(PLATE_IFU, fit_params, filename=vel_rot_filename)
            print(f"Fitting rotational velocity failed for {PLATE_IFU}")
            return

        r_obs_map = plot_result['radius_obs']
        V_obs_map = plot_result['vel_obs']
        ivar_obs_map = plot_result['ivar_obs']
        r_rot_fit = plot_result['radius_rot']
        V_rot_fit = plot_result['vel_rot']
        stderr_rot_fit = plot_result['stderr_rot']
        print(f"Fitted rotation curve has radius range [{np.nanmin(r_rot_fit):.1f}, {np.nanmax(r_rot_fit):.1f}] kpc")
        print(f"Fitted rotation curve has velocity range [{np.nanmin(V_rot_fit):.1f}, {np.nanmax(V_rot_fit):.1f}] km/s")

        inc_rad_fit = float(fit_params['inc'])
        inc_deg_fit = float(np.degrees(inc_rad_fit))
        vel_sys_fit = float(fit_params['Vsys'])
        phi_delta_fit = float(fit_params['phi_delta'])
        Rmax = float(fit_params['Rmax'])
        Rt = float(fit_params['Rt'])

        # Filter fitting parameters
        data_count = np.sum(np.isfinite(V_obs_map))
        NRMSE = float(fit_params['NRMSE'])
        CHI_SQ_V = float(fit_params['CHI_SQ_V'])
        quality_gate = RotCurve.evaluate_fit_quality(fit_params, data_count)
        if isinstance(fit_params, dict):
            fit_params['inc_deg'] = f"{quality_gate['inc_deg']:.3f}"
            fit_params['Rmax_Rt_ratio'] = f"{quality_gate['rmax_rt_ratio']:.3f}"
            fit_params['quality_pass'] = bool(quality_gate['passed'])
            fit_params['quality_summary'] = quality_gate['summary']
        store_params_file(PLATE_IFU, fit_params, filename=vel_rot_filename)

        if not quality_gate['passed']:
            fail_reason_text = '; '.join(quality_gate['fail_reasons'])
            print(f"fitting results failure for {PLATE_IFU}: {quality_gate['summary']}, reasons: {fail_reason_text}, NRMSE(diag): {NRMSE:.3f}, CHI_SQ_V(diag): {CHI_SQ_V:.3f}")
            return

        if debug:
            r_disp_map, V_disp_map, _ = vel_rot.get_vel_obs_disp(inc_rad_fit, vel_sys_fit, phi_delta_fit)
            V_rot_hdi_low = plot_result.get('vel_rot_hdi_low')
            V_rot_hdi_high = plot_result.get('vel_rot_hdi_high')
            V_rot_samples = plot_result.get('vel_rot_samples')
            show_hdi = V_rot_hdi_low is not None and V_rot_hdi_high is not None

            plot_util.plot_rv_curves([
                {
                    'r_map': r_disp_map,
                    'V_map': V_disp_map,
                    'color': COLOR_DATA_POINTS,
                    'label': 'Observed data',
                },
                {
                    'r_map': r_rot_fit,
                    'V_map': V_rot_fit,
                    'V_lower': V_rot_hdi_low if show_hdi else None,
                    'V_upper': V_rot_hdi_high if show_hdi else None,
                    'color': COLOR_MODEL,
                    'linestyle': '-',
                    'alpha': 0.95,
                    'fill_alpha': 0.16,
                    'fill_label': 'Posterior HDI' if show_hdi else None,
                    'label': 'Inferred RC Median',
                },
            ], plateifu=f"{PLATE_IFU}", title="Rotation Curve Posterior-based Sample Selection", savedir=result_dir)

        print(f"Fitting successful for {PLATE_IFU}. Inc: {inc_deg_fit:.2f} deg, Vsys: {vel_sys_fit:.2f} km/s, phi_delta: {phi_delta_fit:.2f} deg, Rmax: {Rmax:.2f} arcsec, Rt: {Rt:.2f} arcsec")
        print("")


        #--------------------------------------------------------
        # DM NFW inference
        #--------------------------------------------------------
        if not process_nfw:
            return

        print(f"## DM NFW inferring {PLATE_IFU} ##")

        r_disp_map, V_disp_map, _ = vel_rot.get_vel_obs_disp(inc_rad=inc_rad_fit, vel_sys=vel_sys_fit, phi_delta=phi_delta_fit)

        dm_nfw = DmNfw(drpall_util)
        dm_nfw.set_PLATE_IFU(PLATE_IFU)
        dm_nfw.set_plot_enable(debug, output_dir=result_dir)
        dm_nfw.set_inf_debug(debug)
        if _active_m200_prior_dex is not None:
            dm_nfw.set_M200_prior(_active_m200_prior_dex)
        if _active_inc_prior_enable is not None:
            dm_nfw.set_inc_prior(_active_inc_prior_enable)
        if _active_r0_frac is not None:
            dm_nfw.set_down_weight_inner(_active_r0_frac)

        vel_param = {
            "radius_obs": r_obs_map,
            "vel_obs": V_obs_map,
            "ivar_obs": ivar_obs_map,
            "vel_sys": vel_sys_fit,
            "inc_rad": inc_rad_fit,
            "phi_map": phi_map,
        }


        success, plot_result, inf_params, posterior_samples = dm_nfw.inf_dm_nfw(vel_param=vel_param, radius_fit=radius_fit)
        store_params_file(PLATE_IFU, inf_params, filename=NFW_PARAM_FILENAME)
        store_posterior_samples_file(PLATE_IFU, posterior_samples, filename=NFW_SAMPLE_FILENAME)

        if not success:
            print(f"Inferring dark matter NFW failed for {PLATE_IFU}")
            return

        r_median = plot_result['radius']
        V_rot_median = plot_result['v_rot']
        V_rot_eti_low = plot_result.get('v_rot_eti_low')
        V_rot_eti_high = plot_result.get('v_rot_eti_high')
        V_rot_samples = plot_result.get('v_rot_samples')
        V_dm_median = plot_result['v_dm']
        V_star_median = plot_result['v_star']
        V_drift_median = plot_result['v_drift']

        print(f"V_obs_map shape: {V_disp_map.shape}, range: [{np.nanmin(V_disp_map):,.1f}, {np.nanmax(V_disp_map):,.1f}] km/s")
        print(f"V_obs_fitted shape: {V_rot_fit.shape}, range: [{np.nanmin(V_rot_fit):,.1f}, {np.nanmax(V_rot_fit):,.1f}] km/s")
        print(f"V_total_median shape: {V_rot_median.shape}, range: [{np.nanmin(V_rot_median):,.1f}, {np.nanmax(V_rot_median):,.1f}] km/s")
        print(f"V_dm_median shape: {V_dm_median.shape}, range: [{np.nanmin(V_dm_median):,.1f}, {np.nanmax(V_dm_median):,.1f}] km/s")
        print(f"V_star_median shape: {V_star_median.shape}, range: [{np.nanmin(V_star_median):,.1f}, {np.nanmax(V_star_median):,.1f}] km/s")
        print(f"V_drift_median shape: {V_drift_median.shape}, range: [{np.nanmin(V_drift_median):,.1f}, {np.nanmax(V_drift_median):,.1f}] km/s")


        # plot observed data and RC with posterior samples
        rv_plot_data = []
        if V_rot_samples is not None:
            V_rot_samples = np.asarray(V_rot_samples, dtype=float)
            if V_rot_samples.ndim == 2 and V_rot_samples.shape[1] > 0:
                sample_idx = np.linspace(0, V_rot_samples.shape[1] - 1, min(100, V_rot_samples.shape[1]), dtype=int)
                for line_idx, sample_id in enumerate(sample_idx):
                    rv_plot_data.append({
                        'r_map': r_median,
                        'V_map': V_rot_samples[:, sample_id],
                        'color': COLOR_BOOTSTRAP_LINES,
                        'linestyle': '-',
                        'alpha': 0.10,
                        'label': 'RC Posterior Samples' if line_idx == 0 else None,
                    })
        rv_plot_data.extend([
            {'r_map': r_disp_map, 'V_map': V_disp_map, 'color': COLOR_DATA_POINTS, 'linestyle': None, 'size': 5, 'label': 'Observed rotation velocity data'},
            {'r_map': r_median, 'V_map': V_rot_median, 'color': COLOR_V_TOTAL, 'linestyle': '-', 'label': 'Inferred rotation curve median'},
        ])
        plot_util.plot_rv_curves(rv_plot_data, plateifu=f"{PLATE_IFU}", title="Rotation Curve posterior samples", savedir=result_dir, savefilename=f"{PLATE_IFU}_nfw_rv_posterior_samples")

        # plot observed data and model RC with ETI
        print(f"V_rot_eti_low shape: {V_rot_eti_low.shape if V_rot_eti_low is not None else None}, range: [{np.nanmin(V_rot_eti_low) if V_rot_eti_low is not None else None:,.1f}, {np.nanmax(V_rot_eti_low) if V_rot_eti_low is not None else None:,.1f}] km/s")
        print(f"V_rot_eti_high shape: {V_rot_eti_high.shape if V_rot_eti_high is not None else None}, range: [{np.nanmin(V_rot_eti_high) if V_rot_eti_high is not None else None:,.1f}, {np.nanmax(V_rot_eti_high) if V_rot_eti_high is not None else None:,.1f}] km/s")

        plot_util.plot_rv_curves([
            {'r_map': r_disp_map, 'V_map': V_disp_map, 'color': COLOR_DATA_POINTS, 'linestyle': None, 'size': 5, 'label': 'Observed rotation velocity data'},
            {'r_map': r_median, 'V_map': V_rot_median, 'V_lower': V_rot_eti_low, 'V_upper': V_rot_eti_high, 'color': COLOR_MODEL, 'linestyle': '-', 'alpha': 0.95, 'fill_alpha': 0.16, 'fill_label': 'Inferred rotation curve: ETI [2.5%, 97.5%]', 'label': 'Inferred rotation curve: Median'},
        ], plateifu=f"{PLATE_IFU}", title="Rotation Curve with Equal Tail Intervals", savedir=result_dir, savefilename=f"{PLATE_IFU}_nfw_rv_eti")

        # plot RC components
        plot_mask = (r_median >= 0) # only plot where radius is non-negative
        plot_util.plot_rv_curves([
            {'r_map': r_median[plot_mask], 'V_map': V_rot_median[plot_mask], 'color': COLOR_V_TOTAL, 'linestyle': '-', 'label': 'Inferred Total velocity median'},
            {'r_map': r_median[plot_mask], 'V_map': V_dm_median[plot_mask], 'color': COLOR_V_DM, 'linestyle': '--', 'label': 'Inferred Dark Matter velocity median'},
            {'r_map': r_median[plot_mask], 'V_map': V_star_median[plot_mask], 'color': COLOR_V_STAR, 'linestyle': '-', 'label': 'Inferred Stellar velocity median'},
        ], plateifu=f"{PLATE_IFU}", title="Rotation Curve components", savedir=result_dir, savefilename=f"{PLATE_IFU}_nfw_rv_components")

        return

    finally:
        # Explicitly close FITS handles every iteration to avoid FD/mmap buildup.
        if maps_util is not None:
            maps_util.close()
        if firefly_util is not None:
            firefly_util.close()

        # Drop large references and force collection after each run.
        del vel_rot
        del dm_nfw
        gc.collect()

TEST_PLATE_IFUS = [
    "7957-3701",
    "8078-1902",
    "10218-6102",
    "8329-6103",
    "8723-12703",
    "8723-12705",
    "7495-12704",
    "10220-12705"
]

PLATES_FILENAME = "plateifus.txt"
def get_plate_ifu_list(filename=None):
    if filename is not None:
        plate_ifu_file = data_dir / filename
    else:
        plate_ifu_file = data_dir / PLATES_FILENAME

    with open(plate_ifu_file, 'r') as f:
        plate_ifu_list = [line.strip() for line in f if line.strip()]

    # sort the list
    plate_ifu_list.sort()
    return plate_ifu_list

import pandas as pd
VEL_ROT_PARAM_FILE = "vel_rot_param_all.csv"
def get_plate_list_from_fit():
    param_file = data_dir / VEL_ROT_PARAM_FILE
    df = pd.read_csv(param_file)
    df = df[df['result'] == 'success']
    plate_ifu_list = df['PLATE_IFU'].tolist()
    plate_ifu_list = [str(plate_ifu) for plate_ifu in plate_ifu_list]
    plate_ifu_list.sort()
    return plate_ifu_list


def _process_plate_ifu_worker(
    plate_ifu: str,
    run_nfw: bool,
    debug: bool,
    result_dir_override: str | None = None,
    r0_frac: float | None = None,
    m200_prior_dex: float | None = None,
    inc_prior_enable: bool | None = None,
):
    try:
        _set_result_dir(result_dir_override)
        _set_r0_frac(r0_frac)
        _set_m200_prior_dex(m200_prior_dex)
        _set_inc_prior_enable(inc_prior_enable)
        process_plate_ifu(plate_ifu, process_nfw=run_nfw, debug=debug)
    except Exception as e:
        print(f"Error processing {plate_ifu}: {e}")


def _is_plate_ifu_id(value: str) -> bool:
    return bool(re.fullmatch(r"\d+-\d+", str(value).strip()))


def main(
    run_nfw: bool = True,
    ifu: str = None,
    debug: bool = False,
    result_dir_override: str | Path | None = None,
    r0_frac: float | None = None,
    m200_prior_dex: float | None = None,
    inc_prior_enable: bool | None = None,
):
    active_result_dir = _set_result_dir(result_dir_override)
    _set_r0_frac(r0_frac)
    _set_m200_prior_dex(m200_prior_dex)
    _set_inc_prior_enable(inc_prior_enable)
    print(f"Using result directory: {active_result_dir}")
    print(f"Inner logistic down-weighting r0_frac: {_active_r0_frac if _active_r0_frac is not None else 'DmNfw default'}")
    print(f"M200 prior width: {f'{_active_m200_prior_dex:.3f} dex' if _active_m200_prior_dex is not None else 'DmNfw default'}")
    print(f"Inclination prior enabled: {_active_inc_prior_enable if _active_inc_prior_enable is not None else 'DmNfw default'}")

    plate_ifu_list = []

    if ifu == "all":
        plate_ifu_list = get_plate_ifu_list()
    elif ifu == "test":
        plate_ifu_list = get_plate_ifu_list("plateifus_test.txt")
    elif _is_plate_ifu_id(ifu):
        plate_ifu_list = [ifu]
    else:
        plate_ifu_list = get_plate_ifu_list(ifu)

    if not plate_ifu_list or len(plate_ifu_list) == 0:
        plate_ifu_list = TEST_PLATE_IFUS

    processed_rc_ifus = get_processed_plate_ifus(RC_PARAM_FILENAME)
    processed_nfw_ifus = get_processed_plate_ifus(NFW_PARAM_FILENAME)

    mp_context = multiprocessing.get_context("spawn")

    def _process(plate_ifu):
        print(f"\n\n########## Processing PLATE_IFU: {plate_ifu} mode=c-m200 ##########")

        # Keep serial execution: start one subprocess and wait for it to finish.
        worker = mp_context.Process(
            target=_process_plate_ifu_worker,
                args=(
                    plate_ifu,
                    run_nfw,
                    debug,
                    str(active_result_dir),
                    _active_r0_frac,
                    _active_m200_prior_dex,
                    _active_inc_prior_enable,
                ),
            name=f"ifu-worker-{plate_ifu}",
        )
        worker.start()
        worker.join()

        exit_code = worker.exitcode
        worker.close()
        del worker
        gc.collect()

        if exit_code not in (0, None):
            print(f"Error processing {plate_ifu}: worker exited with code {exit_code}")

    for plate_ifu in tqdm(plate_ifu_list, total=len(plate_ifu_list), desc="Processing galaxies", unit="galaxy"):
        if plate_ifu in processed_rc_ifus and plate_ifu in processed_nfw_ifus and not debug:
            print(f"Skipping {plate_ifu}: already exists in {RC_PARAM_FILENAME} and {NFW_PARAM_FILENAME}")
            continue
        _process(plate_ifu)
    return

import argparse

if __name__ == "__main__":
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description="Process MaNGA galaxies for velocity rotation and DM NFW fitting.")
    parser.add_argument('--nfw', type=str, default="on", help='Run dark matter NFW fitting.')
    parser.add_argument('--ifu', type=str, default="all", help='Input selector: use `all`, `test`, a single `PLATE-IFU`, or a filename such as `plateifus_test.txt`.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode.')
    parser.add_argument(
        '--result-dir',
        type=str,
        default=None,
        help='Override the result directory used for reading and writing output files.',
    )
    parser.add_argument(
        '--r0-frac',
        type=float,
        default=None,
        help='Inner logistic down-weighting half-weight radius as fraction of r_max; if omitted, keep the DmNfw default.',
    )
    parser.add_argument(
        '--m200-prior-dex',
        type=float,
        default=None,
        help='Set the M200 prior width in dex passed to DmNfw.set_M200_prior; if omitted, keep the DmNfw default.',
    )
    parser.add_argument(
        '--inc-prior',
        action='store_const',
        const=True,
        default=None,
        help='Enable the inclination prior by calling DmNfw.set_inc_prior(True); if omitted, keep the DmNfw default.',
    )

    args = parser.parse_args()

    nfw_enable = args.nfw.lower() in ['on', 'true', 'enable' ,'1']
    _set_result_dir(args.result_dir)
    _set_r0_frac(args.r0_frac)
    _set_m200_prior_dex(args.m200_prior_dex)
    _set_inc_prior_enable(args.inc_prior)

    main(
        run_nfw=nfw_enable,
        ifu=args.ifu,
        debug=args.debug,
        result_dir_override=args.result_dir,
        r0_frac=args.r0_frac,
        m200_prior_dex=args.m200_prior_dex,
        inc_prior_enable=args.inc_prior,
    )