"""Stage 1 pipeline: single-galaxy rotation-curve and NFW DM fitting.

Orchestration logic extracted from ``src-orig/main.py``.
"""

from __future__ import annotations

import gc
import multiprocessing
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.config.constants import (
    COLOR_BOOTSTRAP_LINES,
    COLOR_DATA_POINTS,
    COLOR_MODEL,
    COLOR_V_DM,
    COLOR_V_STAR,
    COLOR_V_TOTAL,
    TEST_PLATE_IFUS,
    PLATES_FILENAME,
)
from src.config.settings import settings
from src.data.catalog import DrpallUtil
from src.data.firefly import FireflyUtil
from src.data.fits import FitsUtil
from src.data.catalog import get_plateifu_list
from src.data.maps import MapsUtil
from src.data.results import (
    get_params_file,
    get_processed_plate_ifus,
    store_params_file,
    store_posterior_samples_file,
)
from src.models.dm_nfw import DmNfw
from src.models.rotation_curve import RotCurve
from src.viz.utils import PlotUtil


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
    r0_frac: float | None = None,
    m200_prior_dex: float | None = None,
    inc_prior_enable: bool | None = None,
    write_lock=None,
) -> None:
    """Run Stage 1 for a single plate-IFU: RC fit + optional NFW DM fit.

    Runs the current layered implementation directly.  The model internals
    remain in ``RotCurve`` and ``DmNfw``; this function only orchestrates I/O,
    quality gating, and result persistence.
    """
    result_dir = settings.resolve_result_dir(result_dir_override)
    result_dir.mkdir(parents=True, exist_ok=True)

    firefly_util = None
    maps_util = None
    vel_rot = None
    dm_nfw = None

    rc_filename = settings.rc_param_filename
    nfw_filename = settings.nfw_param_cm200_filename
    nfw_sample_filename = settings.nfw_param_cm200_sample_filename

    vel_rot_param = get_params_file(plate_ifu, rc_filename, result_dir)
    if not process_nfw and vel_rot_param is not None:
        if vel_rot_param["result"] != "success":
            print(f"Velocity rotation fit previously failed for {plate_ifu}. Skipping processing.")
            return

    if fits_util is None:
        fits_util = FitsUtil(settings.data_dir)

    drpall_file = fits_util.get_drpall_file()
    firefly_file = fits_util.get_firefly_file()
    maps_file = fits_util.get_maps_file(plate_ifu, checksum=False, download=False)
    if maps_file is None:
        print(f"MAPS file for {plate_ifu} not found locally. Skipping processing.")
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
        vel_rot.set_PLATE_IFU(plate_ifu)

        r_obs_map, v_obs_map, ivar_map, phi_map = vel_rot.get_vel_obs()
        gflux_map, _, _ = maps_util.get_eml_gflux_map()
        fwhm = maps_util.get_fwhm()
        pixel_scale = maps_util.get_pixel_scale()
        print(f"Pixel scale: {pixel_scale:.3f} arcsec/pixel, FWHM: {fwhm:.3f} arcsec")

        radius_fit = vel_rot.get_radius_fit(np.nanmax(r_obs_map), count=1000)

        print(f"## RC fitting {plate_ifu} ##")
        vel_param = {
            "radius_obs": r_obs_map,
            "vel_obs": v_obs_map,
            "ivar_obs": ivar_map,
            "phi_map": phi_map,
            "gflux_map": gflux_map,
        }
        success, plot_result, fit_params = vel_rot.fit_vel_rot(
            vel_param,
            radius_fit=radius_fit,
        )

        if not success:
            if isinstance(fit_params, dict):
                fit_params["quality_pass"] = False
                fit_params["quality_fail_reasons"] = "fit_failed"
                fit_params["quality_summary"] = "fit_failed_before_quality_gate"
                store_params_file(
                    plate_ifu,
                    fit_params,
                    rc_filename,
                    result_dir,
                    write_lock=write_lock,
                )
            print(f"Fitting rotational velocity failed for {plate_ifu}")
            return

        r_obs_map = plot_result["radius_obs"]
        v_obs_map = plot_result["vel_obs"]
        ivar_obs_map = plot_result["ivar_obs"]
        r_rot_fit = plot_result["radius_rot"]
        v_rot_fit = plot_result["vel_rot"]
        print(
            f"Fitted rotation curve has radius range "
            f"[{np.nanmin(r_rot_fit):.1f}, {np.nanmax(r_rot_fit):.1f}] kpc"
        )
        print(
            f"Fitted rotation curve has velocity range "
            f"[{np.nanmin(v_rot_fit):.1f}, {np.nanmax(v_rot_fit):.1f}] km/s"
        )

        inc_rad_fit = float(fit_params["inc"])
        inc_deg_fit = float(np.degrees(inc_rad_fit))
        vel_sys_fit = float(fit_params["Vsys"])
        phi_delta_fit = float(fit_params["phi_delta"])
        rmax = float(fit_params["Rmax"])
        rt = float(fit_params["Rt"])

        data_count = np.sum(np.isfinite(v_obs_map))
        nrmse = float(fit_params["NRMSE"])
        chi_sq_v = float(fit_params["CHI_SQ_V"])
        quality_gate = RotCurve.evaluate_fit_quality(fit_params, data_count)
        if isinstance(fit_params, dict):
            fit_params["inc_deg"] = f"{quality_gate['inc_deg']:.3f}"
            fit_params["Rmax_Rt_ratio"] = f"{quality_gate['rmax_rt_ratio']:.3f}"
            fit_params["quality_pass"] = bool(quality_gate["passed"])
            fit_params["quality_summary"] = quality_gate["summary"]
        store_params_file(
            plate_ifu,
            fit_params,
            rc_filename,
            result_dir,
            write_lock=write_lock,
        )

        if not quality_gate["passed"]:
            fail_reason_text = "; ".join(quality_gate["fail_reasons"])
            print(
                f"fitting results failure for {plate_ifu}: {quality_gate['summary']}, "
                f"reasons: {fail_reason_text}, NRMSE(diag): {nrmse:.3f}, "
                f"CHI_SQ_V(diag): {chi_sq_v:.3f}"
            )
            return

        if debug:
            r_disp_map, v_disp_map, _ = vel_rot.get_vel_obs_disp(
                inc_rad_fit,
                vel_sys_fit,
                phi_delta_fit,
            )
            v_rot_hdi_low = plot_result.get("vel_rot_hdi_low")
            v_rot_hdi_high = plot_result.get("vel_rot_hdi_high")
            show_hdi = v_rot_hdi_low is not None and v_rot_hdi_high is not None

            plot_util.plot_rv_curves(
                [
                    {
                        "r_map": r_disp_map,
                        "V_map": v_disp_map,
                        "color": COLOR_DATA_POINTS,
                        "label": "Observed data",
                    },
                    {
                        "r_map": r_rot_fit,
                        "V_map": v_rot_fit,
                        "V_lower": v_rot_hdi_low if show_hdi else None,
                        "V_upper": v_rot_hdi_high if show_hdi else None,
                        "color": COLOR_MODEL,
                        "linestyle": "-",
                        "alpha": 0.95,
                        "fill_alpha": 0.16,
                        "fill_label": "Posterior HDI" if show_hdi else None,
                        "label": "Inferred RC Median",
                    },
                ],
                plateifu=f"{plate_ifu}",
                title="Rotation Curve Posterior-based Sample Selection",
                savedir=result_dir,
            )

        print(
            f"Fitting successful for {plate_ifu}. Inc: {inc_deg_fit:.2f} deg, "
            f"Vsys: {vel_sys_fit:.2f} km/s, phi_delta: {phi_delta_fit:.2f} deg, "
            f"Rmax: {rmax:.2f} arcsec, Rt: {rt:.2f} arcsec"
        )
        print("")

        if not process_nfw:
            return

        print(f"## DM NFW inferring {plate_ifu} ##")
        r_disp_map, v_disp_map, _ = vel_rot.get_vel_obs_disp(
            inc_rad=inc_rad_fit,
            vel_sys=vel_sys_fit,
            phi_delta=phi_delta_fit,
        )

        dm_nfw = DmNfw(drpall_util)
        dm_nfw.set_PLATE_IFU(plate_ifu)
        dm_nfw.set_plot_enable(debug, output_dir=result_dir)
        dm_nfw.set_inf_debug(debug)
        if m200_prior_dex is not None:
            dm_nfw.set_M200_prior(m200_prior_dex)
        if inc_prior_enable is not None:
            dm_nfw.set_inc_prior(inc_prior_enable)
        if r0_frac is not None:
            dm_nfw.set_down_weight_inner(r0_frac)

        dm_vel_param = {
            "radius_obs": r_obs_map,
            "vel_obs": v_obs_map,
            "ivar_obs": ivar_obs_map,
            "vel_sys": vel_sys_fit,
            "inc_rad": inc_rad_fit,
            "phi_map": phi_map,
        }

        success, plot_result, inf_params, posterior_samples = dm_nfw.inf_dm_nfw(
            vel_param=dm_vel_param,
            radius_fit=radius_fit,
        )
        store_params_file(
            plate_ifu,
            inf_params,
            nfw_filename,
            result_dir,
            write_lock=write_lock,
        )
        store_posterior_samples_file(
            plate_ifu,
            posterior_samples,
            nfw_sample_filename,
            result_dir,
        )

        if not success:
            print(f"Inferring dark matter NFW failed for {plate_ifu}")
            return

        r_median = plot_result["radius"]
        v_rot_median = plot_result["v_rot"]
        v_rot_eti_low = plot_result.get("v_rot_eti_low")
        v_rot_eti_high = plot_result.get("v_rot_eti_high")
        v_rot_samples = plot_result.get("v_rot_samples")
        v_dm_median = plot_result["v_dm"]
        v_star_median = plot_result["v_star"]
        v_drift_median = plot_result.get("v_drift")

        rv_plot_data = []
        if v_rot_samples is not None:
            v_rot_samples = np.asarray(v_rot_samples, dtype=float)
            if v_rot_samples.ndim == 2 and v_rot_samples.shape[1] > 0:
                sample_idx = np.linspace(
                    0,
                    v_rot_samples.shape[1] - 1,
                    min(100, v_rot_samples.shape[1]),
                    dtype=int,
                )
                for line_idx, sample_id in enumerate(sample_idx):
                    rv_plot_data.append(
                        {
                            "r_map": r_median,
                            "V_map": v_rot_samples[:, sample_id],
                            "color": COLOR_BOOTSTRAP_LINES,
                            "linestyle": "-",
                            "alpha": 0.10,
                            "label": "RC Posterior Samples" if line_idx == 0 else None,
                        }
                    )
        rv_plot_data.extend(
            [
                {
                    "r_map": r_disp_map,
                    "V_map": v_disp_map,
                    "color": COLOR_DATA_POINTS,
                    "linestyle": None,
                    "size": 5,
                    "label": "Observed rotation velocity data",
                },
                {
                    "r_map": r_median,
                    "V_map": v_rot_median,
                    "color": COLOR_V_TOTAL,
                    "linestyle": "-",
                    "label": "Inferred rotation curve median",
                },
            ]
        )
        plot_util.plot_rv_curves(
            rv_plot_data,
            plateifu=f"{plate_ifu}",
            title="Rotation Curve posterior samples",
            savedir=result_dir,
            savefilename=f"{plate_ifu}_nfw_rv_posterior_samples",
        )

        plot_util.plot_rv_curves(
            [
                {
                    "r_map": r_disp_map,
                    "V_map": v_disp_map,
                    "color": COLOR_DATA_POINTS,
                    "linestyle": None,
                    "size": 5,
                    "label": "Observed rotation velocity data",
                },
                {
                    "r_map": r_median,
                    "V_map": v_rot_median,
                    "V_lower": v_rot_eti_low,
                    "V_upper": v_rot_eti_high,
                    "color": COLOR_MODEL,
                    "linestyle": "-",
                    "alpha": 0.95,
                    "fill_alpha": 0.16,
                    "fill_label": "Inferred rotation curve: ETI [2.5%, 97.5%]",
                    "label": "Inferred rotation curve: Median",
                },
            ],
            plateifu=f"{plate_ifu}",
            title="Rotation Curve with Equal Tail Intervals",
            savedir=result_dir,
            savefilename=f"{plate_ifu}_nfw_rv_eti",
        )

        plot_mask = r_median >= 0
        plot_util.plot_rv_curves(
            [
                {
                    "r_map": r_median[plot_mask],
                    "V_map": v_rot_median[plot_mask],
                    "color": COLOR_V_TOTAL,
                    "linestyle": "-",
                    "label": "Inferred Total velocity median",
                },
                {
                    "r_map": r_median[plot_mask],
                    "V_map": v_dm_median[plot_mask],
                    "color": COLOR_V_DM,
                    "linestyle": "--",
                    "label": "Inferred Dark Matter velocity median",
                },
                {
                    "r_map": r_median[plot_mask],
                    "V_map": v_star_median[plot_mask],
                    "color": COLOR_V_STAR,
                    "linestyle": "-",
                    "label": "Inferred Stellar velocity median",
                },
            ],
            plateifu=f"{plate_ifu}",
            title="Rotation Curve components",
            savedir=result_dir,
            savefilename=f"{plate_ifu}_nfw_rv_components",
        )

    finally:
        if maps_util is not None and hasattr(maps_util, "close"):
            maps_util.close()
        if firefly_util is not None and hasattr(firefly_util, "close"):
            firefly_util.close()

        del vel_rot
        del dm_nfw
        gc.collect()


def process_plate_ifu_worker(
    plate_ifu: str,
    run_nfw: bool,
    debug: bool,
    result_dir_override: str | None = None,
    r0_frac: float | None = None,
    m200_prior_dex: float | None = None,
    inc_prior_enable: bool | None = None,
    write_lock=None,
) -> None:
    """Multiprocessing worker wrapper."""
    process_plate_ifu(
        plate_ifu=plate_ifu,
        process_nfw=run_nfw,
        debug=debug,
        result_dir_override=result_dir_override,
        r0_frac=r0_frac,
        m200_prior_dex=m200_prior_dex,
        inc_prior_enable=inc_prior_enable,
        write_lock=write_lock,
    )


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
        plate_ifus = get_plateifu_list(
            filepath=settings.data_dir / PLATES_FILENAME
        )
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
    processed_filename = (
        settings.nfw_param_cm200_filename if nfw else settings.rc_param_filename
    )
    processed = get_processed_plate_ifus(
        processed_filename,
        result_dir,
        successful_only=True,
        required_sample_filename=(
            settings.nfw_param_cm200_sample_filename if nfw else None
        ),
    )
    todo = [p for p in plate_ifus if p not in processed]
    if not todo:
        print("All plate-IFUs already processed.")
        return

    print(f"Processing {len(todo)} plate-IFUs with {n_cores} workers...")

    if n_cores > 1:
        with multiprocessing.Manager() as manager:
            write_lock = manager.Lock()
            with multiprocessing.Pool(processes=n_cores) as pool:
                args = [
                    (
                        p,
                        nfw,
                        debug,
                        result_dir_override,
                        r0_frac,
                        m200_prior_dex,
                        inc_prior_enable,
                        write_lock,
                    )
                    for p in todo
                ]
                list(tqdm(
                    pool.starmap(process_plate_ifu_worker, args),
                    total=len(todo),
                    desc="Stage 1",
                ))
    else:
        for plate_ifu in tqdm(todo, desc="Stage 1"):
            process_plate_ifu(
                plate_ifu,
                process_nfw=nfw,
                debug=debug,
                result_dir_override=result_dir_override,
                r0_frac=r0_frac,
                m200_prior_dex=m200_prior_dex,
                inc_prior_enable=inc_prior_enable,
            )

    gc.collect()
    print("Stage 1 complete.")
