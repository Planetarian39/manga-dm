from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import tomllib

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from scipy.stats import gaussian_kde, truncnorm

from dm import DmNfw
from rc import RotCurve
from util.drpall_util import DrpallUtil
from util.firefly_util import FireflyUtil
from util.fits_util import FitsUtil
from util.maps_util import MapsUtil

'''
python ./src/figure.py --ifu=10517-6102,8593-12701,7977-3704,8549-3703 --output=data/results/velocity_field_comparison_low_mass.png --posterior-output=data/results/m200_c_comparison_low_mass.png
python ./src/figure.py --ifu=9493-6101,8314-12702,11743-9102,9509-12703 --output=data/results/velocity_field_comparison_high_mass.png --posterior-output=data/results/m200_c_comparison_high_mass.png
'''

DEFAULT_PLATE_IFUS = [
    "8994-12701",
    "7977-3704",
    "9493-6101",
    "11743-9102",
]

COLOR_MODEL = "black"
COLOR_DATA_POINTS = "#4D4D4D"
COLOR_V_TOTAL = "#4D4D4D"
COLOR_V_STAR = "#0072B2"
COLOR_V_DM = "#7F7F7F"
COLOR_PARAM_TEXT = "#7A1E1E"

# Load configuration file
with open("config.toml", "rb") as f:
    config = tomllib.load(f)
    if not config:
        raise ValueError("Error: config.toml file is empty")

HDI_PROB1 = config.get("thresholds", {}).get("HDI_PROB1", 0.68)
HDI_PROB2 = config.get("thresholds", {}).get("HDI_PROB2", 0.95)

plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "mathtext.fontset": "dejavuserif",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    }
)


with open("config.toml", "rb") as f:
    config = tomllib.load(f)
    if not config:
        raise ValueError("Error: config.toml file is empty")

data_directory = config.get("file", {}).get("data_directory", "data")
result_directory = config.get("file", {}).get("result_directory", "results")

root_dir = Path(__file__).resolve().parent.parent
data_dir = root_dir / data_directory
result_dir = data_dir / result_directory
result_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class GalaxyFigureData:
    plateifu: str
    image_path: Path
    extent: tuple[float, float, float, float]
    r_disp_map_all: np.ndarray
    observed_map: np.ndarray
    model_map: np.ndarray
    map_norm: colors.TwoSlopeNorm
    map_ticks: list[float]
    r_disp_map: np.ndarray
    v_disp_map: np.ndarray
    r_rc_fit: np.ndarray
    v_rc_fit: np.ndarray
    v_rc_hdi_low: np.ndarray | None
    v_rc_hdi_high: np.ndarray | None
    r_model: np.ndarray
    v_model: np.ndarray
    v_eti_low: np.ndarray | None
    v_eti_high: np.ndarray | None
    v_dm: np.ndarray
    v_star: np.ndarray
    rc_annotation: str
    nfw_annotation: str
    log10_m200_samples: np.ndarray
    log10_c_samples: np.ndarray
    log10_m200_prior_mu: float
    log10_m200_prior_sigma: float
    log10_m200_prior_lower: float
    log10_m200_prior_upper: float
    log10_c_prior_mu: float
    log10_c_prior_sigma: float


def _format_float(value: float, digits: int = 1) -> str:
    value = float(value)
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def _build_rc_annotation(fit_params: dict) -> str:
    vc = _format_float(float(fit_params["Vc"]), 0)
    rt = _format_float(float(fit_params["Rt"]), 1)
    s_out = _format_float(float(fit_params["s_out"]), 1)
    inc_deg = _format_float(np.degrees(float(fit_params["inc"])), 1)
    phi_deg = _format_float(np.degrees(float(fit_params["phi_delta"])), 1)
    return "\n".join(
        [
            rf"$V_c = {vc}$ km s$^{{-1}}$",
            rf"$R_t = {rt}$ kpc",
            rf"$s_{{out}} = {s_out}$ km s$^{{-1}}$ kpc$^{{-1}}$",
            rf"$i = {inc_deg}^\circ$, $\Delta\phi = {phi_deg}^\circ$",
        ]
    )


def _build_nfw_annotation(inf_params: dict) -> str:
    log10_m200 = _format_float(float(inf_params["log10_M200"]), 2)
    log10_mstar = _format_float(float(inf_params["log10_Mstar"]), 2)
    c_value = _format_float(float(inf_params["c"]), 2)
    sersic_n = _format_float(float(inf_params["sersic_n"]), 2)
    return "\n".join(
        [
            rf"$n_{{\rm Sersic}} = {sersic_n}$",
            rf"$\log_{{10}} M_\star = {log10_mstar}$",
            rf"$\log_{{10}} M_{{200}} = {log10_m200}$",
            rf"$c = {c_value}$",
        ]
    )


def _symmetric_norm(*arrays: np.ndarray) -> tuple[colors.TwoSlopeNorm, list[float]]:
    finite_values = []
    for array in arrays:
        values = np.asarray(array, dtype=float)
        values = values[np.isfinite(values)]
        if values.size > 0:
            finite_values.append(values)

    if not finite_values:
        vmax = 1.0
    else:
        merged = np.concatenate(finite_values)
        p_low, p_high = np.nanpercentile(merged, [2, 98])
        vmax = float(max(abs(p_low), abs(p_high)))
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0

    vmax = float(np.ceil(vmax / 10.0) * 10.0)
    return colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax), [-vmax, 0.0, vmax]


def _get_extent(offset_x: np.ndarray, offset_y: np.ndarray) -> tuple[float, float, float, float]:
    x_valid = np.asarray(offset_x, dtype=float)
    y_valid = np.asarray(offset_y, dtype=float)
    return (
        float(np.nanmax(x_valid)),
        float(np.nanmin(x_valid)),
        float(np.nanmin(y_valid)),
        float(np.nanmax(y_valid)),
    )


def _interp_profile_on_map(radius_grid: np.ndarray, velocity_grid: np.ndarray, signed_radius_map: np.ndarray) -> np.ndarray:
    radius = np.asarray(radius_grid, dtype=float).ravel()
    velocity = np.asarray(velocity_grid, dtype=float).ravel()
    signed_radius = np.asarray(signed_radius_map, dtype=float)

    profile_mask = np.isfinite(radius) & np.isfinite(velocity)
    if not np.any(profile_mask):
        return np.full_like(signed_radius, np.nan, dtype=float)

    radius_sorted = radius[profile_mask]
    velocity_sorted = velocity[profile_mask]
    sort_idx = np.argsort(radius_sorted)
    radius_sorted = radius_sorted[sort_idx]
    velocity_sorted = velocity_sorted[sort_idx]

    output = np.full(signed_radius.shape, np.nan, dtype=float)
    data_mask = np.isfinite(signed_radius)
    if np.any(data_mask):
        output[data_mask] = np.interp(
            signed_radius[data_mask],
            radius_sorted,
            velocity_sorted,
            left=np.nan,
            right=np.nan,
        )
    return output


def _signed_radius(radius_map: np.ndarray, velocity_map: np.ndarray) -> np.ndarray:
    radius = np.asarray(radius_map, dtype=float)
    velocity = np.asarray(velocity_map, dtype=float)
    return np.sign(velocity) * np.abs(radius)


def _prepare_galaxy_figure_data(plateifu: str, radius_count: int = 1000) -> GalaxyFigureData:
    fits_util = FitsUtil(data_dir)
    drpall_util = DrpallUtil(fits_util.get_drpall_file())
    firefly_util = FireflyUtil(fits_util.get_firefly_file())
    maps_file = fits_util.get_maps_file(plateifu, checksum=False, download=False)
    if maps_file is None:
        raise FileNotFoundError(f"MAPS file for {plateifu} not found locally")

    maps_util = MapsUtil(maps_file)

    try:
        vel_rot = RotCurve(drpall_util, firefly_util, maps_util, plot_util=None)
        vel_rot.set_PLATE_IFU(plateifu)

        r_obs_map, v_obs_map, ivar_obs_map, phi_map = vel_rot.get_vel_obs()
        gflux_map, _, _ = maps_util.get_eml_gflux_map()

        radius_fit = vel_rot.get_radius_fit(np.nanmax(r_obs_map), count=radius_count)
        rc_vel_param = {
            "radius_obs": r_obs_map,
            "vel_obs": v_obs_map,
            "ivar_obs": ivar_obs_map,
            "phi_map": phi_map,
            "gflux_map": gflux_map,
        }

        rc_success, rc_plot_result, fit_params = vel_rot.fit_vel_rot(rc_vel_param, radius_fit=radius_fit)
        if not rc_success:
            raise RuntimeError(f"RC fitting failed for {plateifu}")

        data_count = int(np.sum(np.isfinite(v_obs_map)))
        quality_gate = RotCurve.evaluate_fit_quality(fit_params, data_count)
        if not quality_gate.get("passed", False):
            raise RuntimeError(
                f"RC quality gate failed for {plateifu}: {quality_gate.get('summary', 'unknown')}"
            )

        inc_rad_fit = float(fit_params["inc"])
        vel_sys_fit = float(fit_params["Vsys"])
        phi_delta_fit = float(fit_params["phi_delta"])

        r_disp_map, v_disp_map, _ = vel_rot.get_vel_obs_disp(
            inc_rad=inc_rad_fit, vel_sys=vel_sys_fit, phi_delta=phi_delta_fit)

        r_disp_map_all, v_disp_map_all, _ = vel_rot.get_vel_obs_disp(
            inc_rad=inc_rad_fit, vel_sys=vel_sys_fit, phi_delta=phi_delta_fit, is_filter=True)

        dm_vel_param = {
            "radius_obs": r_obs_map,
            "vel_obs": v_obs_map,
            "ivar_obs": ivar_obs_map,
            "vel_sys": vel_sys_fit,
            "inc_rad": inc_rad_fit,
            "phi_map": phi_map,
        }

        dm_nfw = DmNfw(drpall_util)
        dm_nfw.set_PLATE_IFU(plateifu)
        dm_nfw.set_plot_enable(False, output_dir=str(result_dir))
        dm_nfw.set_inf_debug(False)

        dm_success, plot_result, inf_params, posterior_samples = dm_nfw.inf_dm_nfw(dm_vel_param, radius_fit=radius_fit)
        if not dm_success:
            raise RuntimeError(f"DM NFW inference failed for {plateifu}")

        signed_radius_map = _signed_radius(r_disp_map_all, v_disp_map_all)
        v_model_map = _interp_profile_on_map(plot_result["radius"], plot_result["v_rot"], signed_radius_map)
        v_model_map = np.where(np.isfinite(v_disp_map_all), v_model_map, np.nan)

        offset_x, offset_y = maps_util.get_sky_offsets()
        map_norm, map_ticks = _symmetric_norm(v_disp_map_all, v_model_map)

        return GalaxyFigureData(
            plateifu=plateifu,
            image_path=fits_util.get_image_file(plateifu),
            extent=_get_extent(offset_x, offset_y),
            r_disp_map_all=r_disp_map_all,
            observed_map=v_disp_map_all,
            model_map=v_model_map,
            map_norm=map_norm,
            map_ticks=map_ticks,
            r_disp_map=r_disp_map,
            v_disp_map=v_disp_map,
            r_rc_fit=np.asarray(rc_plot_result["radius_rot"], dtype=float),
            v_rc_fit=np.asarray(rc_plot_result["vel_rot"], dtype=float),
            v_rc_hdi_low=(
                None
                if rc_plot_result.get("vel_rot_hdi_low") is None
                else np.asarray(rc_plot_result["vel_rot_hdi_low"], dtype=float)
            ),
            v_rc_hdi_high=(
                None
                if rc_plot_result.get("vel_rot_hdi_high") is None
                else np.asarray(rc_plot_result["vel_rot_hdi_high"], dtype=float)
            ),
            r_model=np.asarray(plot_result["radius"], dtype=float),
            v_model=np.asarray(plot_result["v_rot"], dtype=float),
            v_eti_low=(
                None
                if plot_result.get("v_rot_eti_low") is None
                else np.asarray(plot_result["v_rot_eti_low"], dtype=float)
            ),
            v_eti_high=(
                None
                if plot_result.get("v_rot_eti_high") is None
                else np.asarray(plot_result["v_rot_eti_high"], dtype=float)
            ),
            v_dm=np.asarray(plot_result["v_dm"], dtype=float),
            v_star=np.asarray(plot_result["v_star"], dtype=float),
            rc_annotation=_build_rc_annotation(fit_params),
            nfw_annotation=_build_nfw_annotation(inf_params),
            log10_m200_samples=np.asarray(posterior_samples["log10_M200_samples"], dtype=float),
            log10_c_samples=np.asarray(posterior_samples["log10_c_samples"], dtype=float),
            log10_m200_prior_mu=float(inf_params["log10_M200_prior_mu"]),
            log10_m200_prior_sigma=float(inf_params["log10_M200_prior_sigma"]),
            log10_m200_prior_lower=float(inf_params["log10_M200_prior_lower"]),
            log10_m200_prior_upper=float(inf_params["log10_M200_prior_upper"]),
            log10_c_prior_mu=float(inf_params["log10_c_prior_mu"]),
            log10_c_prior_sigma=float(inf_params["log10_c_prior_sigma"]),
        )
    finally:
        maps_util.close()
        firefly_util.close()


def _prepare_galaxy_figure_items(plateifus: list[str], radius_count: int = 1000) -> list[GalaxyFigureData]:
    return [_prepare_galaxy_figure_data(plateifu, radius_count=radius_count) for plateifu in plateifus]


def _sample_prior_log10_values(item: GalaxyFigureData, sample_count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = (item.log10_m200_prior_lower - item.log10_m200_prior_mu) / item.log10_m200_prior_sigma
    b = (item.log10_m200_prior_upper - item.log10_m200_prior_mu) / item.log10_m200_prior_sigma
    prior_log10_m200 = truncnorm.rvs(
        a,
        b,
        loc=item.log10_m200_prior_mu,
        scale=item.log10_m200_prior_sigma,
        size=sample_count,
        random_state=rng,
    )
    prior_log10_c = rng.normal(
        loc=item.log10_c_prior_mu,
        scale=item.log10_c_prior_sigma,
        size=sample_count,
    )
    return np.asarray(prior_log10_m200, dtype=float), np.asarray(prior_log10_c, dtype=float)


def _build_m200_c_inferencedata(item: GalaxyFigureData, seed: int = 42):
    sample_count = max(len(item.log10_m200_samples), 1000)
    prior_log10_m200, prior_log10_c = _sample_prior_log10_values(item, sample_count=sample_count, seed=seed)

    return {
        "prior": {
            "M200": np.power(10.0, prior_log10_m200),
            "c": np.power(10.0, prior_log10_c),
        },
        "posterior": {
            "M200": np.power(10.0, np.asarray(item.log10_m200_samples, dtype=float)),
            "c": np.power(10.0, np.asarray(item.log10_c_samples, dtype=float)),
        },
    }


def _finite_values(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float).ravel()
    return array[np.isfinite(array)]


def _density_range(*arrays: np.ndarray, pad_fraction: float = 0.06) -> tuple[float, float]:
    finite_arrays = [_finite_values(array) for array in arrays]
    finite_arrays = [array for array in finite_arrays if array.size > 0]
    if not finite_arrays:
        return 0.0, 1.0

    merged = np.concatenate(finite_arrays)
    x_min = float(np.min(merged))
    x_max = float(np.max(merged))
    if not np.isfinite(x_min) or not np.isfinite(x_max):
        return 0.0, 1.0
    if x_min == x_max:
        delta = max(abs(x_min) * 0.05, 1.0)
        return x_min - delta, x_max + delta

    padding = (x_max - x_min) * pad_fraction
    return x_min - padding, x_max + padding


def _plot_density_comparison(
    ax: plt.Axes,
    prior_values: np.ndarray,
    posterior_values: np.ndarray,
    xlabel: str,
    colors_for_series: list[str],
    show_legend: bool,
) -> None:
    prior = _finite_values(prior_values)
    posterior = _finite_values(posterior_values)
    x_min, x_max = _density_range(prior, posterior)
    bins = np.linspace(x_min, x_max, 40)

    if prior.size > 0:
        ax.hist(
            prior,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.2,
            color=colors_for_series[0],
            label="Prior",
        )
        ax.hist(
            prior,
            bins=bins,
            density=True,
            histtype="stepfilled",
            alpha=0.08,
            color=colors_for_series[0],
        )
    if posterior.size > 0:
        ax.hist(
            posterior,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.4,
            color=colors_for_series[1],
            label="Posterior",
        )
        ax.hist(
            posterior,
            bins=bins,
            density=True,
            histtype="stepfilled",
            alpha=0.14,
            color=colors_for_series[1],
        )

    ax.set_xlim(x_min, x_max)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel("Density", fontsize=8)
    ax.tick_params(labelsize=7)
    if show_legend:
        legend = ax.legend(frameon=False, fontsize=7, loc="upper right")
        if legend is not None:
            legend.set_title(None)


def _density_thresholds(density: np.ndarray, probs: list[float]) -> list[float]:
    flat = np.asarray(density, dtype=float).ravel()
    flat = flat[np.isfinite(flat)]
    if flat.size == 0 or np.all(flat <= 0):
        return []

    order = np.argsort(flat)[::-1]
    cumulative = np.cumsum(flat[order])
    cumulative /= cumulative[-1]

    thresholds = []
    for prob in probs:
        idx = int(np.searchsorted(cumulative, prob, side="left"))
        idx = min(idx, order.size - 1)
        thresholds.append(float(flat[order[idx]]))
    return sorted(set(thresholds))


def _add_eti_reference_lines(ax: plt.Axes, values: np.ndarray, *, orientation: str) -> None:
    finite = _finite_values(values)
    if finite.size == 0:
        return

    eti_values = np.percentile(finite, [16.0, 50.0, 84.0])
    for eti_value in eti_values:
        if orientation == "vertical":
            ax.axvline(eti_value, color="#7F7F7F", linestyle="--", linewidth=0.9, alpha=0.9)
        elif orientation == "horizontal":
            ax.axhline(eti_value, color="#7F7F7F", linestyle="--", linewidth=0.9, alpha=0.9)
        else:
            raise ValueError(f"Unsupported orientation: {orientation}")


def _plot_pair_summary(pair_axes: np.ndarray, m200_values: np.ndarray, c_values: np.ndarray) -> None:
    top_left = pair_axes[0, 0]
    top_right = pair_axes[0, 1]
    bottom_left = pair_axes[1, 0]
    bottom_right = pair_axes[1, 1]

    m200 = _finite_values(m200_values)
    c = _finite_values(c_values)
    sample_count = min(m200.size, c.size)
    if sample_count == 0:
        top_right.axis("off")
        return

    m200 = m200[:sample_count]
    c = c[:sample_count]

    x_min, x_max = _density_range(m200)
    y_min, y_max = _density_range(c)
    x_bins = np.linspace(x_min, x_max, 35)
    y_bins = np.linspace(y_min, y_max, 35)

    top_left.hist(m200, bins=x_bins, density=True, histtype="step", linewidth=1.2, color=COLOR_V_STAR)
    _add_eti_reference_lines(top_left, m200, orientation="vertical")
    top_left.set_xlim(x_min, x_max)
    top_left.set_ylabel("Density", fontsize=7)
    top_left.tick_params(labelsize=7)
    top_left.tick_params(axis="x", labelbottom=False)

    bottom_right.hist(c, bins=y_bins, density=True, histtype="step", linewidth=1.2, color=COLOR_V_STAR, orientation="horizontal")
    _add_eti_reference_lines(bottom_right, c, orientation="horizontal")
    bottom_right.set_ylim(y_min, y_max)
    bottom_right.set_xlabel("Density", fontsize=7)
    bottom_right.tick_params(labelsize=7)
    bottom_right.tick_params(axis="y", labelleft=False)

    top_right.axis("off")

    try:
        kde = gaussian_kde(np.vstack([m200, c]))
        x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, 120), np.linspace(y_min, y_max, 120))
        density = kde(np.vstack([x_grid.ravel(), y_grid.ravel()])).reshape(x_grid.shape)
        levels = _density_thresholds(density, [HDI_PROB2, HDI_PROB1])
        if levels:
            base_rgb = np.array(colors.to_rgb(COLOR_V_STAR), dtype=float)
            light_rgb = tuple(1.0 - (1.0 - base_rgb) * 0.30)
            dark_rgb = tuple(1.0 - (1.0 - base_rgb) * 0.75)
            bottom_left.contourf(
                x_grid,
                y_grid,
                density,
                levels=levels + [float(np.nanmax(density))],
                colors=[light_rgb, dark_rgb],
                alpha=0.8,
            )
            bottom_left.contour(
                x_grid,
                y_grid,
                density,
                levels=levels,
                colors=[dark_rgb, COLOR_V_STAR][: len(levels)],
                linewidths=1.0,
            )
        else:
            raise ValueError("Invalid KDE levels")
    except (np.linalg.LinAlgError, ValueError):
        bottom_left.scatter(m200, c, s=5, alpha=0.18, color=COLOR_V_STAR, linewidths=0)

    bottom_left.set_xlim(x_min, x_max)
    bottom_left.set_ylim(y_min, y_max)
    bottom_left.set_xlabel("M200", fontsize=8)
    bottom_left.set_ylabel("c", fontsize=8)
    bottom_left.tick_params(labelsize=7)


def _add_m200_c_summary_column_titles(fig: plt.Figure, outer_grid, y_offset: float = 0.012) -> None:
    labels = ["Prior vs posterior", "Posterior pair"]
    for col_idx, label in enumerate(labels):
        bbox = outer_grid[0, col_idx].get_position(fig)
        x_center = 0.5 * (bbox.x0 + bbox.x1)
        fig.text(x_center, bbox.y1 + y_offset, label, ha="center", va="bottom", fontsize=12)


def _add_m200_c_summary_row_labels(fig: plt.Figure, outer_grid, items: list[GalaxyFigureData]) -> None:
    for row_idx, item in enumerate(items):
        bbox = outer_grid[row_idx, 0].get_position(fig)
        y_center = 0.5 * (bbox.y0 + bbox.y1)
        fig.text(0.035, y_center, item.plateifu, rotation=90, va="center", ha="center", fontsize=11)


def _style_density_axes(compare_axes: np.ndarray, show_legend: bool) -> None:
    for axis in np.atleast_1d(compare_axes):
        axis.tick_params(labelsize=7)
        legend = axis.get_legend()
        if legend is not None:
            if show_legend:
                legend.set_frame_on(False)
                for text in legend.get_texts():
                    text.set_fontsize(7)
            else:
                legend.remove()


def _style_pair_axes(pair_axes_array: np.ndarray) -> None:
    top_left = pair_axes_array[0, 0]
    top_right = pair_axes_array[0, 1]
    bottom_left = pair_axes_array[1, 0]
    bottom_right = pair_axes_array[1, 1]

    top_left.spines["top"].set_visible(False)
    top_left.spines["right"].set_visible(False)
    bottom_right.spines["top"].set_visible(False)
    bottom_right.spines["right"].set_visible(False)
    bottom_right.spines["left"].set_visible(False)
    top_right.axis("off")

    bottom_left.spines["top"].set_visible(False)
    bottom_left.spines["right"].set_visible(False)


def _save_figure_file(
    fig: plt.Figure,
    path: Path,
    *,
    dpi: int | None = None,
    format: str | None = None,
) -> Path:
    save_kwargs = {"bbox_inches": "tight"}
    if dpi is not None:
        save_kwargs["dpi"] = dpi
    if format is not None:
        save_kwargs["format"] = format

    try:
        fig.savefig(path, **save_kwargs)
        return path
    except PermissionError:
        fallback_path = path.with_name(f"{path.stem}_new{path.suffix}")
        counter = 2
        while fallback_path.exists():
            fallback_path = path.with_name(f"{path.stem}_new{counter}{path.suffix}")
            counter += 1

        fig.savefig(fallback_path, **save_kwargs)
        print(
            f"Warning: Could not overwrite {path} because it is in use. "
            f"Saved to {fallback_path} instead."
        )
        return fallback_path


def plot_m200_c_summary_comparison(
    plateifus: list[str],
    output_path: Path,
    radius_count: int = 1000,
    dpi: int = 300,
    items: list[GalaxyFigureData] | None = None,
) -> Path:
    if len(plateifus) != 4:
        raise ValueError(f"This figure expects exactly 4 galaxies, got {len(plateifus)}")

    if items is None:
        items = _prepare_galaxy_figure_items(plateifus, radius_count=radius_count)

    fig = plt.figure(figsize=(10.0, 20.0))
    outer_grid = fig.add_gridspec(
        4,
        2,
        left=0.08,
        right=0.985,
        top=0.95,
        bottom=0.08,
        hspace=0.28,
        wspace=0.16,
        width_ratios=[1.0, 1.0],
    )

    density_colors = ["#999999", "#0072B2"]

    for row_idx, item in enumerate(items):
        sample_sets = _build_m200_c_inferencedata(item, seed=42 + row_idx)

        density_grid = outer_grid[row_idx, 0].subgridspec(1, 2, wspace=0.18)
        density_axes = np.array([
            fig.add_subplot(density_grid[0, 0]),
            fig.add_subplot(density_grid[0, 1]),
        ], dtype=object)
        _plot_density_comparison(
            density_axes[0],
            sample_sets["prior"]["M200"],
            sample_sets["posterior"]["M200"],
            xlabel="M200",
            colors_for_series=density_colors,
            show_legend=(row_idx == 0),
        )
        _plot_density_comparison(
            density_axes[1],
            sample_sets["prior"]["c"],
            sample_sets["posterior"]["c"],
            xlabel="c",
            colors_for_series=density_colors,
            show_legend=False,
        )
        for axis, var_name in zip(density_axes, ["M200", "c"]):
            axis.set_title(f"{var_name}", fontsize=9)
        _style_density_axes(density_axes, show_legend=(row_idx == 0))

        pair_grid = outer_grid[row_idx, 1].subgridspec(2, 2, wspace=0.05, hspace=0.05)
        pair_axes = np.array([
            [fig.add_subplot(pair_grid[0, 0]), fig.add_subplot(pair_grid[0, 1])],
            [fig.add_subplot(pair_grid[1, 0]), fig.add_subplot(pair_grid[1, 1])],
        ], dtype=object)
        _plot_pair_summary(
            pair_axes,
            sample_sets["posterior"]["M200"],
            sample_sets["posterior"]["c"],
        )
        _style_pair_axes(np.asarray(pair_axes, dtype=object))

    _add_m200_c_summary_column_titles(fig, outer_grid)
    _add_m200_c_summary_row_labels(fig, outer_grid, items)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _save_figure_file(fig, output_path, dpi=dpi)
    _save_figure_file(fig, output_path.with_suffix(".pdf"), format="pdf")
    plt.close(fig)
    return output_path


def plot_m200_c_summary_panels(
    plateifus: list[str],
    output_path: Path,
    radius_count: int = 1000,
    dpi: int = 300,
    items: list[GalaxyFigureData] | None = None,
) -> list[Path]:
    if len(plateifus) != 4:
        raise ValueError(f"This figure expects exactly 4 galaxies, got {len(plateifus)}")

    if items is None:
        items = _prepare_galaxy_figure_items(plateifus, radius_count=radius_count)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel_labels = ["a", "b", "c", "d"]
    density_colors = ["#999999", "#0072B2"]
    saved_paths: list[Path] = []

    for row_idx, item in enumerate(items):
        sample_sets = _build_m200_c_inferencedata(item, seed=42 + row_idx)
        panel_label = panel_labels[row_idx]

        density_fig = plt.figure(figsize=(4.5, 3.8))
        density_grid = density_fig.add_gridspec(
            1,
            2,
            left=0.12,
            right=0.97,
            top=0.84,
            bottom=0.16,
            wspace=0.18,
        )
        density_fig.suptitle(item.plateifu, fontsize=12)
        density_axes = np.array(
            [
                density_fig.add_subplot(density_grid[0, 0]),
                density_fig.add_subplot(density_grid[0, 1]),
            ],
            dtype=object,
        )
        _plot_density_comparison(
            density_axes[0],
            sample_sets["prior"]["M200"],
            sample_sets["posterior"]["M200"],
            xlabel="M200",
            colors_for_series=density_colors,
            show_legend=True,
        )
        _plot_density_comparison(
            density_axes[1],
            sample_sets["prior"]["c"],
            sample_sets["posterior"]["c"],
            xlabel="c",
            colors_for_series=density_colors,
            show_legend=False,
        )
        for axis, var_name in zip(density_axes, ["M200", "c"]):
            axis.set_title(var_name, fontsize=9)
        _style_density_axes(density_axes, show_legend=True)
        density_path = output_path.with_name(
            f"{output_path.stem}_density_panel_{panel_label}{output_path.suffix}"
        )
        saved_paths.append(_save_panel_figure(density_fig, density_path, dpi))

        pair_fig = plt.figure(figsize=(4.2, 4.0))
        pair_grid = pair_fig.add_gridspec(
            2,
            2,
            left=0.14,
            right=0.97,
            top=0.88,
            bottom=0.14,
            wspace=0.05,
            hspace=0.05,
        )
        pair_fig.suptitle(item.plateifu, fontsize=12)
        pair_axes = np.array(
            [
                [pair_fig.add_subplot(pair_grid[0, 0]), pair_fig.add_subplot(pair_grid[0, 1])],
                [pair_fig.add_subplot(pair_grid[1, 0]), pair_fig.add_subplot(pair_grid[1, 1])],
            ],
            dtype=object,
        )
        _plot_pair_summary(
            pair_axes,
            sample_sets["posterior"]["M200"],
            sample_sets["posterior"]["c"],
        )
        _style_pair_axes(np.asarray(pair_axes, dtype=object))

        pair_path = output_path.with_name(
            f"{output_path.stem}_pair_panel_{panel_label}{output_path.suffix}"
        )
        saved_paths.append(_save_panel_figure(pair_fig, pair_path, dpi))

    return saved_paths


def _plot_image(ax: plt.Axes, item: GalaxyFigureData) -> None:
    img = plt.imread(str(item.image_path))
    if img.ndim == 2:
        ax.imshow(img, origin="upper", cmap="gray")
    else:
        ax.imshow(img, origin="upper")
    ax.set_title(item.plateifu, fontsize=12, pad=8)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _plot_velocity_map(
    ax: plt.Axes,
    map_data: np.ndarray,
    item: GalaxyFigureData,
    show_ylabel: bool,
) -> None:
    im = ax.imshow(
        map_data,
        origin="lower",
        cmap="RdBu_r",
        norm=item.map_norm,
        extent=item.extent,
        interpolation="nearest",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("$\\Delta \\alpha$ [arcsec]", fontsize=9)
    if show_ylabel:
        ax.set_ylabel("$\\Delta \\delta$ [arcsec]", fontsize=9)
    else:
        ax.set_ylabel("")
    ax.tick_params(labelsize=8)
    return im


def _add_column_colorbar(ax: plt.Axes, mappable, ticks: list[float]) -> None:
    cax = ax.inset_axes([0.02, 1.07, 0.96, 0.07])
    cbar = plt.colorbar(mappable, cax=cax, orientation="horizontal")
    cbar.set_ticks(ticks)
    cbar.ax.tick_params(labelsize=8, pad=1)
    cbar.set_label("$V_{los}$ [km s$^{-1}$]", fontsize=8, labelpad=1)


def _add_parameter_box(
    ax: plt.Axes,
    text: str,
    x: float,
    y: float,
    ha: str = "right",
    va: str = "bottom",
) -> None:
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=8.8,
        color=COLOR_PARAM_TEXT,
        linespacing=1.15,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "pad": 2.0},
    )


def _plot_rc_with_eti(ax: plt.Axes, item: GalaxyFigureData, show_ylabel: bool) -> None:
    r_obs = _signed_radius(item.r_disp_map, item.v_disp_map).ravel()
    v_obs = np.asarray(item.v_disp_map, dtype=float).ravel()
    obs_mask = np.isfinite(r_obs) & np.isfinite(v_obs)

    ax.scatter(
        r_obs[obs_mask],
        v_obs[obs_mask],
        color=COLOR_DATA_POINTS,
        s=8,
        alpha=0.55,
        linewidths=0,
        label="Observed rotation velocity data",
        zorder=4,
    )

    model_mask = np.isfinite(item.r_model) & np.isfinite(item.v_model)
    r_model = np.asarray(item.r_model[model_mask], dtype=float)
    v_model = np.asarray(item.v_model[model_mask], dtype=float)
    sort_idx = np.argsort(r_model)
    r_model = r_model[sort_idx]
    v_model = v_model[sort_idx]

    if item.v_eti_low is not None and item.v_eti_high is not None:
        v_low = np.asarray(item.v_eti_low[model_mask], dtype=float)[sort_idx]
        v_high = np.asarray(item.v_eti_high[model_mask], dtype=float)[sort_idx]
        v_low, v_high = np.minimum(v_low, v_high), np.maximum(v_low, v_high)
        ax.fill_between(
            r_model,
            v_low,
            v_high,
            color=COLOR_MODEL,
            alpha=0.16,
            linewidth=0,
            label="Inferred rotation curve: ETI [2.5%, 97.5%]",
            zorder=1,
        )

    ax.plot(
        r_model,
        v_model,
        color=COLOR_MODEL,
        linewidth=1.6,
        label="Inferred rotation curve: Median",
        zorder=3,
    )

    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.axvline(0.0, color="black", linewidth=0.6)
    ax.set_xlabel("$r$ [kpc]", fontsize=9)
    if show_ylabel:
        ax.set_ylabel("$V$ [km s$^{-1}$]", fontsize=9)
    ax.tick_params(labelsize=8)
    _add_parameter_box(ax, item.rc_annotation, x=0.97, y=0.07, ha="right", va="bottom")


def _plot_rc_fit_summary(ax: plt.Axes, item: GalaxyFigureData, show_ylabel: bool) -> None:
    r_obs = _signed_radius(item.r_disp_map, item.v_disp_map).ravel()
    v_obs = np.asarray(item.v_disp_map, dtype=float).ravel()
    obs_mask = np.isfinite(r_obs) & np.isfinite(v_obs)

    ax.scatter(
        r_obs[obs_mask],
        v_obs[obs_mask],
        color=COLOR_DATA_POINTS,
        s=10,
        alpha=0.55,
        linewidths=0,
        label="Observed data",
        zorder=4,
    )

    model_mask = np.isfinite(item.r_rc_fit) & np.isfinite(item.v_rc_fit)
    r_model = np.asarray(item.r_rc_fit[model_mask], dtype=float)
    v_model = np.asarray(item.v_rc_fit[model_mask], dtype=float)
    sort_idx = np.argsort(r_model)
    r_model = r_model[sort_idx]
    v_model = v_model[sort_idx]

    if item.v_rc_hdi_low is not None and item.v_rc_hdi_high is not None:
        v_low = np.asarray(item.v_rc_hdi_low[model_mask], dtype=float)[sort_idx]
        v_high = np.asarray(item.v_rc_hdi_high[model_mask], dtype=float)[sort_idx]
        v_low, v_high = np.minimum(v_low, v_high), np.maximum(v_low, v_high)
        ax.fill_between(
            r_model,
            v_low,
            v_high,
            color=COLOR_MODEL,
            alpha=0.16,
            linewidth=0,
            label="Posterior HDI",
            zorder=1,
        )

    ax.plot(
        r_model,
        v_model,
        color=COLOR_MODEL,
        linewidth=1.7,
        label="Inferred RC Median",
        zorder=3,
    )

    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.axvline(0.0, color="black", linewidth=0.6)
    ax.set_title(item.plateifu, fontsize=12, pad=8)
    ax.set_xlabel("$r$ [kpc]", fontsize=9)
    if show_ylabel:
        ax.set_ylabel("$V$ [km s$^{-1}$]", fontsize=9)
    else:
        ax.set_ylabel("")
    ax.tick_params(labelsize=8)
    _add_parameter_box(ax, item.rc_annotation, x=0.97, y=0.07, ha="right", va="bottom")


def _plot_rc_components(ax: plt.Axes, item: GalaxyFigureData, show_ylabel: bool) -> None:
    plot_mask = np.isfinite(item.r_model) & (item.r_model >= 0)
    radius = np.asarray(item.r_model[plot_mask], dtype=float)
    total_velocity = np.asarray(item.v_model[plot_mask], dtype=float)
    dm_velocity = np.asarray(item.v_dm[plot_mask], dtype=float)
    star_velocity = np.asarray(item.v_star[plot_mask], dtype=float)

    order = np.argsort(radius)
    radius = radius[order]
    total_velocity = total_velocity[order]
    dm_velocity = dm_velocity[order]
    star_velocity = star_velocity[order]

    ax.plot(radius, total_velocity, color=COLOR_V_TOTAL, linewidth=1.6, label="Inferred Total velocity median")
    ax.plot(radius, dm_velocity, color=COLOR_V_DM, linewidth=1.4, linestyle="--", label="Inferred Dark Matter velocity median")
    ax.plot(radius, star_velocity, color=COLOR_V_STAR, linewidth=1.4, label="Inferred Stellar velocity median")

    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.set_xlabel("$r$ [kpc]", fontsize=9)
    if show_ylabel:
        ax.set_ylabel("$V$ [km s$^{-1}$]", fontsize=9)
    ax.tick_params(labelsize=8)
    _add_parameter_box(ax, item.nfw_annotation, x=0.97, y=0.07, ha="right", va="bottom")


def _add_row_labels(fig: plt.Figure, axes: np.ndarray) -> None:
    labels = [
        "Optical image",
        "Observed velocity field",
        "Model velocity field",
        "RC + 95% ETI + data",
        "RC components",
    ]
    for row_idx, label in enumerate(labels):
        bbox = axes[row_idx, 0].get_position()
        y_center = 0.5 * (bbox.y0 + bbox.y1)
        fig.text(0.035, y_center, label, rotation=90, va="center", ha="center", fontsize=11)


def _build_combined_legend(fig: plt.Figure) -> None:
    handles = [
        Line2D([0], [0], marker="o", linestyle="None", color=COLOR_DATA_POINTS, markersize=4.5, label="Observed data"),
        Patch(facecolor=COLOR_MODEL, alpha=0.16, edgecolor="none", label="95% ETI"),
        Line2D([0], [0], color=COLOR_MODEL, linewidth=1.6, label="Median RC"),
        Line2D([0], [0], color=COLOR_V_TOTAL, linewidth=1.6, label="Total"),
        Line2D([0], [0], color=COLOR_V_DM, linewidth=1.4, linestyle="--", label="Dark matter"),
        Line2D([0], [0], color=COLOR_V_STAR, linewidth=1.4, label="Stellar"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.04),
        ncol=3,
        frameon=False,
        fontsize=9,
        handlelength=2.0,
        columnspacing=1.6,
    )


def _rc_fit_legend_handles() -> list[Line2D | Patch]:
    return [
        Line2D([0], [0], marker="o", linestyle="None", color=COLOR_DATA_POINTS, markersize=4.5, label="Observed data"),
        Patch(facecolor=COLOR_MODEL, alpha=0.16, edgecolor="none", label="Posterior HDI"),
        Line2D([0], [0], color=COLOR_MODEL, linewidth=1.7, label="Inferred RC Median"),
    ]


def _build_rc_fit_legend(fig: plt.Figure) -> None:
    handles = _rc_fit_legend_handles()
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.03),
        ncol=3,
        frameon=False,
        fontsize=10,
        handlelength=2.2,
        columnspacing=1.8,
    )


def _velocity_panel_base_path(output_path: Path, panel_type: str, panel_label: str) -> Path:
    return output_path.with_name(f"{output_path.stem}_{panel_type}_panel_{panel_label}{output_path.suffix}")


def _save_panel_figure(fig: plt.Figure, output_path: Path, dpi: int) -> Path:
    saved_path = _save_figure_file(fig, output_path, dpi=dpi)
    _save_figure_file(fig, output_path.with_suffix(".pdf"), format="pdf")
    plt.close(fig)
    return saved_path


def plot_velocity_field_panels(
    plateifus: list[str],
    output_path: Path,
    radius_count: int = 1000,
    dpi: int = 300,
    items: list[GalaxyFigureData] | None = None,
) -> dict[str, list[Path]]:
    if len(plateifus) != 4:
        raise ValueError(f"This figure expects exactly 4 galaxies, got {len(plateifus)}")

    if items is None:
        items = _prepare_galaxy_figure_items(plateifus, radius_count=radius_count)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel_labels = ["a", "b", "c", "d"]
    saved_paths: dict[str, list[Path]] = {
        "observed": [],
        "model": [],
        "rc": [],
        "components": [],
    }

    for idx, item in enumerate(items):
        panel_label = panel_labels[idx]
        show_ylabel = True
        figure_width = 3.45
        left_margin = 0.18

        fig, ax = plt.subplots(figsize=(figure_width, 3.35))
        fig.subplots_adjust(left=left_margin, right=0.98, top=0.88, bottom=0.14)
        observed_im = _plot_velocity_map(ax, item.observed_map, item, show_ylabel=show_ylabel)
        if not show_ylabel:
            ax.tick_params(labelleft=False)
        _add_column_colorbar(ax, observed_im, item.map_ticks)
        observed_path = _velocity_panel_base_path(output_path, "observed", panel_label)
        saved_paths["observed"].append(_save_panel_figure(fig, observed_path, dpi))

        fig, ax = plt.subplots(figsize=(figure_width, 3.15))
        fig.subplots_adjust(left=left_margin, right=0.98, top=0.96, bottom=0.14)
        _plot_velocity_map(ax, item.model_map, item, show_ylabel=show_ylabel)
        if not show_ylabel:
            ax.tick_params(labelleft=False)
        model_path = _velocity_panel_base_path(output_path, "model", panel_label)
        saved_paths["model"].append(_save_panel_figure(fig, model_path, dpi))

        fig, ax = plt.subplots(figsize=(figure_width, 2.95))
        fig.subplots_adjust(left=left_margin, right=0.98, top=0.96, bottom=0.14)
        _plot_rc_with_eti(ax, item, show_ylabel=show_ylabel)
        if not show_ylabel:
            ax.tick_params(labelleft=False)
        rc_path = _velocity_panel_base_path(output_path, "rc", panel_label)
        saved_paths["rc"].append(_save_panel_figure(fig, rc_path, dpi))

        fig, ax = plt.subplots(figsize=(figure_width, 2.95))
        fig.subplots_adjust(left=left_margin, right=0.98, top=0.96, bottom=0.14)
        _plot_rc_components(ax, item, show_ylabel=show_ylabel)
        if not show_ylabel:
            ax.tick_params(labelleft=False)
        component_path = _velocity_panel_base_path(output_path, "components", panel_label)
        saved_paths["components"].append(_save_panel_figure(fig, component_path, dpi))

    return saved_paths


def plot_velocity_field_comparison(
    plateifus: list[str],
    output_path: Path,
    radius_count: int = 1000,
    dpi: int = 300,
    items: list[GalaxyFigureData] | None = None,
) -> Path:
    if len(plateifus) != 4:
        raise ValueError(f"This figure expects exactly 4 galaxies, got {len(plateifus)}")

    if items is None:
        items = _prepare_galaxy_figure_items(plateifus, radius_count=radius_count)

    fig, axes = plt.subplots(
        5,
        4,
        figsize=(16, 17.5),
        gridspec_kw={"height_ratios": [0.92, 1.08, 1.08, 1.0, 1.0]},
    )
    fig.subplots_adjust(left=0.08, right=0.985, top=0.97, bottom=0.10, hspace=0.24, wspace=0.16)

    for col_idx, item in enumerate(items):
        _plot_image(axes[0, col_idx], item)
        observed_im = _plot_velocity_map(
            axes[1, col_idx],
            item.observed_map,
            item,
            show_ylabel=(col_idx == 0),
        )
        _add_column_colorbar(axes[1, col_idx], observed_im, item.map_ticks)
        _plot_velocity_map(
            axes[2, col_idx],
            item.model_map,
            item,
            show_ylabel=(col_idx == 0),
        )
        _plot_rc_with_eti(axes[3, col_idx], item, show_ylabel=(col_idx == 0))
        _plot_rc_components(axes[4, col_idx], item, show_ylabel=(col_idx == 0))

        if col_idx != 0:
            axes[1, col_idx].tick_params(labelleft=False)
            axes[2, col_idx].tick_params(labelleft=False)
            axes[3, col_idx].set_ylabel("")
            axes[4, col_idx].set_ylabel("")

    _add_row_labels(fig, axes)
    _build_combined_legend(fig)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _save_figure_file(fig, output_path, dpi=dpi)
    _save_figure_file(fig, output_path.with_suffix(".pdf"), format="pdf")
    plt.close(fig)
    return output_path


def plot_rc_fit_summary_comparison(
    plateifus: list[str],
    output_path: Path,
    radius_count: int = 1000,
    dpi: int = 300,
    items: list[GalaxyFigureData] | None = None,
) -> Path:
    if len(plateifus) != 4:
        raise ValueError(f"This figure expects exactly 4 galaxies, got {len(plateifus)}")

    if items is None:
        items = _prepare_galaxy_figure_items(plateifus, radius_count=radius_count)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.0))
    fig.subplots_adjust(left=0.08, right=0.985, top=0.94, bottom=0.11, hspace=0.22, wspace=0.18)

    axes_array = np.asarray(axes, dtype=object)
    for idx, item in enumerate(items):
        row_idx, col_idx = divmod(idx, 2)
        _plot_rc_fit_summary(axes_array[row_idx, col_idx], item, show_ylabel=(col_idx == 0))

    _build_rc_fit_legend(fig)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _save_figure_file(fig, output_path, dpi=dpi)
    _save_figure_file(fig, output_path.with_suffix(".pdf"), format="pdf")
    plt.close(fig)
    return output_path


def plot_rc_fit_summary_panels(
    plateifus: list[str],
    output_path: Path,
    radius_count: int = 1000,
    dpi: int = 300,
    items: list[GalaxyFigureData] | None = None,
) -> list[Path]:
    if len(plateifus) != 4:
        raise ValueError(f"This figure expects exactly 4 galaxies, got {len(plateifus)}")

    if items is None:
        items = _prepare_galaxy_figure_items(plateifus, radius_count=radius_count)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    legend_handles = _rc_fit_legend_handles()
    panel_labels = ["a", "b", "c", "d"]

    for idx, item in enumerate(items, start=1):
        fig, ax = plt.subplots(figsize=(6.2, 4.9))
        fig.subplots_adjust(left=0.14, right=0.98, top=0.88, bottom=0.24)

        _plot_rc_fit_summary(ax, item, show_ylabel=True)

        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.05),
            ncol=3,
            frameon=False,
            fontsize=8,
            handlelength=2.0,
            columnspacing=1.2,
        )

        panel_base_path = output_path.with_name(
            f"empirical_rc_panel_{panel_labels[idx - 1]}{output_path.suffix}"
        )
        saved_paths.append(_save_figure_file(fig, panel_base_path, dpi=dpi))
        _save_figure_file(fig, panel_base_path.with_suffix(".pdf"), format="pdf")
        plt.close(fig)

    return saved_paths


def _parse_plateifus(raw_values: list[str]) -> list[str]:
    parsed = []
    for value in raw_values:
        for token in value.split(","):
            token = token.strip()
            if token:
                parsed.append(token)
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the 5x4 paper figure for four MaNGA galaxies.")
    parser.add_argument(
        "--ifu",
        nargs="*",
        default=DEFAULT_PLATE_IFUS,
        help="Four PLATE-IFU identifiers. You can pass them as separate values or comma-separated.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=result_dir / "velocity_field_comparison.png",
        help="Output PNG path. A PDF with the same stem will also be written.",
    )
    parser.add_argument(
        "--posterior-output",
        type=Path,
        default=result_dir / "m200_c_comparison.png",
        help="Output path for the M200/c prior-posterior summary figure. A PDF with the same stem will also be written.",
    )
    parser.add_argument(
        "--rcfit-output",
        type=Path,
        default=result_dir / "rc_fit_summary_comparison.png",
        help="Output path for the 2x2 RC-fit summary figure. A PDF with the same stem will also be written.",
    )
    parser.add_argument("--radius-count", type=int, default=1000, help="Number of radius samples used for RC and NFW reconstruction.")
    parser.add_argument("--dpi", type=int, default=300, help="Output figure DPI.")

    args = parser.parse_args()
    plateifus = _parse_plateifus(args.ifu)
    items = _prepare_galaxy_figure_items(plateifus, radius_count=args.radius_count)
    plot_path = plot_velocity_field_comparison(
        plateifus=plateifus,
        output_path=args.output,
        radius_count=args.radius_count,
        dpi=args.dpi,
        items=items,
    )
    posterior_plot_path = plot_m200_c_summary_comparison(
        plateifus=plateifus,
        output_path=args.posterior_output,
        radius_count=args.radius_count,
        dpi=args.dpi,
        items=items,
    )
    posterior_panel_paths = plot_m200_c_summary_panels(
        plateifus=plateifus,
        output_path=args.posterior_output,
        radius_count=args.radius_count,
        dpi=args.dpi,
        items=items,
    )
    rcfit_plot_path = plot_rc_fit_summary_comparison(
        plateifus=plateifus,
        output_path=args.rcfit_output,
        radius_count=args.radius_count,
        dpi=args.dpi,
        items=items,
    )
    rcfit_panel_paths = plot_rc_fit_summary_panels(
        plateifus=plateifus,
        output_path=args.rcfit_output,
        radius_count=args.radius_count,
        dpi=args.dpi,
        items=items,
    )
    velocity_panel_paths = plot_velocity_field_panels(
        plateifus=plateifus,
        output_path=args.output,
        radius_count=args.radius_count,
        dpi=args.dpi,
        items=items,
    )

    # print(f"Paper figure saved to {plot_path}")
    # print(f"M200/c summary figure saved to {posterior_plot_path}")
    # print(f"M200/c split panel figures saved to {[str(path) for path in posterior_panel_paths]}")
    # print(f"RC-fit summary figure saved to {rcfit_plot_path}")
    # print(f"RC-fit panel figures saved to {[str(path) for path in rcfit_panel_paths]}")
    # print(
    #     "Velocity-field panel figures saved to "
    #     f"{ {key: [str(path) for path in value] for key, value in velocity_panel_paths.items()} }"
    # )


if __name__ == "__main__":
    main()