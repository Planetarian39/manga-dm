"""Shared visualization utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

from src.data.fits import FitsUtil

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]


def _posterior_value_formatter(samples, *interval_bounds):
    finite_values = np.asarray(samples, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return lambda value: f"{value:.3f}"

    candidate_widths = []
    for low, high in interval_bounds:
        width = float(high) - float(low)
        if np.isfinite(width) and width > 0:
            candidate_widths.append(width)

    scale = min(candidate_widths) if candidate_widths else float(np.nanstd(finite_values))
    if not np.isfinite(scale) or scale <= 0:
        scale = max(abs(float(np.nanmedian(finite_values))), 1.0)

    decimals = int(np.clip(np.ceil(-np.log10(scale)) + 1, 2, 6))

    def _format_value(value):
        value = float(value)
        if np.isfinite(value) and abs(value) >= 1000:
            exponent = int(np.floor(np.log10(abs(value))))
            scale_factor = 10 ** exponent
            return rf"${value / scale_factor:.1f}\times 10^{{{exponent}}}$"
        return f"{value:.{decimals}f}"

    return _format_value


def _posterior_histogram_bin_edges(samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    if samples.size < 2:
        raise ValueError("Not enough finite samples to compute histogram bins")

    unique_count = np.unique(samples).size
    if unique_count < 2:
        center = float(samples[0])
        half_width = max(abs(center) * 0.05, 1e-6)
        return np.array([center - half_width, center + half_width], dtype=float)

    for bin_strategy in ("auto", "sturges"):
        bin_edges = np.histogram_bin_edges(samples, bins=bin_strategy)
        if (
            bin_edges.size >= 2
            and np.all(np.isfinite(bin_edges))
            and np.all(np.diff(bin_edges) > 0)
        ):
            return bin_edges

    x_min, x_max = np.min(samples), np.max(samples)
    x_pad = max((x_max - x_min) * 0.10, 1e-6)
    return np.linspace(x_min - x_pad, x_max + x_pad, 11)


def _strip_math_wrappers(text: str) -> str:
    text = str(text).strip()
    if text.startswith("$") and text.endswith("$") and len(text) >= 2:
        return text[1:-1]
    return text


def _format_interval_supsub(point_value: float, low: float, high: float, formatter) -> str:
    point_value = float(point_value)
    low = float(low)
    high = float(high)
    if not (np.isfinite(point_value) and np.isfinite(low) and np.isfinite(high)):
        return str(formatter(point_value))

    lower_err = point_value - low
    upper_err = high - point_value
    if lower_err < 0 or upper_err < 0:
        return str(formatter(point_value))

    point_text = _strip_math_wrappers(formatter(point_value))
    lower_text = _strip_math_wrappers(formatter(lower_err))
    upper_text = _strip_math_wrappers(formatter(upper_err))
    return rf"${{{point_text}}}_{{-{lower_text}}}^{{+{upper_text}}}$"


def plot_posterior_1d_hdi(
    samples,
    title: str = "",
    base_color: str = "#4D4D4D",
    ax=None,
    hdi_probs=(0.6827, 0.9545),
    point_estimate: str = "median",
    annotate: bool = True,
    show_legend: bool = True,
    show_interval_bars: bool = False,
    fill_alphas=(0.45, 0.22),
    linewidth: float = 1.8,
):
    """Draw a single-variable posterior density histogram with ETI shading."""
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    if samples.size < 2:
        raise ValueError(f"Not enough finite samples to plot histogram for {title or 'posterior'}")

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.figure

    probs = tuple(sorted(float(prob) for prob in hdi_probs))
    if len(fill_alphas) < len(probs):
        last_alpha = fill_alphas[-1] if len(fill_alphas) > 0 else 0.2
        fill_alphas = tuple(fill_alphas) + (last_alpha,) * (len(probs) - len(fill_alphas))

    bin_edges = _posterior_histogram_bin_edges(samples)

    density, bin_edges = np.histogram(samples, bins=bin_edges, density=True)
    bin_left = bin_edges[:-1]
    bin_right = bin_edges[1:]
    bin_widths = np.diff(bin_edges)

    if point_estimate == "mean":
        point_value = float(np.mean(samples))
        point_label = "Mean"
    else:
        point_value = float(np.median(samples))
        point_label = "Median"

    intervals = []
    for prob in probs:
        tail = (1.0 - prob) / 2.0
        low, high = np.quantile(samples, [tail, 1.0 - tail])
        intervals.append((prob, (float(low), float(high))))
    formatter = _posterior_value_formatter(samples, *(bounds for _, bounds in intervals))

    ax.stairs(density, bin_edges, color=base_color, linewidth=linewidth)
    for (prob, (low, high)), alpha in reversed(list(zip(intervals, fill_alphas))):
        mask = (bin_left < high) & (bin_right > low)
        ax.bar(
            bin_left[mask],
            density[mask],
            width=bin_widths[mask],
            align="edge",
            color=base_color,
            alpha=alpha,
            label=f"{prob * 100:.0f}% ETI",
            edgecolor="none",
        )

    ax.axvline(point_value, color=base_color, linestyle="--", linewidth=1.2, label=point_label)

    y_max = float(np.max(density))
    if show_interval_bars:
        n_intervals = len(intervals)
        y_fracs = np.linspace(0.16, 0.08, n_intervals)
        tick_height = y_max * 0.035
        for y_frac, (_, (low, high)) in zip(y_fracs, intervals):
            y_bar = y_max * y_frac
            ax.hlines(y_bar, low, high, color=base_color, linewidth=2.0, alpha=0.9)
            ax.vlines(
                [low, high],
                y_bar - tick_height,
                y_bar + tick_height,
                color=base_color,
                linewidth=1.4,
                alpha=0.9,
            )

    if annotate:
        text_lines = []
        for prob, (low, high) in intervals:
            interval_text = _format_interval_supsub(point_value, low, high, formatter)
            text_lines.append(f"{point_label} ({prob:.0%} ETI) = {interval_text}")
        ax.text(
            0.98,
            0.98,
            "\n".join(text_lines),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.80, edgecolor="none"),
        )

    ax.set_ylim(bottom=0)
    if show_legend:
        ax.legend(frameon=False, fontsize=8, loc="upper left")

    stats_dict = {
        "point_estimate": point_value,
        "point_label": point_label,
        "intervals": {prob: {"low": low, "high": high} for prob, (low, high) in intervals},
        "formatter": formatter,
    }
    return fig, ax, stats_dict


class PlotUtil:
    fits_util = None

    def __init__(self, fits_util: FitsUtil) -> None:
        self.fits_util = fits_util

    def plot_galaxy_image(self, plateifu):
        image_file = self.fits_util.get_image_file(plateifu)
        if image_file is None or not image_file.exists():
            print(f"Warning: image file for {plateifu} does not exist.")
            return

        try:
            img = plt.imread(str(image_file))
        except Exception:
            return

        fig, ax = plt.subplots(figsize=(7, 6))
        if img.ndim == 2:
            ax.imshow(img, origin="upper", cmap="gray")
        else:
            ax.imshow(img, origin="upper")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        fig.tight_layout()
        plt.show()
        plt.close(fig)

    def plot_vel_map_bin(self, vel_map, uindx, ra_map, dec_map, pa_rad=None, title: str = ""):
        fig, ax = plt.subplots(figsize=(8, 6))

        ra_flat, dec_flat, vel_flat = ra_map.ravel(), dec_map.ravel(), vel_map.ravel()
        ra_u, dec_u, vel_u = ra_flat[uindx], dec_flat[uindx], vel_flat[uindx]

        valid_vel_mask = np.isfinite(vel_u)
        vel_u_clean = vel_u[valid_vel_mask]

        if vel_u_clean.size == 0:
            vmin, vmax = -1.0, 1.0
        else:
            p_low, p_high = np.nanpercentile(vel_u_clean, [2, 98])
            vmax = max(abs(p_low), abs(p_high))
            vmin = -vmax
        norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

        sc = ax.scatter(ra_u, dec_u, c=vel_u, cmap="RdBu_r", norm=norm, s=30, edgecolors="face", alpha=0.9)
        cbar = fig.colorbar(sc, ax=ax, label="Velocity (km/s)")
        cbar.set_ticks([vmin, 0, vmax])

        if pa_rad is not None and ra_u[valid_vel_mask].size > 1:
            pa_rad = pa_rad % (2 * np.pi)
            ra_center = np.mean(ra_u[valid_vel_mask])
            dec_center = np.mean(dec_u[valid_vel_mask])
            ra_range = np.ptp(ra_u[valid_vel_mask])
            dec_range = np.ptp(dec_u[valid_vel_mask])
            line_length = 0.6 * np.hypot(ra_range, dec_range)

            dx = -line_length * np.sin(pa_rad)
            dy = line_length * np.cos(pa_rad)

            ax.plot(
                [ra_center - dx, ra_center + dx],
                [dec_center - dy, dec_center + dy],
                color="gray",
                linestyle="--",
                linewidth=1.5,
                label="Major Axis (PA)",
            )
            ax.legend()

        ax.set_xlabel("RA (deg)")
        ax.set_ylabel("Dec (deg)")
        ax.invert_xaxis()
        ax.set_aspect("equal", adjustable="box")

        fig.tight_layout()
        plt.show()
        plt.close(fig)

    def plot_vel_map(self, vel_map, title: str = ""):
        fig, ax = plt.subplots(figsize=(8, 6))

        valid_vel_mask = np.isfinite(vel_map)
        vel_clean = vel_map[valid_vel_mask]

        if vel_clean.size == 0:
            vmin, vmax = -1.0, 1.0
        else:
            p_low, p_high = np.nanpercentile(vel_clean, [2, 98])
            vmax = max(abs(p_low), abs(p_high))
            vmin = -vmax
        norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

        im = ax.imshow(vel_map, origin="lower", cmap="RdBu_r", norm=norm)
        cbar = fig.colorbar(im, ax=ax, label="Velocity (km/s)")
        cbar.set_ticks([vmin, 0, vmax])

        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        fig.tight_layout()
        plt.show()
        plt.close(fig)

    def plot_rv_curves(
        self,
        rv_data_list,
        plateifu: str = "",
        title: str = "",
        savedir: str | Path | None = None,
        savefilename: str | None = None,
    ):
        fig, ax = plt.subplots(figsize=(10, 6))

        for rv_data in rv_data_list:
            r_map = np.asarray(rv_data["r_map"], dtype=float)
            v_map = np.asarray(rv_data["V_map"], dtype=float)
            v_lower = rv_data.get("V_lower", None)
            v_upper = rv_data.get("V_upper", None)
            color = rv_data.get("color", None)
            size = rv_data.get("size", 2)
            linestyle = rv_data.get("linestyle", None)
            alpha = rv_data.get("alpha", 0.7 if linestyle else 0.2)
            label = rv_data.get("label", "Velocity")
            fill_alpha = rv_data.get("fill_alpha", 0.18)
            fill_label = rv_data.get("fill_label", None)
            zorder = rv_data.get("zorder", 3 if linestyle else 4)
            band_zorder = rv_data.get("band_zorder", 1)

            r_map = np.sign(v_map) * np.abs(r_map)

            r = r_map.ravel()
            v = v_map.ravel()

            valid = np.isfinite(r) & np.isfinite(v)
            r_valid = r[valid]
            v_valid = v[valid]

            if v_lower is not None and v_upper is not None:
                v_lower = np.asarray(v_lower, dtype=float).ravel()
                v_upper = np.asarray(v_upper, dtype=float).ravel()
                valid_band = np.isfinite(r) & np.isfinite(v_lower) & np.isfinite(v_upper)
                r_band = r[valid_band]
                v_lower_band = np.minimum(v_lower[valid_band], v_upper[valid_band])
                v_upper_band = np.maximum(v_lower[valid_band], v_upper[valid_band])
                sort_band = np.argsort(r_band)
                ax.fill_between(
                    r_band[sort_band],
                    v_lower_band[sort_band],
                    v_upper_band[sort_band],
                    color=color,
                    alpha=fill_alpha,
                    label=fill_label,
                    linewidth=0,
                    zorder=band_zorder,
                )

            if linestyle:
                sort_idx = np.argsort(r_valid)
                r_valid = r_valid[sort_idx]
                v_valid = v_valid[sort_idx]

                ax.plot(
                    r_valid,
                    v_valid,
                    alpha=alpha,
                    label=label,
                    color=color,
                    linestyle=linestyle,
                    zorder=zorder,
                )
            else:
                ax.scatter(r_valid, v_valid, alpha=alpha, label=label, color=color, marker="o", s=size, zorder=zorder)

        ax.set_xlabel("Radius R (kpc/h)")
        ax.set_ylabel("Velocity V (km/s)")
        ax.axhline(0, color="black", linestyle="-", linewidth=0.5)
        ax.axvline(0, color="black", linestyle="-", linewidth=0.5)
        ax.legend()
        fig.tight_layout()

        if savedir is not None:
            savedir = Path(savedir)
            savedir.mkdir(parents=True, exist_ok=True)
            filename = savefilename if savefilename is not None else f"{plateifu}_{title.replace(' ', '_')}"
            fig.savefig(savedir / f"{filename}.png", bbox_inches="tight")
            fig.savefig(savedir / f"{filename}.pdf", format="pdf", bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()
            plt.close(fig)
