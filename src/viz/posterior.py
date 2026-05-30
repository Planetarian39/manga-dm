"""Posterior-distribution visualization helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from src.config.constants import COLOR_DATA_POINTS, COLOR_HDI_BAND, COLOR_HIGH_N, COLOR_LOW_N
from src.config.settings import settings
from src.stats.intervals import format_pair_interval_title


def _get_pair_plot_label(var_name: str) -> str:
    label_map = {
        "Mstar": r"M_\star",
        "M200": r"M_{200}",
        "c": "c",
        "v_sys": r"v_{\mathrm{sys}}",
        "inc": "i",
        "f_bulge": r"f_{\mathrm{bulge}}",
        "sigma_0": r"\sigma_0",
        "Re": "R_e",
        "sigma_int": r"\sigma_{\mathrm{int}}",
        "log10_c0": r"\log_{10} c_0",
        "alpha": r"\alpha",
        "M200_mu": r"\mu_{M200}",
        "M200_sigma": r"\sigma_{M200}",
    }
    return label_map.get(var_name, var_name)


def _get_pair_plot_unit_label(var_name: str) -> str:
    unit_map = {
        "Mstar": r"M_\odot",
        "M200": r"M_\odot",
        "v_sys": r"\mathrm{km\,s^{-1}}",
        "sigma_0": r"\mathrm{km\,s^{-1}}",
        "Re": r"\mathrm{kpc}",
        "sigma_int": r"\mathrm{km\,s^{-1}}",
    }
    return unit_map.get(var_name, "")


def _get_trace_values(trace, var_name: str) -> np.ndarray | None:
    if var_name not in trace:
        return None
    value = trace[var_name]
    values = getattr(value, "values", value)
    samples = np.asarray(values, dtype=float).reshape(-1)
    samples = samples[np.isfinite(samples)]
    return samples if samples.size >= 2 else None


def annotate_pair_marginals(
    pair_axes,
    flat_trace,
    plotted_var_names: list[str],
    title_fontsize: float = 9,
    plot_median_line: bool = False,
) -> None:
    """Annotate diagonal marginal plots in an ArviZ pair-plot grid."""
    axes = np.asarray(pair_axes, dtype=object)
    if axes.ndim != 2:
        return

    diagonal_count = min(len(plotted_var_names), axes.shape[0], axes.shape[1])
    for idx in range(diagonal_count):
        ax = axes[idx, idx]
        if ax is None:
            continue

        var_name = plotted_var_names[idx]
        samples = _get_trace_values(flat_trace, var_name)
        if samples is None:
            continue

        low, median, high = np.percentile(samples, [16, 50, 84])

        if plot_median_line:
            title_line_color = ax.title.get_color()
            if title_line_color in (None, "auto"):
                title_line_color = "0.35"

            for value in (median, low, high):
                ax.axvline(
                    value,
                    color=title_line_color,
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.85,
                    zorder=3,
                )

        lower_err = median - low
        upper_err = high - median
        unit_label = _get_pair_plot_unit_label(var_name)
        interval_text = format_pair_interval_title(median, lower_err, upper_err, unit_label)
        interval_text = interval_text[1:-1] if interval_text.startswith("$") and interval_text.endswith("$") else interval_text
        ax.set_title(
            rf"${_get_pair_plot_label(var_name)} = {interval_text}$",
            fontsize=title_fontsize,
            pad=4,
        )


def annotate_pair_marginals_m200(*args, **kwargs) -> None:
    """Population-model alias for pair-plot marginal annotations."""
    annotate_pair_marginals(*args, **kwargs)


def plot_population_inference_diagnostics(
    M200: np.ndarray,
    c: np.ndarray,
    fit_results: dict | None = None,
    plot_suffix: str = "",
    sample_m200=None,
    sample_c=None,
    sample_plot: int = 20,
    output_dir: str | Path | None = None,
) -> Path | None:
    """Create compact residual diagnostics for the population-level c-M fit."""
    if not fit_results:
        return None

    log10_c0_fit = fit_results.get("log10_c0_median")
    alpha_fit = fit_results.get("alpha_median")
    sigma_int = fit_results.get("sigma_int_median")
    if log10_c0_fit is None or alpha_fit is None or sigma_int is None:
        return None

    M200 = np.asarray(M200, dtype=float)
    c = np.asarray(c, dtype=float)
    valid_mask = (M200 > 0) & (c > 0) & np.isfinite(M200) & np.isfinite(c)
    if not np.any(valid_mask):
        return None

    from src.models.population import H_0, log10_c_m200_relation_profile

    M200 = M200[valid_mask]
    c = c[valid_mask]
    c_pred = log10_c_m200_relation_profile(M200, float(log10_c0_fit), float(alpha_fit), h=H_0)

    log10_residuals = np.log10(c) - np.log10(c_pred)
    sigma_int = max(float(sigma_int), 1e-6)
    standardized_residuals = log10_residuals / sigma_int

    within_1sigma = float(np.mean(np.abs(standardized_residuals) <= 1.0))
    within_2sigma = float(np.mean(np.abs(standardized_residuals) <= 2.0))

    hist_max = max(4.0, np.nanmax(np.abs(standardized_residuals)) * 1.1)
    bins = np.linspace(-hist_max, hist_max, 28)
    x_pdf = np.linspace(-hist_max, hist_max, 400)
    result_dir = Path(output_dir) if output_dir is not None else settings.result_dir
    result_dir.mkdir(parents=True, exist_ok=True)
    suffix = plot_suffix if plot_suffix else "_all"
    plot_path = result_dir / f"c-M_relation_diagnostics{suffix}.png"

    fig_resid = plt.figure(figsize=(6.0, 5.6))
    ax_resid = fig_resid.add_subplot(1, 1, 1)
    ax_resid.axhspan(-2 * sigma_int, 2 * sigma_int, color=COLOR_HDI_BAND, alpha=0.18)
    ax_resid.axhspan(-sigma_int, sigma_int, color=COLOR_LOW_N, alpha=0.10)
    ax_resid.axhline(0.0, color="0.2", linewidth=1.1)

    drew_samples = False
    if sample_m200 is not None and sample_c is not None:
        rng = np.random.default_rng(0)
        samples_m_masked = np.asarray(sample_m200, dtype=object)[valid_mask]
        samples_c_masked = np.asarray(sample_c, dtype=object)[valid_mask]
        pool_m200: list[np.ndarray] = []
        pool_resid: list[np.ndarray] = []
        for gal_m200, gal_c in zip(samples_m_masked, samples_c_masked):
            if gal_m200 is None or gal_c is None:
                continue
            arr_m = np.asarray(gal_m200, dtype=float)
            arr_c = np.asarray(gal_c, dtype=float)
            finite = np.isfinite(arr_m) & np.isfinite(arr_c)
            if not np.any(finite):
                continue
            arr_m, arr_c = arr_m[finite], arr_c[finite]
            if len(arr_m) > sample_plot:
                idx = rng.choice(len(arr_m), size=sample_plot, replace=False)
                arr_m, arr_c = arr_m[idx], arr_c[idx]
            m200_samples = 10.0 ** arr_m
            c_pred_samples = log10_c_m200_relation_profile(
                m200_samples,
                float(log10_c0_fit),
                float(alpha_fit),
                h=H_0,
            )
            pool_m200.append(m200_samples)
            pool_resid.append(arr_c - np.log10(c_pred_samples))
        if pool_m200:
            ax_resid.scatter(
                np.concatenate(pool_m200),
                np.concatenate(pool_resid),
                s=6,
                color=COLOR_DATA_POINTS,
                alpha=0.15,
                edgecolors="none",
                rasterized=True,
                zorder=1,
            )
            drew_samples = True
    if not drew_samples:
        ax_resid.scatter(M200, log10_residuals, s=18, color=COLOR_DATA_POINTS, alpha=0.55, edgecolors="none")
    ax_resid.set_xscale("log")
    ax_resid.set_xlabel(r"$M_{200} \ [M_\odot]$")
    ax_resid.set_ylabel(r"$\Delta \log_{10} c$")
    ax_resid.set_title("Residuals vs. Halo Mass", fontsize=11)
    fig_resid.savefig(plot_path.with_name(f"{plot_path.stem}_residuals{plot_path.suffix}"), dpi=300, bbox_inches="tight")
    fig_resid.savefig(plot_path.with_name(f"{plot_path.stem}_residuals.pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig_resid)

    fig_hist = plt.figure(figsize=(6.0, 5.6))
    ax_hist = fig_hist.add_subplot(1, 1, 1)
    ax_hist.hist(
        standardized_residuals,
        bins=bins,
        density=True,
        histtype="stepfilled",
        color=COLOR_LOW_N,
        alpha=0.22,
        edgecolor=COLOR_LOW_N,
        linewidth=1.2,
    )
    ax_hist.plot(x_pdf, norm.pdf(x_pdf, loc=0.0, scale=1.0), color=COLOR_HIGH_N, linewidth=1.8)
    ax_hist.axvline(0.0, color="0.2", linewidth=1.1)
    ax_hist.axvline(-1.0, color="0.5", linewidth=1.0, linestyle="--")
    ax_hist.axvline(1.0, color="0.5", linewidth=1.0, linestyle="--")
    ax_hist.set_xlabel(r"$\Delta \log_{10} c / \sigma_{\mathrm{int}}$")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title("Standardized Residual Distribution", fontsize=11)
    coverage_text = (
        rf"$|\Delta|/\sigma_{{\mathrm{{int}}}}\leq1$: {within_1sigma:.1%}" "\n"
        rf"$|\Delta|/\sigma_{{\mathrm{{int}}}}\leq2$: {within_2sigma:.1%}"
    )
    ax_hist.text(
        0.97,
        0.97,
        coverage_text,
        transform=ax_hist.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.7", alpha=0.85),
    )
    fig_hist.savefig(plot_path.with_name(f"{plot_path.stem}_hist{plot_path.suffix}"), dpi=300, bbox_inches="tight")
    fig_hist.savefig(plot_path.with_name(f"{plot_path.stem}_hist.pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig_hist)

    print(f"Population diagnostic plot saved to {plot_path}")
    return plot_path
