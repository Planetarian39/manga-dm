"""Top-level paper-figure generation (multi-panel composites).

Extracted from ``src-orig/figure.py`` and ``src-orig/m200.py``.
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config.constants import (
    ALPHA_DM14,
    ALPHA_LI20,
    ALPHA_SIGMA_LI20,
    ALPHA_YASIN23,
    ALPHA_SIGMA_YASIN23,
    COLOR_DATA_POINTS,
    COLOR_DM14,
    COLOR_LI20,
    COLOR_POSTERIOR_MEDIAN,
    COLOR_YASIN23,
    H_ACTUAL,
    LOG10_C0_DM14,
    LOG10_C0_LI20,
    LOG10_C0_SIGMA_LI20,
    LOG10_C0_YASIN23,
    LOG10_C0_SIGMA_YASIN23,
    LOG10_C_SIGMA_DM14,
    LOG10_C_SCATTER_LI20,
    LOG10_C_SCATTER_YASIN23,
    M_PIVOT_H_INV,
)
from src.config.settings import settings
from src.data.catalog import load_sample_catalog_from_ifu_file


@dataclass
class GalaxyFigureData:
    """Data container for a single galaxy's figure preparation.

    Migrated from ``src-orig/figure.py`` (placeholder).
    """
    plate_ifu: str
    rc_params: dict | None = None
    nfw_params: dict | None = None


def _format_interval_supsub(value, low, high, decimals: int = 3) -> str:
    if value is None or low is None or high is None:
        return f"{float(value):.{decimals}f}" if value is not None else "n/a"
    value = float(value)
    low = float(low)
    high = float(high)
    if not (np.isfinite(value) and np.isfinite(low) and np.isfinite(high)):
        return f"{value:.{decimals}f}" if np.isfinite(value) else "n/a"
    return (
        rf"{value:.{decimals}f}_{{-{value - low:.{decimals}f}}}"
        rf"^{{+{high - value:.{decimals}f}}}"
    )


def _log10_c_m200_relation_profile(
    M200: np.ndarray,
    log10_c0: float,
    alpha: float,
    h: float = H_ACTUAL,
) -> np.ndarray:
    M_pivot = M_PIVOT_H_INV / h
    log10_c = log10_c0 + alpha * (np.log10(M200) - np.log10(M_pivot))
    return 10 ** log10_c


def _reference_log10_c_band(
    M200: np.ndarray,
    log10_c0: float,
    alpha: float,
    log10_c_scatter: float,
    log10_c0_sigma: float = 0.0,
    alpha_sigma: float = 0.0,
    sigma_scale: float = 2.0,
    h: float = H_ACTUAL,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = _log10_c_m200_relation_profile(M200, log10_c0, alpha, h=h)
    log10_center = np.log10(center)

    if log10_c0_sigma > 0 or alpha_sigma > 0:
        curves = np.vstack(
            [
                _log10_c_m200_relation_profile(
                    M200,
                    log10_c0 + log10_c0_sign * sigma_scale * log10_c0_sigma,
                    alpha + alpha_sign * sigma_scale * alpha_sigma,
                    h=h,
                )
                for log10_c0_sign in (-1.0, 1.0)
                for alpha_sign in (-1.0, 1.0)
            ]
        )
        log10_low = np.nanmin(np.log10(curves), axis=0)
        log10_high = np.nanmax(np.log10(curves), axis=0)
    else:
        log10_low = log10_center.copy()
        log10_high = log10_center.copy()

    if log10_c_scatter > 0:
        log10_low = log10_low - sigma_scale * log10_c_scatter
        log10_high = log10_high + sigma_scale * log10_c_scatter

    return center, 10 ** log10_low, 10 ** log10_high


def _compute_linear_fit(
    M200: np.ndarray,
    c: np.ndarray,
    log10_M_pivot: float,
) -> tuple[float | None, float | None]:
    if len(M200) < 3:
        return None, None

    log10_M = np.log10(M200)
    log10_c = np.log10(c)
    x = log10_M - log10_M_pivot
    design = np.column_stack([x, np.ones_like(x)])
    coeffs, *_ = np.linalg.lstsq(design, log10_c, rcond=None)
    return coeffs[1], coeffs[0]


def plot_m200_c_relation_all(
    M200: np.ndarray,
    c: np.ndarray,
    fit_results: dict | None = None,
    n_boot: int = 50,
    plot_suffix: str = "",
    sample_m200=None,
    sample_c=None,
    sample_plot: int = 20,
    output_dir: str | Path | None = None,
) -> Path | None:
    """Generate the full c-M200 relation figure."""
    M200 = np.asarray(M200, dtype=float)
    c = np.asarray(c, dtype=float)
    valid_mask = (M200 > 0) & (c > 0) & np.isfinite(M200) & np.isfinite(c)
    if not np.any(valid_mask):
        return None

    M200 = M200[valid_mask]
    c = c[valid_mask]

    m_plot = np.logspace(np.log10(np.min(M200)), np.log10(np.max(M200)), 50)
    log10_m_plot = np.log10(m_plot)
    log10_m_pivot = np.log10(M_PIVOT_H_INV / H_ACTUAL)

    fig, ax_top = plt.subplots(1, 1, figsize=(10, 6))
    ax_top.scatter(np.log10(M200), np.log10(c), color=COLOR_DATA_POINTS, alpha=0.7, label="Data Points", s=20)

    c_reference, c_reference_low, c_reference_high = _reference_log10_c_band(
        m_plot,
        LOG10_C0_DM14,
        ALPHA_DM14,
        LOG10_C_SIGMA_DM14,
        h=H_ACTUAL,
    )
    ax_top.plot(log10_m_plot, np.log10(c_reference), color=COLOR_DM14, linewidth=2, linestyle="--", label="Dutton & Maccio 2014")
    ax_top.fill_between(
        log10_m_plot,
        np.log10(c_reference_low),
        np.log10(c_reference_high),
        color=COLOR_DM14,
        alpha=0.18,
        label=rf"Dutton & Maccio (2014) $\pm 2\sigma$",
    )

    c_li20, _, _ = _reference_log10_c_band(
        m_plot,
        LOG10_C0_LI20,
        ALPHA_LI20,
        LOG10_C_SCATTER_LI20,
        log10_c0_sigma=LOG10_C0_SIGMA_LI20,
        alpha_sigma=ALPHA_SIGMA_LI20,
        h=H_ACTUAL,
    )
    ax_top.plot(log10_m_plot, np.log10(c_li20), color=COLOR_LI20, linewidth=2, linestyle="--", label="Li et al. 2020 (SPARC)")

    c_yasin23, _, _ = _reference_log10_c_band(
        m_plot,
        LOG10_C0_YASIN23,
        ALPHA_YASIN23,
        LOG10_C_SCATTER_YASIN23,
        log10_c0_sigma=LOG10_C0_SIGMA_YASIN23,
        alpha_sigma=ALPHA_SIGMA_YASIN23,
        h=H_ACTUAL,
    )
    ax_top.plot(log10_m_plot, np.log10(c_yasin23), color=COLOR_YASIN23, linewidth=2, linestyle="--", label="Yasin et al. 2023 (HI)")

    log10_c0_fit = fit_results.get("log10_c0_median") if fit_results else None
    alpha_fit = fit_results.get("alpha_median") if fit_results else None
    log10_c0_fit_eti_low = fit_results.get("log10_c0_eti_low") if fit_results else None
    log10_c0_fit_eti_high = fit_results.get("log10_c0_eti_high") if fit_results else None
    alpha_fit_eti_low = fit_results.get("alpha_eti_low") if fit_results else None
    alpha_fit_eti_high = fit_results.get("alpha_eti_high") if fit_results else None
    sigma_int = fit_results.get("sigma_int_median") if fit_results else None
    sigma_int_eti_low = fit_results.get("sigma_int_eti_low") if fit_results else None
    sigma_int_eti_high = fit_results.get("sigma_int_eti_high") if fit_results else None

    if log10_c0_fit is None or alpha_fit is None:
        log10_c0_fit, alpha_fit = _compute_linear_fit(M200, c, log10_m_pivot)

    if log10_c0_fit is not None and alpha_fit is not None:
        c_median = _log10_c_m200_relation_profile(m_plot, log10_c0_fit, alpha_fit, h=H_ACTUAL)
        ax_top.plot(log10_m_plot, np.log10(c_median), color=COLOR_POSTERIOR_MEDIAN, linewidth=2, label="Posterior Median")

        sigma_band = max(float(sigma_int), 1e-6) if sigma_int is not None else 0.0
        if all(
            value is not None
            for value in (
                log10_c0_fit_eti_low,
                log10_c0_fit_eti_high,
                alpha_fit_eti_low,
                alpha_fit_eti_high,
            )
        ):
            eti_curves = np.vstack(
                [
                    _log10_c_m200_relation_profile(m_plot, log10_c0_value, alpha_value, h=H_ACTUAL)
                    for log10_c0_value in (log10_c0_fit_eti_low, log10_c0_fit_eti_high)
                    for alpha_value in (alpha_fit_eti_low, alpha_fit_eti_high)
                ]
            )
            ax_top.fill_between(
                log10_m_plot,
                np.nanmin(np.log10(eti_curves), axis=0) - sigma_band,
                np.nanmax(np.log10(eti_curves), axis=0) + sigma_band,
                color=COLOR_POSTERIOR_MEDIAN,
                alpha=0.18,
                label=rf"{settings.HDI_PROB2:.0%} ETI",
            )
        elif sigma_band > 0:
            ax_top.fill_between(
                log10_m_plot,
                np.log10(c_median) - sigma_band,
                np.log10(c_median) + sigma_band,
                color=COLOR_POSTERIOR_MEDIAN,
                alpha=0.18,
                label=r"Median $\pm\ \sigma_{int}$",
            )

    log10_c_data = np.log10(c)
    ax_top.set_xlim(log10_m_plot[0] - 0.05, log10_m_plot[-1] + 0.05)
    ax_top.set_ylim(np.percentile(log10_c_data, 2) - 0.2, np.percentile(log10_c_data, 98) + 0.2)
    ax_top.set_xlabel(r"$\log_{10}(M_{200}/M_\odot)$", fontsize=12)
    ax_top.set_ylabel(r"$\log_{10}\,c_{200}$", fontsize=12)
    ax_top.legend(fontsize=10, loc="lower left")

    infer_text = (
        rf"Posterior Median (95% ETI): " "\n"
        rf"$\log_{{10}}c_0 = {_format_interval_supsub(log10_c0_fit, log10_c0_fit_eti_low, log10_c0_fit_eti_high)}$, "
        rf"$\alpha = {_format_interval_supsub(alpha_fit, alpha_fit_eti_low, alpha_fit_eti_high)}$" "\n"
        rf"$\sigma_{{int}} = {_format_interval_supsub(sigma_int, sigma_int_eti_low, sigma_int_eti_high)}$"
    )
    ax_top.text(
        0.98,
        0.02,
        infer_text,
        transform=ax_top.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    result_dir = Path(output_dir) if output_dir is not None else settings.result_dir
    result_dir.mkdir(parents=True, exist_ok=True)
    plot_path = result_dir / "c-M_relation_all.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    fig.savefig(plot_path.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Spaghetti plot saved to {plot_path}")
    return plot_path


def plot_m200_c_summary_comparison(*args, **kwargs):
    """Generate M200/c prior-posterior summary comparison plot."""
    from src.viz.figure_panels import plot_m200_c_summary_comparison as _impl

    return _impl(*args, **kwargs)


def plot_m200_c_summary_panels(*args, **kwargs):
    """Generate split M200/c prior-posterior summary panels."""
    from src.viz.figure_panels import plot_m200_c_summary_panels as _impl

    return _impl(*args, **kwargs)


def _hist_step_density(
    ax: plt.Axes,
    values: np.ndarray,
    bins: np.ndarray,
    label: str,
    color: str,
    linestyle: str,
    linewidth: float,
) -> None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return

    ax.hist(
        values,
        bins=bins,
        density=True,
        histtype="step",
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        label=label,
    )

    median = float(np.nanmedian(values))
    ymin, ymax = ax.get_ylim()
    tick_top = ymin + 0.10 * (ymax - ymin if ymax > ymin else 1.0)
    ax.vlines(
        median,
        ymin,
        tick_top,
        colors=color,
        linestyles=linestyle,
        linewidth=linewidth * 0.8,
        alpha=0.85,
    )


def _plot_attrition_metric_panel(
    ax: plt.Axes,
    stages: list[tuple[str, pd.DataFrame, str, str, float]],
    column_name: str,
    xlabel: str,
    *,
    show_legend: bool,
    legend_loc: str = "upper right",
) -> None:
    finite_values = []
    for _, frame, _, _, _ in stages:
        values = np.asarray(frame[column_name], dtype=float)
        values = values[np.isfinite(values)]
        if values.size > 0:
            finite_values.append(values)

    if finite_values:
        merged = np.concatenate(finite_values)
        x_min = float(np.nanmin(merged))
        x_max = float(np.nanmax(merged))
        if np.isclose(x_min, x_max):
            x_min -= 0.5
            x_max += 0.5
        bins = np.linspace(x_min, x_max, 26)
    else:
        bins = np.linspace(0.0, 1.0, 26)

    for label, frame, color, linestyle, linewidth in stages:
        _hist_step_density(
            ax,
            frame[column_name].to_numpy(dtype=float),
            bins,
            label,
            color,
            linestyle,
            linewidth,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_facecolor("#FCFCFC")
    ax.grid(True, alpha=0.18, linewidth=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=12)

    if show_legend:
        legend = ax.legend(frameon=False, fontsize=10.5, loc=legend_loc, handlelength=3.0)
        if legend is not None:
            legend._legend_box.align = "left"
            for text in legend.get_texts():
                text.set_ha("left")
                text.set_linespacing(1.1)


def _save_attrition_metric_panels(
    output_path: Path,
    stages: list[tuple[str, pd.DataFrame, str, str, float]],
) -> list[Path]:
    metric_specs = [
        ("redshift", "Redshift z", "redshift"),
        ("log10_mstar", r"$\log_{10}(M_\star / M_\odot)$", "mstar"),
        ("inclination", "Inclination [deg]", "inclination"),
        ("sersic_n", "Sersic n", "sersic_n"),
    ]
    saved_paths: list[Path] = []

    for column_name, xlabel, suffix in metric_specs:
        fig, ax = plt.subplots(figsize=(5.6, 4.2), facecolor="white")
        _plot_attrition_metric_panel(
            ax,
            stages,
            column_name,
            xlabel,
            show_legend=True,
            legend_loc="upper left",
        )
        panel_path = output_path.with_name(f"{output_path.stem}_{suffix}{output_path.suffix}")
        fig.savefig(panel_path, dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(panel_path.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(panel_path)

    return saved_paths


def plot_sample_attrition_pipeline(
    sample_specs: tuple[tuple[str | Path, str], tuple[str | Path, str]],
    output_path: Path | None = None,
) -> Path:
    """Generate the sample-attrition comparison figure."""
    sample_file_a, label_a = sample_specs[0]
    sample_file_b, label_b = sample_specs[1]
    sample_path_a, sample_catalog_a = load_sample_catalog_from_ifu_file(sample_file_a)
    sample_path_b, sample_catalog_b = load_sample_catalog_from_ifu_file(sample_file_b)

    if output_path is None:
        output_path = settings.result_dir / "galaxy_select_compare.png"

    print("[attrition] Comparing two IFU-list samples from --plot-attrition.")
    print(f"[attrition] Sample A label: {label_a}")
    print(f"[attrition] Sample A file: {sample_path_a.resolve()}")
    print(f"[attrition] Sample A galaxy count: {len(sample_catalog_a)}")
    print(f"[attrition] Sample B label: {label_b}")
    print(f"[attrition] Sample B file: {sample_path_b.resolve()}")
    print(f"[attrition] Sample B galaxy count: {len(sample_catalog_b)}")

    fig = plt.figure(figsize=(12.0, 7.6), facecolor="white")
    grid = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.20)
    ax_redshift = fig.add_subplot(grid[0, 0])
    ax_mstar = fig.add_subplot(grid[0, 1])
    ax_inclination = fig.add_subplot(grid[1, 0])
    ax_sersic = fig.add_subplot(grid[1, 1])

    stages = [
        (label_a, sample_catalog_a, "#D55E00", "-", 1.5),
        (label_b, sample_catalog_b, "#0072B2", "-", 1.5),
    ]
    metric_specs = [
        (ax_redshift, "redshift", "Redshift z"),
        (ax_mstar, "log10_mstar", r"$\log_{10}(M_\star / M_\odot)$"),
        (ax_inclination, "inclination", "Inclination [deg]"),
        (ax_sersic, "sersic_n", "Sersic n"),
    ]

    for idx, (ax, column_name, xlabel) in enumerate(metric_specs):
        _plot_attrition_metric_panel(
            ax,
            stages,
            column_name,
            xlabel,
            show_legend=(idx == 0),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_path.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)
    saved_panel_paths = _save_attrition_metric_panels(output_path, stages)
    print(f"Sample attrition metric panels saved to {[str(path) for path in saved_panel_paths]}")
    return output_path
