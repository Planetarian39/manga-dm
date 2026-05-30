"""Top-level paper-figure generation (multi-panel composites).

Extracted from ``src-orig/figure.py`` and ``src-orig/m200.py``.
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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
    from src.models.population import (
        H_0,
        _compute_linear_fit,
        log10_c_m200_relation_profile,
        reference_log10_c_band,
    )

    M200 = np.asarray(M200, dtype=float)
    c = np.asarray(c, dtype=float)
    valid_mask = (M200 > 0) & (c > 0) & np.isfinite(M200) & np.isfinite(c)
    if not np.any(valid_mask):
        return None

    M200 = M200[valid_mask]
    c = c[valid_mask]

    m_plot = np.logspace(np.log10(np.min(M200)), np.log10(np.max(M200)), 50)
    log10_m_plot = np.log10(m_plot)
    log10_m_pivot = np.log10(M_PIVOT_H_INV / H_0)

    fig, ax_top = plt.subplots(1, 1, figsize=(10, 6))
    ax_top.scatter(np.log10(M200), np.log10(c), color=COLOR_DATA_POINTS, alpha=0.7, label="Data Points", s=20)

    c_reference, c_reference_low, c_reference_high = reference_log10_c_band(
        m_plot,
        LOG10_C0_DM14,
        ALPHA_DM14,
        LOG10_C_SIGMA_DM14,
        h=H_0,
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

    c_li20, _, _ = reference_log10_c_band(
        m_plot,
        LOG10_C0_LI20,
        ALPHA_LI20,
        LOG10_C_SCATTER_LI20,
        log10_c0_sigma=LOG10_C0_SIGMA_LI20,
        alpha_sigma=ALPHA_SIGMA_LI20,
        h=H_0,
    )
    ax_top.plot(log10_m_plot, np.log10(c_li20), color=COLOR_LI20, linewidth=2, linestyle="--", label="Li et al. 2020 (SPARC)")

    c_yasin23, _, _ = reference_log10_c_band(
        m_plot,
        LOG10_C0_YASIN23,
        ALPHA_YASIN23,
        LOG10_C_SCATTER_YASIN23,
        log10_c0_sigma=LOG10_C0_SIGMA_YASIN23,
        alpha_sigma=ALPHA_SIGMA_YASIN23,
        h=H_0,
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
        c_median = log10_c_m200_relation_profile(m_plot, log10_c0_fit, alpha_fit, h=H_0)
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
                    log10_c_m200_relation_profile(m_plot, log10_c0_value, alpha_value, h=H_0)
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


def plot_sample_attrition_pipeline(*args, **kwargs):
    """Generate sample-attrition pipeline figure."""
    from src.models.population import plot_sample_attrition_pipeline as _impl

    return _impl(*args, **kwargs)
