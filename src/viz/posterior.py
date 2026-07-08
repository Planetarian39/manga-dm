"""Posterior-distribution visualization helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from src.config.constants import (
    ALPHA_PRIOR_MEAN,
    ALPHA_PRIOR_SIGMA,
    COLOR_DATA_POINTS,
    COLOR_HDI_BAND,
    COLOR_HIGH_N,
    COLOR_LOW_N,
    LOG10_C0_PRIOR_MEAN,
    LOG10_C0_PRIOR_SIGMA,
)
from src.config.settings import settings
from src.models.relations import H_0, log10_c_m200_relation_profile
from src.stats.intervals import format_pair_interval_title
from src.viz.utils import plot_posterior_1d_hdi


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


def _build_prior_posterior_density_data(
    az_api,
    posterior_samples: dict[str, np.ndarray],
    prior_draw_count: int,
) -> tuple[object, object]:
    rng = np.random.default_rng(42)
    prior_samples = {
        "log10_c0": rng.normal(
            loc=LOG10_C0_PRIOR_MEAN,
            scale=LOG10_C0_PRIOR_SIGMA,
            size=prior_draw_count,
        ),
        "alpha": rng.normal(
            loc=ALPHA_PRIOR_MEAN,
            scale=ALPHA_PRIOR_SIGMA,
            size=prior_draw_count,
        ),
    }
    prior = az_api.from_dict(
        posterior={key: value[None, :] for key, value in prior_samples.items()}
    )
    posterior = az_api.from_dict(
        posterior={
            key: np.asarray(value, dtype=float).reshape(1, -1)
            for key, value in posterior_samples.items()
        }
    )
    return prior, posterior


def plot_population_posterior_diagnostics(
    *,
    trace,
    posterior,
    az_api,
    log10_c0_samples: np.ndarray,
    alpha_samples: np.ndarray,
    dataset_label: str,
    dataset_tag: str,
    result_dir: str | Path,
    hdi_prob1: float,
    hdi_prob2: float,
) -> None:
    """Save population posterior density and pair diagnostic plots."""
    result_dir = Path(result_dir)
    try:
        prior_draw_count = max(len(log10_c0_samples), len(alpha_samples), 1000)
        prior_idata, posterior_idata = _build_prior_posterior_density_data(
            az_api,
            posterior_samples={
                "log10_c0": np.asarray(log10_c0_samples, dtype=float),
                "alpha": np.asarray(alpha_samples, dtype=float),
            },
            prior_draw_count=prior_draw_count,
        )

        def _save_single_density_plot(
            var_name: str,
            samples: np.ndarray,
            title_prefix: str,
            color: str,
            *,
            save_combined: bool = True,
        ) -> None:
            if save_combined:
                fig, axes = plt.subplots(2, 1, figsize=(6.2, 7.0))
                az_api.plot_density(
                    [prior_idata, posterior_idata],
                    data_labels=["Prior", "Posterior"],
                    var_names=[var_name],
                    ax=np.atleast_1d(axes[0]),
                    point_estimate=None,
                    hdi_prob=hdi_prob2,
                    shade=0.15,
                    colors=["#9A9A9A", color],
                    outline=True,
                    textsize=8,
                )
                axes[0].set_title(f"{title_prefix} Prior vs Posterior")
                axes[0].set_xlabel(title_prefix)
                plot_posterior_1d_hdi(
                    samples,
                    title=f"{title_prefix} Posterior KDE",
                    base_color=color,
                    ax=axes[1],
                    hdi_probs=(hdi_prob1, hdi_prob2),
                    show_interval_bars=False,
                )
                fig.tight_layout()
                out = result_dir / f"c-M_relation_posterior_{dataset_tag}_{var_name}.png"
                fig.savefig(out, dpi=300, bbox_inches="tight")
                fig.savefig(out.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
                plt.close(fig)

            density_fig, density_ax = plt.subplots(1, 1, figsize=(6.2, 3.2))
            az_api.plot_density(
                [prior_idata, posterior_idata],
                data_labels=["Prior", "Posterior"],
                var_names=[var_name],
                ax=np.atleast_1d(density_ax),
                point_estimate=None,
                hdi_prob=hdi_prob2,
                shade=0.15,
                colors=["#9A9A9A", color],
                outline=True,
                textsize=8,
            )
            density_ax.set_title(f"{title_prefix} Prior vs Posterior")
            density_ax.set_xlabel(title_prefix)
            density_fig.tight_layout()
            density_out = result_dir / f"c-M_relation_posterior_{dataset_tag}_{var_name}_density.png"
            density_fig.savefig(density_out, dpi=300, bbox_inches="tight")
            density_fig.savefig(density_out.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
            plt.close(density_fig)

            kde_fig, kde_ax = plt.subplots(1, 1, figsize=(6.2, 3.2))
            plot_posterior_1d_hdi(
                samples,
                title=f"{title_prefix} Posterior KDE",
                base_color=color,
                ax=kde_ax,
                hdi_probs=(hdi_prob1, hdi_prob2),
                show_interval_bars=False,
            )
            kde_fig.tight_layout()
            kde_out = result_dir / f"c-M_relation_posterior_{dataset_tag}_{var_name}_kde.png"
            kde_fig.savefig(kde_out, dpi=300, bbox_inches="tight")
            kde_fig.savefig(kde_out.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
            plt.close(kde_fig)

        skip_combined_plot_vars = {"log10_c0", "alpha"} if dataset_tag == "all" else set()
        _save_single_density_plot(
            "log10_c0",
            log10_c0_samples,
            r"$\log_{10} c_0$",
            COLOR_LOW_N,
            save_combined="log10_c0" not in skip_combined_plot_vars,
        )
        _save_single_density_plot(
            "alpha",
            alpha_samples,
            r"$\alpha$",
            COLOR_HIGH_N,
            save_combined="alpha" not in skip_combined_plot_vars,
        )
        print("Split prior/posterior and KDE plots saved for log10_c0 and alpha")
    except Exception as exc:
        print(f"Warning: Split prior/posterior and KDE plots failed: {exc}")

    try:
        pair_var_names_all = ["log10_c0", "alpha", "M200_mu", "M200_sigma", "sigma_int"]
        pair_axes_all = az_api.plot_pair(
            trace,
            var_names=pair_var_names_all,
            kind=["kde"],
            marginals=True,
            marginal_kwargs={
                "kind": "hist",
                "hist_kwargs": {
                    "bins": 30,
                    "histtype": "step",
                    "linewidth": 1.5,
                    "density": True,
                },
            },
            kde_kwargs={"hdi_probs": [hdi_prob1, hdi_prob2]},
            point_estimate=None,
            textsize=8,
            divergences=False,
        )
        annotate_pair_marginals_m200(
            pair_axes_all,
            posterior,
            pair_var_names_all,
            title_fontsize=9,
            plot_median_line=True,
        )
        for var_name in pair_var_names_all:
            pair_axes_single = az_api.plot_pair(
                trace,
                var_names=[var_name],
                kind=["kde"],
                marginals=True,
                marginal_kwargs={
                    "kind": "hist",
                    "hist_kwargs": {
                        "bins": 30,
                        "histtype": "step",
                        "linewidth": 1.5,
                        "density": True,
                    },
                },
                kde_kwargs={"hdi_probs": [hdi_prob1, hdi_prob2]},
                point_estimate=None,
                textsize=8,
                divergences=False,
            )
            pair_axes_single_array = np.asarray(pair_axes_single, dtype=object)
            fig = pair_axes_single_array.flat[0].figure
            fig.set_size_inches(4.2, 4.2)
            out = result_dir / f"c-M_relation_pair_{dataset_label}_{var_name}.png"
            fig.savefig(out, dpi=300, bbox_inches="tight")
            fig.savefig(out.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
            plt.close(fig)

        pair_axes_all_array = np.asarray(pair_axes_all, dtype=object)
        for ax in pair_axes_all_array.flat:
            if ax is not None:
                ax.set_xticks([])
                ax.set_yticks([])

        pair_all_fig = pair_axes_all_array.flat[0].figure
        pair_all_fig.set_size_inches(12, 10)
        pair_all_path = result_dir / f"c-M_relation_pair_{dataset_tag}.png"
        pair_all_fig.savefig(pair_all_path, dpi=300, bbox_inches="tight")
        pair_all_fig.savefig(pair_all_path.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
        plt.close(pair_all_fig)
        print(f"All-parameter pair plot saved to {pair_all_path}")
    except Exception as exc:
        print(f"Warning: All-parameter pair plot failed: {exc}")


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
