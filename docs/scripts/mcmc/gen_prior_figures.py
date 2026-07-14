"""Generate the English prior-versus-data figures used by the MCMC guide.

Run from the repository root:

    python docs/scripts/mcmc/gen_prior_figures.py \
        --output-dir docs/public/assets/mcmc \
        --seed 42

Each figure owns a deterministic random stream derived from the base seed.
Calling one generator on its own therefore produces the same image data as
calling generate_all.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final

import matplotlib
import numpy as np
from scipy.stats import beta as beta_dist

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


MU_TRUE: Final = 0.7
MU_GRID: Final = np.linspace(0.0, 1.0, 500)
DEFAULT_OUTPUT_DIR: Final = (
    Path(__file__).resolve().parents[2] / "public" / "assets" / "mcmc"
)

# alpha, beta, label, color, line style
PRIORS: Final = (
    (2, 2, "Prior A: Beta(2, 2), mean 0.50", "#0072B2", "-"),
    (1, 8, "Prior B: Beta(1, 8), mean 0.11", "#D55E00", "--"),
    (8, 1, "Prior C: Beta(8, 1), mean 0.89", "#009E73", "-."),
)


def _rng(seed: int, stream: int) -> np.random.Generator:
    """Return an invocation-independent random stream."""
    return np.random.default_rng(np.random.SeedSequence([seed, stream]))


def _output_path(output_dir: Path | str, filename: str) -> Path:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / filename


def _posterior(alpha: float, beta: float, n: int, heads: int):
    return beta_dist(alpha + heads, beta + n - heads)


def format_head_count(heads: int) -> str:
    noun = "head" if heads == 1 else "heads"
    return f"{heads} {noun}"


def generate_posterior_by_sample_size(
    output_dir: Path | str, *, seed: int = 42
) -> Path:
    """Generate six posterior panels for increasing sample sizes."""
    rng = _rng(seed, 1)
    sample_sizes = (3, 10, 30, 100, 300, 1000)
    heads = {n: int(rng.binomial(n, MU_TRUE)) for n in sample_sizes}

    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle("Posterior distributions under three priors", fontsize=15)

    for axis, n in zip(axes.flat, sample_sizes):
        h = heads[n]
        for alpha, beta, label, color, line_style in PRIORS:
            distribution = _posterior(alpha, beta, n, h)
            axis.plot(
                MU_GRID,
                distribution.pdf(MU_GRID),
                color=color,
                linestyle=line_style,
                label=label.split(", mean")[0],
                linewidth=2,
            )
        axis.axvline(
            MU_TRUE,
            color="#222222",
            linestyle=":",
            linewidth=1.5,
            label=f"True value mu = {MU_TRUE}",
        )
        axis.set(
            title=f"n = {n} ({format_head_count(h)})",
            xlabel="Heads probability, mu",
            ylabel="Probability density",
            xlim=(0, 1),
        )
        axis.set_ylim(bottom=0)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    path = _output_path(output_dir, "posterior-by-sample-size.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_posterior_mean_by_sample_size(
    output_dir: Path | str, *, seed: int = 42
) -> Path:
    """Generate posterior means as observations accumulate."""
    rng = _rng(seed, 2)
    sample_sizes = np.arange(1, 201)
    flips = (rng.random(200) < MU_TRUE).astype(int)
    cumulative_heads = np.cumsum(flips)

    fig, axis = plt.subplots(figsize=(8, 5))
    for alpha, beta, label, color, line_style in PRIORS:
        means = (alpha + cumulative_heads) / (alpha + beta + sample_sizes)
        axis.plot(
            sample_sizes,
            means,
            color=color,
            linestyle=line_style,
            label=label.split(", mean")[0],
            linewidth=2,
        )

    axis.axhline(
        MU_TRUE,
        color="#222222",
        linestyle=":",
        linewidth=1.5,
        label=f"True value mu = {MU_TRUE}",
    )
    axis.set(
        title="Posterior mean as observations accumulate",
        xlabel="Number of observations, n",
        ylabel="Posterior mean",
        xlim=(1, 200),
        ylim=(0, 1),
    )
    axis.legend(fontsize=9)
    fig.tight_layout()
    path = _output_path(output_dir, "posterior-mean-by-sample-size.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_small_and_large_data(
    output_dir: Path | str, *, seed: int = 42
) -> Path:
    """Compare priors and posteriors for small and large data sets."""
    rng = _rng(seed, 3)
    sample_sizes = (2, 200)
    heads = {n: int(rng.binomial(n, MU_TRUE)) for n in sample_sizes}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    panel_titles = ("Small data", "Large data")

    for axis, n, title in zip(axes, sample_sizes, panel_titles):
        h = heads[n]
        for alpha, beta, label, color, line_style in PRIORS:
            prior_pdf = beta_dist(alpha, beta).pdf(MU_GRID)
            posterior_pdf = _posterior(alpha, beta, n, h).pdf(MU_GRID)
            short_label = label.split(":")[0]
            axis.plot(
                MU_GRID,
                prior_pdf,
                color=color,
                linestyle=line_style,
                linewidth=1.3,
                alpha=0.45,
                label=f"{short_label} prior",
            )
            axis.plot(
                MU_GRID,
                posterior_pdf,
                color=color,
                linestyle=line_style,
                linewidth=2.3,
                marker="o",
                markevery=80,
                markersize=3,
                label=f"{short_label} posterior",
            )
        axis.axvline(
            MU_TRUE,
            color="#222222",
            linestyle=":",
            linewidth=1.5,
            label=f"True value mu = {MU_TRUE}",
        )
        axis.set(
            title=f"{title}: n = {n} ({format_head_count(h)})",
            xlabel="Heads probability, mu",
            ylabel="Probability density",
            xlim=(0, 1),
        )
        axis.set_ylim(bottom=0)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.08),
    )
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    path = _output_path(output_dir, "prior-posterior-small-large-data.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_all(output_dir: Path | str, *, seed: int = 42) -> list[Path]:
    """Generate all maintained generic MCMC teaching figures."""
    return [
        generate_posterior_by_sample_size(output_dir, seed=seed),
        generate_posterior_mean_by_sample_size(output_dir, seed=seed),
        generate_small_and_large_data(output_dir, seed=seed),
    ]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic English figures for the MCMC guide."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed; each figure derives an independent stream.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    for path in generate_all(args.output_dir, seed=args.seed):
        print(f"Saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
