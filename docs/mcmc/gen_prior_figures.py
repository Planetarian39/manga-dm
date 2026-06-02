"""
Generate figures for the "Prior vs Data influence on posterior" chapter
in docs/mcmc/how-and-why-to-use-mcmc.md.

Run from repo root:
    conda run -n manga-dev python docs/mcmc/_gen_prior_figures.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import beta as beta_dist

# Use a CJK-compatible font for Chinese labels; fall back to DejaVu Sans
for _font in ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]:
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = [_font, "DejaVu Sans"]
    from matplotlib.font_manager import findfont, FontProperties
    if findfont(FontProperties(family=[_font])) != findfont(FontProperties(family=["DejaVu Sans"])):
        break  # found a valid CJK font
rcParams['axes.unicode_minus'] = False  # prevent minus sign rendering issue

OUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT_DIR, exist_ok=True)

MU_TRUE = 0.7
MU_GRID = np.linspace(0, 1, 500)

# Three priors: (alpha0, beta0, label, color)
PRIORS = [
    (2, 2,  "先验A: Beta(2,2)\n均值=0.50",  "C0"),
    (1, 8,  "先验B: Beta(1,8)\n均值=0.11",  "C1"),
    (8, 1,  "先验C: Beta(8,1)\n均值=0.89",  "C2"),
]

RNG = np.random.default_rng(42)


def simulate_flips(n):
    """Simulate n coin flips with true prob MU_TRUE."""
    return int(RNG.binomial(n, MU_TRUE))


def posterior(a0, b0, n, h):
    """Return Beta posterior given prior Beta(a0,b0) and n flips with h heads."""
    return beta_dist(a0 + h, b0 + (n - h))


# ── Figure 1: 3×2 grid, n = 3 / 10 / 30 / 100 / 300 / 1000 ──────────────────────────────
def fig1_n_comparison():
    ns = [3, 10, 30, 100, 300, 1000]
    heads = {n: simulate_flips(n) for n in ns}

    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle("不同数据量下三种先验的后验分布", fontsize=14)

    for ax, n in zip(axes.flat, ns):
        h = heads[n]
        for a0, b0, label, color in PRIORS:
            post = posterior(a0, b0, n, h)
            ax.plot(MU_GRID, post.pdf(MU_GRID), color=color, label=label.split("\n")[0], lw=2)
        ax.axvline(MU_TRUE, color="k", ls="--", lw=1.2, label=f"真实值 μ={MU_TRUE}")
        ax.set_title(f"n = {n}  (正面 {h} 次)", fontsize=12)
        ax.set_xlabel("μ")
        ax.set_ylabel("概率密度")
        ax.set_xlim(0, 1)
        ax.set_ylim(bottom=0)

    # single legend outside
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    path = os.path.join(OUT_DIR, "prior_posterior_n_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── Figure 2: posterior mean vs n for 3 priors ──────────────────────────────
def fig2_convergence():
    ns = np.arange(1, 201)
    # Cumulative flips: draw 200 and slice
    flips = RNG.integers(0, 2, size=200)  # 0=tails, 1=heads (biased)
    # Re-draw with correct bias
    flips = (RNG.random(200) < MU_TRUE).astype(int)
    cum_heads = np.cumsum(flips)

    fig, ax = plt.subplots(figsize=(8, 5))
    for a0, b0, label, color in PRIORS:
        post_means = (a0 + cum_heads) / (a0 + b0 + ns)
        ax.plot(ns, post_means, color=color, label=label.split("\n")[0], lw=2)

    ax.axhline(MU_TRUE, color="k", ls="--", lw=1.2, label=f"真实值 μ={MU_TRUE}")
    ax.set_xlabel("数据量 n")
    ax.set_ylabel("后验均值")
    ax.set_title("后验均值随数据量的收敛")
    ax.set_xlim(1, 200)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "prior_posterior_convergence.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── Figure 3: n=2 vs n=200 side-by-side ─────────────────────────────────────
def fig3_small_vs_large():
    ns = [2, 200]
    heads = {n: simulate_flips(n) for n in ns}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    titles = ["少量数据：n = 2", "大量数据：n = 200"]

    for ax, n, title in zip(axes, ns, titles):
        h = heads[n]
        for a0, b0, label, color in PRIORS:
            prior_pdf = beta_dist(a0, b0).pdf(MU_GRID)
            post_pdf  = posterior(a0, b0, n, h).pdf(MU_GRID)
            lbl = label.split("\n")[0]
            ax.plot(MU_GRID, prior_pdf, color=color, ls="--", lw=1.5, alpha=0.6, label=f"{lbl} 先验")
            ax.plot(MU_GRID, post_pdf,  color=color, ls="-",  lw=2.0,             label=f"{lbl} 后验")
        ax.axvline(MU_TRUE, color="k", ls=":", lw=1.2, label=f"真实值 μ={MU_TRUE}")
        ax.set_title(f"{title}  (正面 {h} 次)", fontsize=12)
        ax.set_xlabel("μ")
        ax.set_ylabel("概率密度")
        ax.set_xlim(0, 1)
        ax.set_ylim(bottom=0)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(rect=[0, 0.1, 1, 1])
    path = os.path.join(OUT_DIR, "prior_vs_posterior_small_large_n.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    fig1_n_comparison()
    fig2_convergence()
    fig3_small_vs_large()
    print("All figures generated.")
