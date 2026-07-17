# MaNGA Dark Matter

`manga-dm` is a Python research pipeline for MaNGA gas-kinematic screening, empirical rotation curves, single-galaxy NFW inference, posterior-sample management, and population-model diagnostics. The public interface is the `manga` CLI; scientific modules live under `src/`.

Developed by **Hongyi Xu**, an undergraduate student in the Department of Physics at the University of Toronto. The project accompanies a first-author manuscript in preparation for arXiv and subsequent journal submission. The public repository documents methods, software, and allowlisted single-galaxy artifacts without releasing aggregate manuscript findings.

## Documentation

- [Documentation home](https://planetarian39.github.io/manga-dm/)
- [Quick start](https://planetarian39.github.io/manga-dm/run/cli-workflow.html)
- [Method overview](https://planetarian39.github.io/manga-dm/methods/)
- [11743-9102 case study](https://planetarian39.github.io/manga-dm/case-studies/11743-9102.html)
- [About the researcher](https://planetarian39.github.io/manga-dm/about/)
- [Application snapshot](https://planetarian39.github.io/manga-dm/project/application-snapshot.html)

The site presents methods, implementation status, MCMC background, and four allowlisted single-galaxy examples. Unpublished aggregate findings, discussion, conclusions, paper sources, and full-sample products are intentionally excluded.

## Pipeline

```text
MaNGA products
  → sample and kinematic screening
  → empirical rotation-curve fit
  → single-galaxy stellar + NFW inference
  → posterior-sample merge
  → population fit and diagnostics
```

## Installation

The documented development environment is Windows with Anaconda and Python 3.11 or newer.

```powershell
conda create -c conda-forge -n manga-dm python=3.12
conda activate manga-dm
conda install -c conda-forge numpy scipy pandas xarray matplotlib astropy pymc arviz lmfit
pip install -e .
```

See the [installation guide](https://planetarian39.github.io/manga-dm/run/installation.html) for environment notes and troubleshooting.

## Quick start

```powershell
manga --help
manga select --download
manga stage1 --ifu test --nfw
manga merge --ifu-file data/plateifus.txt
manga stage2 --fit --quality-cut recommended
manga stage2 --diagnose --quality-cut recommended
```

The equivalent module entry point is `python -m src`.

> [!IMPORTANT]
> Current fallback thresholds and the default Stage 2 path are not a versioned manuscript-aligned profile. See the [implementation-status page](https://planetarian39.github.io/manga-dm/project/implementation-status.html) before making reproducibility claims.

## Repository layout

| Path | Responsibility |
|---|---|
| `src/cli/` | CLI parsing and dispatch |
| `src/config/` | Settings, constants, and path resolution |
| `src/data/` | Catalog, download, result, and posterior I/O |
| `src/models/` | Rotation-curve, NFW, and population models |
| `src/pipeline/` | Workflow orchestration |
| `src/stats/` | Statistical utilities and diagnostics |
| `src/viz/` | Figures and visual diagnostics |
| `docs/` | VitePress source and curated public assets |
| `tests/` | Python and documentation checks |

## Documentation development

```powershell
npm ci
npm run docs:dev
npm run docs:check
npm run docs:build
```

GitHub Pages deployment uses `.github/workflows/deploy-docs.yml` and the official Pages artifact/deploy flow. The repository Pages source must be configured as **GitHub Actions**.

## License

See [LICENSE](LICENSE).
