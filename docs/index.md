---
title: MaNGA Dark Matter Documentation
---

# MaNGA Dark Matter Documentation

`manga-dm` is a MaNGA dark-matter analysis pipeline for rotation-curve fitting, NFW halo inference, and population-level `M200-c` modeling.

This documentation is organized for public review: start with the workflow overview, then use the method notes for Bayesian context and research direction.

## Start Here

| Page | Use it for |
|---|---|
| [Data Processing Pipeline](Data-Processing-Pipeline.md) | End-to-end CLI workflow, inputs, outputs, quality gates, and current limitations |
| [How and Why to Use MCMC](mcmc/how-and-why-to-use-mcmc.md) | Practical Bayesian inference background and the MaNGA NFW example |
| [Research Option: Inner Rotation-Curve Shapes](future/manga-dm-rc-shapes.md) | Future research direction for central dynamical concentration classes |

## Command-Line Workflow

```bash
pip install -e .
manga --help
manga select --download
manga stage1 --ifu test --nfw
manga merge --ifu-file data/plateifus.txt
manga stage2 --fit --quality-cut recommended
manga stage2 --diagnose --quality-cut recommended
```

| Command | Purpose |
|---|---|
| `manga select` | Select MaNGA plate-IFU targets and optionally download data |
| `manga stage1` | Fit per-galaxy rotation curves and optional NFW halos |
| `manga merge` | Merge per-galaxy posterior samples into one NetCDF file |
| `manga stage2` | Fit and diagnose the population-level `M200-c` relation |
| `manga figures` | Generate analysis figures |
| `manga sample` | Generate robustness subsamples |

## Project Layout

```text
src/
  cli/       command-line dispatch
  config/    settings and constants
  data/      catalog, FITS/MAPS, download, and result I/O
  models/    rotation-curve, NFW, and population models
  pipeline/  workflow orchestration
  stats/     diagnostics and interval utilities
  viz/       figures and plotting helpers
```

The public entry point is `manga`; `python -m src` is equivalent for local development.
