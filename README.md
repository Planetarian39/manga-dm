# MaNGA Dark Matter

`manga-dm` is a research pipeline for MaNGA dark-matter analysis. It selects MaNGA galaxies, fits gas rotation curves, runs single-galaxy NFW halo inference, merges posterior samples, and fits a population-level halo mass-concentration relation.

The current implementation lives in `src/` and is exposed through the `manga` command-line interface.

## What It Does

- Selects MaNGA DR17 galaxy samples and optionally downloads required data products.
- Fits per-galaxy rotation curves from H-alpha velocity maps.
- Runs Bayesian NFW halo inference with PyMC for galaxies that pass quality gates.
- Merges per-galaxy posterior samples into a common NetCDF product.
- Fits and diagnoses the population-level `M200-c` relation.
- Generates figures and robustness subsamples for analysis workflows.

## Repository Status

This is a scientific research codebase. The public interface is the `manga` CLI and the modules under `src/`. Legacy scripts are kept only for reference and are not required for the documented workflow.

## Documentation

- [Documentation home](docs/index.md)
- [Data processing pipeline](docs/Data-Processing-Pipeline.md)
- [MCMC guide for Bayesian inference](docs/mcmc/how-and-why-to-use-mcmc.md)
- [Research option: inner rotation-curve shapes](docs/future/manga-dm-rc-shapes.md)

## Installation

The documented environment is Windows with Anaconda. Python 3.11 or newer is required.

```bash
conda create -c conda-forge -n manga-dm python=3.12
conda activate manga-dm
```

Install the core scientific stack with conda:

```bash
conda install -c conda-forge nomkl numpy scipy lmfit
conda install -c conda-forge pytensor pymc arviz nutpie dm-tree
conda install -c conda-forge jax jaxlib
conda install -c conda-forge pandas pytz h5py pyarrow fsspec s3fs bottleneck certifi tqdm mpmath jplephem beautifulsoup4 html5lib bleach
conda install -c conda-forge ipython jupyter dask
conda install -c conda-forge astropy astroquery reproject asdf-astropy
conda install -c conda-forge pvextractor
conda install -c conda-forge matplotlib seaborn xarray-einstats numba
conda install -c conda-forge m2w64-toolchain libpython
```

Install remaining packages and the local package:

```bash
pip install ppxf spectral_cube sdss-access
pip install -e .
```

## Quick Start

```bash
manga --help
manga select --download
manga stage1 --ifu test --nfw
manga merge --ifu-file data/plateifus.txt
manga stage2 --fit --quality-cut recommended
manga stage2 --diagnose --quality-cut recommended
```

The equivalent module entry point is:

```bash
python -m src --help
```

## Workflow

| Step | Command | Output |
|---|---|---|
| Select targets | `manga select --download` | `data/plateifus.txt` and downloaded MaNGA products |
| Fit one or more galaxies | `manga stage1 --ifu test --nfw` | rotation-curve rows and NFW posterior files |
| Merge samples | `manga merge --ifu-file data/plateifus.txt` | merged posterior NetCDF |
| Fit population relation | `manga stage2 --fit --quality-cut recommended` | population `M200-c` fit products |
| Run diagnostics | `manga stage2 --diagnose --quality-cut recommended` | PSIS diagnostics |
| Generate figures | `manga figures --ifu 8994-12701 7977-3704` | analysis figures |
| Draw subsamples | `manga sample --n 60` | robustness sample files |

## CLI Reference

```bash
manga <subcommand> --help
```

| Subcommand | Purpose |
|---|---|
| `select` | Select galaxy sample and optionally download data |
| `stage1` | Run single-galaxy rotation-curve and NFW fitting |
| `merge` | Merge per-galaxy posterior sample files |
| `stage2` | Run population inference and diagnostics |
| `figures` | Generate analysis figures |
| `sample` | Generate robustness subsamples |

Global options include `--config`, `--data-dir`, `--result-dir`, and `--verbose`.

## Configuration

Runtime configuration is resolved through `src.config.settings`. Prefer CLI overrides or the supported `config.toml` lookup path instead of hard-coding local directories.

Common defaults:

| Purpose | Default |
|---|---|
| Data root | `data/` |
| Result directory | `data/results/` |
| Selected IFU list | `data/plateifus.txt` |
| Rotation-curve table | `rc_param.csv` |
| NFW parameter table | `nfw_param_cm200.csv` |
| Merged posterior samples | `nfw_param_cm200_samples.nc` |

## Python Modules

The package is organized by responsibility:

```text
src/
  __main__.py    # python -m src -> CLI
  cli/           # argparse dispatch only
  config/        # settings and constants
  data/          # catalogs, FITS/MAPS access, result I/O
  models/        # rotation-curve, NFW, and population models
  pipeline/      # workflow orchestration
  stats/         # statistical utilities
  viz/           # figure generation and plotting helpers
```

Useful imports:

```python
from src.config import settings
from src.pipeline.selection import select_and_download
from src.pipeline.stage1 import run_stage1
from src.pipeline.stage2 import run_stage2, merge_samples
from src.models.rotation_curve import RotCurve
from src.models.dm_nfw import DmNfw
```

## Smoke Checks

These checks do not require local FITS data:

```bash
python -m src --help
manga --help
manga select --help
manga stage1 --help
manga stage2 --help
manga figures --help
manga merge --help
manga sample --help
python -m unittest discover -v
```
