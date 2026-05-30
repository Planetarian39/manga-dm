# manga-dm

MaNGA dark matter analysis and fitting pipeline.

The current official implementation lives in the `src/` package. The official
command-line entry point is `manga`, with `python -m src` as an equivalent
module entry point.

## Development Environment

The documented development environment is Windows + Anaconda:

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

Install the remaining packages with pip:

```bash
pip install ppxf spectral_cube sdss-access
```

Install the package in editable mode:

```bash
pip install -e .
```

## CLI Usage

```bash
python -m src --help
manga --help
manga <subcommand> --help
```

| Subcommand | Description | Legacy equivalent |
|---|---|---|
| `manga select` | Select galaxy sample and optionally download data | `python src-orig/plates.py` |
| `manga stage1` | Single-galaxy RC + DM NFW fitting | `python src-orig/main.py` |
| `manga stage2` | Population inference and PSIS diagnostics | `python src-orig/m200.py` |
| `manga figures` | Generate paper figures | `python src-orig/figure.py` |
| `manga merge` | Merge posterior sample files | extracted from `src-orig/m200.py` |
| `manga sample` | Generate robustness sub-samples | extracted from `src-orig/m200.py` |

The installed `manga` command and `python -m src` run the current `src/`
package. `src-orig/` is retained only as a legacy compatibility layer and
should not receive new feature development.

## Typical Workflow

```bash
# 1. Select galaxies by inclination and download data
manga select --download

# 2. Stage 1: fit rotation curve + NFW profile for the test set
manga stage1 --ifu test --nfw

# 3. Stage 1: full sample from plateifus.txt
manga stage1 --ifu all --nfw --n-cores 8

# 4. Merge posterior samples after Stage 1 completes
manga merge --ifu-file data/plateifus.txt

# 5. Stage 2: population-level c-M200 relation
manga stage2 --fit --quality-cut recommended

# 6. Stage 2 diagnostics after a fit
manga stage2 --diagnose --quality-cut recommended

# 7. Generate figures for specific galaxies
manga figures --ifu 8994-12701 7977-3704

# 8. Generate robustness sub-samples
manga sample --n 60
```

## Smoke Tests Without Local Science Data

When FITS files and Stage 1/2 outputs are not available, command dispatch and
imports can still be checked:

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

## Python API

```python
# Configuration
from src.config import settings
print(settings.data_dir)
print(settings.result_dir)

# Constants
from src.config.constants import H0_PHYS, COLOR_V_STAR

# Data access
from src.data.catalog import DrpallUtil, get_plateifu_list
from src.data.fits import FitsUtil
from src.data.maps import MapsUtil
from src.data.firefly import FireflyUtil
from src.data.results import store_params_file, merge_posterior_samples_file

# Science models
from src.models.rotation_curve import RotCurve
from src.models.dm_nfw import DmNfw
from src.models.population import fit_m200_c_mcmc

# Statistical tools
from src.stats.intervals import calc_eti_from_sample_matrix
from src.stats.psis import compute_psis_importance_diagnostics

# Pipeline orchestration
from src.pipeline.stage1 import run_stage1
from src.pipeline.stage2 import run_stage2
from src.pipeline.selection import select_and_download
```

## Package Structure

```text
src/
  __main__.py               # python -m src -> CLI
  cli/                      # argparse dispatch layer
  config/                   # settings and constants
  data/                     # data loading, download, and result I/O
  models/                   # scientific model implementations
  pipeline/                 # workflow orchestration
  stats/                    # model-independent statistical utilities
  viz/                      # visualization helpers and paper figures
src-orig/                   # historical scripts kept for compatibility
docs/                       # refactor notes and user stories
```

## Legacy Scripts

The original monolithic scripts remain in `src-orig/` for backward
compatibility:

```bash
python src-orig/main.py
python src-orig/m200.py
```

The current `src/` package does not rely on `src-orig/` at runtime. Treat
`src-orig/` as historical compatibility code, not as the place for new
features.
