# manga-dm

## How to install the Anaconda development environment (Windows)

1.  Create manga-dm environment:

    ```bash
    conda create -c conda-forge -n manga-dm python=3.12
    conda activate manga-dm
    ```

2. Install the Astropy packages:

   Use conda to install packages.
    ```bash
    conda install -c conda-forge nomkl numpy scipy lmfit
    conda install -c conda-forge pytensor pymc arviz nutpie dm-tree
    conda install -c conda-forge jax jaxlib
    conda install -c conda-forge \
                    pandas pytz h5py pyarrow fsspec s3fs bottleneck \
                    certifi tqdm mpmath jplephem \
                    beautifulsoup4 html5lib bleach
    conda install -c conda-forge ipython jupyter dask
    conda install -c conda-forge astropy astroquery reproject asdf-astropy
    conda install -c conda-forge pvextractor
    conda install -c conda-forge matplotlib seaborn xarray-einstats numba
    conda install -c conda-forge m2w64-toolchain libpython
    ```

    Some packages need used pip to install.
    ```
    pip install ppxf spectral_cube sdss-access
    ```

## Usage

### Install the package (editable mode)

```bash
pip install -e .
```

### Unified CLI `manga`

The project provides a single `manga` command with subcommands:

```bash
manga --help
manga <subcommand> --help
```

| Subcommand | Description | Equivalent old script |
|---|---|---|
| `manga select` | Select galaxy sample, download data | `python src/plates.py` |
| `manga stage1` | Single-galaxy RC + DM NFW fitting | `python src/main.py` |
| `manga stage2` | Population model inference (c–M relation) | `python src/m200.py` |
| `manga figures` | Generate paper figures | `python src/figure.py` |
| `manga merge` | Merge posterior sample files | (extracted from `m200.py`) |
| `manga sample` | Generate robustness sub-samples | (extracted from `m200.py`) |

### Typical workflow

```bash
# 1. Select galaxies by inclination and download data
manga select --download

# 2. Stage 1: fit rotation curve + NFW profile (8 test galaxies)
manga stage1 --ifu test --nfw

# 3. Stage 1: full sample (all galaxies in plateifus.txt)
manga stage1 --ifu all --nfw --n-cores 8

# 4. Merge posterior samples after Stage 1 completes
manga merge --ifu-file data/plateifus.txt

# 5. Stage 2: population-level c–M200 relation
manga stage2 --fit --quality-cut recommended

# 6. Generate figures for specific galaxies
manga figures --ifu 8994-12701 7977-3704
```

### Python API

```python
# Configuration (singleton)
from src.config import settings
print(settings.data_dir)     # Path to data
print(settings.result_dir)   # Path to results

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

### Package structure

```
src/
├── __init__.py              # Top-level package
├── __main__.py               # python -m src → CLI
├── config/                   # Configuration & constants
│   ├── settings.py           # Singleton Settings (from config.toml)
│   └── constants.py          # Physical/cosmological constants, colours
├── data/                     # Data access layer
│   ├── catalog.py            # DRPALL catalogue + plate-IFU tools
│   ├── fits.py               # FITS download & I/O
│   ├── maps.py               # MaNGA MAPS reader
│   ├── firefly.py            # Firefly stellar-mass data
│   └── results.py            # CSV/NetCDF result I/O
├── models/                   # Scientific models (PyMC)
│   ├── rotation_curve.py     # RotCurve class
│   ├── dm_nfw.py             # DmNfw class
│   └── population.py         # fit_m200_c_mcmc
├── stats/                    # Statistical tools
│   ├── arviz_compat.py       # ArviZ version compatibility
│   ├── intervals.py          # ETI/HDI interval computation
│   ├── gmm.py                # Gaussian mixture model fitting
│   └── psis.py               # PSIS importance sampling diagnostics
├── pipeline/                 # Workflow orchestration
│   ├── stage1.py             # Stage 1 RC + DM pipeline
│   ├── stage2.py             # Stage 2 population inference
│   └── selection.py          # Sample selection & filtering
├── viz/                      # Visualisation
│   ├── utils.py              # Shared plot helpers
│   ├── posterior.py          # Posterior distribution plots
│   ├── rc_curves.py          # Rotation-curve figures
│   ├── velocity_maps.py      # Velocity-field maps
│   └── paper.py              # Multi-panel paper figures
└── cli/                      # CLI layer
    └── main.py               # argparse + dispatcher
```

### Legacy scripts (src-orig/)

The original monolithic scripts remain in `src-orig/` for backward compatibility:

```bash
pip install -e .              # install manga CLI
python src-orig/main.py       # old Stage 1 entry point (still works)
python src-orig/m200.py       # old Stage 2 entry point (still works)
```
