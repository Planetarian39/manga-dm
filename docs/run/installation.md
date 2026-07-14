# Installation

The documented development environment is Windows with Anaconda and Python 3.11 or newer. A conda-forge environment is recommended because the scientific stack includes compiled numerical and astronomy packages.

## Create the environment

```powershell
conda create -c conda-forge -n manga-dm python=3.12
conda activate manga-dm
```

Install the core numerical, probabilistic, and astronomy dependencies with conda, then install the package in editable mode:

```powershell
conda install -c conda-forge numpy scipy pandas xarray matplotlib astropy pymc arviz lmfit
pip install -e .
```

Some development workflows also use `nutpie`, JAX, SDSS access tools, and visualization helpers. Use `pyproject.toml` and the repository README as the dependency authority for the checkout you are running.

## Verify the interface

```powershell
manga --help
manga stage1 --help
manga stage2 --help
```

If the console script is not on `PATH`, use:

```powershell
python -m src --help
```

## Install the documentation toolchain

The website is independent of the Python environment:

```powershell
npm ci
npm run docs:dev
```

The VitePress site is served locally with the same `/manga-dm/` base used by GitHub Pages.

Maintainers who regenerate the public case summaries or MCMC teaching figures
also need `h5py`, NumPy, SciPy, and Matplotlib. These are documentation-tool
dependencies; they do not change the `manga` CLI interface.

## Common setup failures

- **Missing compiled package:** prefer a conda-forge build before compiling locally.
- **`manga` not found:** reactivate the environment or use `python -m src`.
- **Configuration not found:** pass an explicit `--config` path or run from a checkout containing `config.toml`.
- **Input product missing:** confirm `--data-dir`, catalog paths, and the relevant download step.
