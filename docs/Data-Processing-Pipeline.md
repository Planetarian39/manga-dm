---
title: Data Processing Pipeline
---

# Data Processing Pipeline

This page describes the current `src/` pipeline behind the `manga` CLI. It is the operational reference for running the MaNGA dark-matter workflow from target selection through population-level inference.

## Workflow Summary

1. Select MaNGA plate-IFU targets and optionally download DR17 input products.
2. Run Stage 1 per galaxy: fit the gas rotation curve, apply quality gates, and optionally infer an NFW halo.
3. Merge per-galaxy posterior samples into a single NetCDF file.
4. Run Stage 2 population inference for the halo mass-concentration relation.
5. Generate figures or robustness subsamples from the stored products.

```bash
manga select --download
manga stage1 --ifu test --nfw
manga merge --ifu-file data/plateifus.txt
manga stage2 --fit --quality-cut recommended
manga stage2 --diagnose --quality-cut recommended
```

`python -m src` provides the same command interface for local development.

## Configuration and Paths

Runtime configuration is resolved through `src.config.settings`. Business logic should use the settings helpers rather than reading `config.toml` directly.

| Purpose | Default |
|---|---|
| Data root | `data/` |
| Result directory | `data/results/` |
| Selected IFU list | `data/plateifus.txt` |
| Rotation-curve parameters | `rc_param.csv` |
| NFW parameters | `nfw_param_cm200.csv` |
| Merged NFW posterior samples | `nfw_param_cm200_samples.nc` |

Global CLI options `--config`, `--data-dir`, and `--result-dir` flow through the same settings layer.

## Input Data

The pipeline uses MaNGA DR17 products:

| Product | Used for |
|---|---|
| DRPALL | catalog metadata, target bits, quality bits, redshift, stellar mass, axis ratio, effective radius, Sersic index |
| DAP MAPS | H-alpha velocity, inverse variance, flux, gas dispersion, SNR, sky coordinates, radial maps |
| Firefly MASTAR | stellar population data used by single-galaxy modeling |
| SDSS image PNGs | optional visual products downloaded with selected galaxies |

Downloaded files are stored under `data/redux/`, `data/analysis/`, `data/firefly/`, and `data/images/`.

## 1. Sample Selection

```bash
manga select --download
```

Selection is implemented by `src.pipeline.selection.select_and_download` and `src.data.catalog.DrpallUtil`.

The current target-selection logic:

- starts from DRPALL;
- keeps rows with nonzero MaNGA target bits;
- excludes configured `MNGTARG3` bits 19, 20, 21, and 27;
- excludes DRP quality failures using `DRP3QUAL` bits 14 and 30;
- converts axis ratio `b/a` to inclination using intrinsic thickness `q0 = 0.2`;
- keeps galaxies in the configured inclination range, default `25 <= i <= 70` degrees;
- deduplicates by `MANGAID`;
- writes the selected plate-IFU list to `data/plateifus.txt` unless `--ifu-file` is supplied.

With `--download`, the command downloads MAPS files and image PNGs for selected galaxies. DRPALL and Firefly files are downloaded lazily when needed.

## 2. Stage 1: Rotation-Curve and NFW Fits

```bash
manga stage1 --ifu test --nfw
manga stage1 --ifu all --nfw --n-cores 8
manga stage1 --ifu 8994-12701 --nfw
```

Stage 1 is orchestrated by `src.pipeline.stage1.run_stage1`.

### Velocity Map Preparation

For each plate-IFU, the code loads DRPALL, Firefly, and MAPS. The gas velocity field comes from the H-alpha channel (`Ha-6564`) in the DAP MAPS file.

Velocity spaxels are filtered by `RotCurve._build_vel_quality_mask` using:

- finite velocity values;
- SNR at least `SNR_THRESHOLD`, default `10.0`;
- azimuth within `PHI_DEG_THRESHOLD`, default 45 degrees, of the major-axis direction used by the code;
- positive finite velocity inverse variance;
- optional gas-sigma filtering when `GSIGMA_MAX > 0`;
- removal of the lowest configured IVAR percentile, default bottom 10%.

DAP radii are converted from MaNGA `h^-1 kpc` to physical kpc using the configured Hubble ratio.

### Rotation-Curve Fit

The default rotation-curve fit uses PyMC MCMC. The intrinsic profile is:

```text
V_rot(r) = Vc * tanh(r / Rt) + s_out * r
```

The projected velocity model is:

```text
V_obs = Vsys + V_rot(r) * sin(i) * cos(phi - phi_delta)
```

The default MCMC path infers `Vc`, `Rt`, `s_out`, `Vsys`, intrinsic scatter, Student-t degrees of freedom, inclination, and `phi_delta`; the latter two use priors centered on the photometric geometry. It records posterior summaries, RMSE, NRMSE, reduced chi-square, posterior-predictive metrics, and a clipped velocity map in `rc_param.csv`. `RotCurve` retains an internal `lmfit` alternative with fixed inclination and `phi_delta`, but the current CLI does not select it.

### Rotation-Curve Quality Gate

`RotCurve.evaluate_fit_quality` decides whether a galaxy can proceed to NFW inference. The default checks are:

- at least `VEL_OBS_COUNT_THRESHOLD` valid velocity points, default `150`;
- fitted or photometric inclination within the configured range, default 25 to 70 degrees;
- radial extent condition `Rmax / Rt >= RMAX_RT_FACTOR`, default `2`;
- posterior-predictive coverage and overlap checks when those metrics are present.

Failed galaxies are still recorded in `rc_param.csv`, but NFW inference stops for that object.

### NFW Halo Inference

When `--nfw` is set, `src.models.dm_nfw.DmNfw` runs a PyMC/NUTS model for each galaxy that passes the rotation-curve gate.

The model combines:

- a stellar component with a Hernquist bulge and Freeman exponential disk;
- an NFW dark-matter halo;
- a simple asymmetric-drift correction;
- a Student-t likelihood for robust velocity residuals;
- smooth radial down-weighting for the inner region.

The circular-velocity decomposition is:

```text
V_rot^2(r) = V_dm^2(r) + V_star^2(r) - V_drift^2(r)
```

For each successful NFW run, the pipeline writes:

- per-galaxy parameter rows to `nfw_param_cm200.csv`;
- per-galaxy posterior samples to `<plateifu>_nfw_param_cm200_samples.nc`;
- optional diagnostic plots when debug or plotting output is enabled.

The saved sample variables are `log10_M200_samples` and `log10_c_samples`.

## 3. Merge Posterior Samples

```bash
manga merge --ifu-file data/plateifus.txt
```

`src.pipeline.stage2.merge_samples` merges per-IFU sample files matching:

```text
*_nfw_param_cm200_samples.nc
```

The merged output is:

```text
data/results/nfw_param_cm200_samples.nc
```

The merged NetCDF is keyed by `plate_ifu` and contains padded sample arrays plus per-galaxy sample counts.

## 4. Stage 2: Population-Level Inference

```bash
manga stage2 --fit --quality-cut recommended
manga stage2 --diagnose --quality-cut recommended
```

Stage 2 is orchestrated by `src.pipeline.stage2.run_stage2` and `src.pipeline.population.fit_m200_c_population`.

The Stage 2 input table comes from `nfw_param_cm200.csv`, optionally enriched with posterior samples from the merged NetCDF file. The population model fits:

```text
log10(c200) = log10(c0) + alpha * log10(M200 / M_pivot)
```

Quality filters are applied before population fitting:

| Preset | Criteria |
|---|---|
| `recommended` | `redchi <= 3.0`, `0.05 <= dev_ppc_p <= 0.95`, `PPC overlap >= 0.5`, `abs(c_M200_corr) <= 0.95` |
| `strict` | `redchi <= 2.0`, `0.10 <= dev_ppc_p <= 0.90`, `PPC value coverage >= 0.80`, `PPC overlap >= 0.60`, `abs(c_M200_corr) <= 0.90` |

`--diagnose` runs PSIS diagnostics using the saved population fit and merged per-galaxy posterior samples.

## 5. Figures and Robustness Samples

```bash
manga figures --ifu 8994-12701 7977-3704
manga sample --n 60
```

`manga figures` builds velocity-field, rotation-curve, and `M200-c` summary figures from stored result files.

`manga sample` draws robustness subsamples from successful Stage 1/NFW results. Available internal modes include random, stellar-mass matched, Sersic-index matched, and PSIS-k-hat ranked sampling.

## Current Limitations

- Stage 1 skips galaxies whose MAPS files are not local when running without the selection/download step.
- The rotation-curve quality gate is intentionally conservative; failed rows remain in the CSV so screening decisions are auditable.
- The NFW model uses a pure NFW halo plus baryonic components and does not model adiabatic contraction explicitly.
- Inner velocity points are down-weighted rather than beam-smearing corrected.
- Stage 2 quality depends on Stage 1 posterior diagnostics and saved sample files; `--diagnose` requires the merged posterior NetCDF.
