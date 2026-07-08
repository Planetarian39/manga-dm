# Data Processing Pipeline for MaNGA Rotation-Curve Fitting and NFW Inference

## 1. Overview

This document describes the current `src/` pipeline behind the `manga` CLI. The workflow is split into four practical stages:

1. Select MaNGA plate-IFU targets and optionally download DR17 input files.
2. Run Stage 1 per galaxy: fit the gas rotation curve, apply quality gates, and optionally infer an NFW halo.
3. Merge per-galaxy posterior samples into a single NetCDF file.
4. Run Stage 2 population inference for the halo mass-concentration relation.

The official entry point is:

```bash
manga <subcommand> [options]
```

The equivalent module entry point is:

```bash
python -m src <subcommand> [options]
```

## 2. Configuration and Paths

Runtime configuration is resolved through `src.config.settings`, not by reading `config.toml` directly from business logic. If a `config.toml` file is present, it can override the defaults; otherwise the code uses built-in defaults.

Important default paths and filenames:

| Purpose | Default |
|---|---|
| Data root | `data/` |
| Result directory | `data/results/` |
| Selected IFU list | `data/plateifus.txt` |
| Rotation-curve parameters | `rc_param.csv` |
| NFW parameters | `nfw_param_cm200.csv` |
| Merged NFW posterior samples | `nfw_param_cm200_samples.nc` |

CLI-level `--data-dir`, `--result-dir`, and `--config` options are resolved through the same settings layer.

## 3. Input Data

The pipeline currently uses MaNGA DR17 products:

- **DRPALL**: catalog metadata, target bits, DRP quality bits, redshift, stellar mass, axis ratio, effective radius, and Sersic index.
- **DAP MAPS**: H-alpha emission-line velocity, velocity inverse variance, emission-line flux, gas velocity dispersion, SNR, sky coordinates, and radial maps.
- **Firefly MASTAR**: stellar population data used by the single-galaxy modeling code.
- **SDSS image PNGs**: optional visual products downloaded with selected galaxies.

The data downloader stores files under `data/redux/`, `data/analysis/`, `data/firefly/`, and `data/images/`.

## 4. Sample Selection

Run:

```bash
manga select --download
```

Selection is implemented in `src.pipeline.selection.select_and_download` and `src.data.catalog.DrpallUtil`.

The current target-selection logic:

- Starts from DRPALL.
- Keeps rows with nonzero MaNGA target bits.
- Excludes configured `MNGTARG3` bits 19, 20, 21, and 27.
- Excludes DRP quality failures using `DRP3QUAL` bits 14 and 30.
- Converts axis ratio `b/a` to inclination using an intrinsic thickness of `q0 = 0.2`.
- Keeps galaxies with inclination in the configured range, default `25 <= i <= 70` degrees.
- Deduplicates by `MANGAID`.
- Writes the selected plate-IFU list to `data/plateifus.txt` unless `--ifu-file` is supplied.

With `--download`, the code downloads MAPS files and image PNGs for the selected galaxies. DRPALL and Firefly files are downloaded lazily when needed.

## 5. Stage 1: Per-Galaxy Rotation Curve and NFW Fit

Run one of:

```bash
manga stage1 --ifu test --nfw
manga stage1 --ifu all --nfw --n-cores 8
manga stage1 --ifu 8994-12701 --nfw
```

Stage 1 is orchestrated by `src.pipeline.stage1.run_stage1`.

### 5.1 Velocity Map Preparation

For each plate-IFU, the code loads DRPALL, Firefly, and MAPS. The gas velocity field comes from the H-alpha channel (`Ha-6564`) in the DAP MAPS file.

Velocity spaxels are filtered by `RotCurve._build_vel_quality_mask`:

- finite velocity values;
- SNR at least `SNR_THRESHOLD`, default `10.0`;
- azimuth within `PHI_DEG_THRESHOLD`, default 45 degrees, of the major-axis direction used by the code;
- positive finite velocity inverse variance;
- optional gas-sigma filtering if `GSIGMA_MAX > 0`;
- removal of the lowest configured IVAR percentile, default bottom 10%.

DAP radii are converted from MaNGA `h^-1 kpc` to physical kpc using the configured Hubble ratio.

### 5.2 Rotation-Curve Fit

The default rotation-curve fit uses `lmfit` least squares. The intrinsic profile is:

```text
V_rot(r) = Vc * tanh(r / Rt) + s_out * r
```

The projected velocity model is:

```text
V_obs = Vsys + V_rot(r) * sin(i) * cos(phi - phi_delta)
```

The fit uses the filtered velocity map, measurement uncertainty from IVAR, and a configured velocity error floor. In the current `lmfit` path, inclination is fixed to the photometric inclination and `phi_delta` is fixed at zero; `Vc`, `Rt`, `s_out`, and `Vsys` are fit.

After fitting, the code records RMSE, NRMSE, reduced chi-square, fitted parameters, and a clipped velocity map. Results are stored in `rc_param.csv`.

### 5.3 Rotation-Curve Quality Gate

`RotCurve.evaluate_fit_quality` decides whether a galaxy can proceed to the NFW fit. The default checks are:

- at least `VEL_OBS_COUNT_THRESHOLD` valid velocity points, default `150`;
- fitted/photometric inclination within the configured range, default 25 to 70 degrees;
- radial extent condition `Rmax / Rt >= RMAX_RT_FACTOR`, default `2`;
- posterior-predictive coverage and overlap checks when those metrics are present.

Failed galaxies are still recorded in `rc_param.csv`, but NFW inference stops for that object.

## 6. Stage 1 NFW Halo Inference

When `--nfw` is set, `src.models.dm_nfw.DmNfw` runs a PyMC/NUTS model for each galaxy that passes the rotation-curve gate.

The model combines:

- a stellar component with a Hernquist bulge and Freeman exponential disk;
- an NFW dark-matter halo;
- a simple asymmetric-drift correction;
- a Student-t likelihood for robust velocity residuals;
- a smooth radial down-weighting potential for the inner region.

The circular-velocity decomposition is:

```text
V_rot^2(r) = V_dm^2(r) + V_star^2(r) - V_drift^2(r)
```

Main priors in the current code:

- `Mstar`: log10-normal prior centered on the DRPALL/stellar mass estimate, default width 0.05 dex.
- `M200`: truncated log10-normal prior centered on the Moster-style SHMR estimate from `Mstar`, default width `M200_DEX = 0.15` dex.
- `c`: independent log-normal prior centered near `c = 9`, default width 0.5 dex.
- `sigma_0`: log-normal prior centered near 10 km/s.
- `v_sys`: truncated normal prior centered on the Stage 1 velocity-system value.
- `inc`: fixed by default to the photometric/Stage 1 inclination; it can be made stochastic through the model setter used by programmatic callers.
- `Re`: log-normal prior centered on the DRPALL effective radius.
- `a`: deterministic Hernquist scale, `a = Re / 1.8153`.
- `f_bulge`: logistic-normal prior centered from the galaxy Sersic index.
- `sigma_int`: exponential intrinsic-scatter prior.
- `nu`: Student-t degrees of freedom with `nu = Gamma(2, 0.1) + 2`.

For each successful NFW run, the pipeline writes:

- per-galaxy parameter rows to `nfw_param_cm200.csv`;
- per-galaxy posterior samples to `<plateifu>_nfw_param_cm200_samples.nc`;
- optional diagnostic plots when debug/plotting is enabled.

The saved sample variables are `log10_M200_samples` and `log10_c_samples`.

## 7. Merge Posterior Samples

Run:

```bash
manga merge --ifu-file data/plateifus.txt
```

`src.pipeline.stage2.merge_samples` merges all per-IFU sample files matching:

```text
*_nfw_param_cm200_samples.nc
```

The merged file is:

```text
data/results/nfw_param_cm200_samples.nc
```

The merged NetCDF is keyed by `plate_ifu` and contains padded sample arrays plus per-galaxy sample counts.

## 8. Stage 2: Population-Level M200-c Inference

Run:

```bash
manga stage2 --fit --quality-cut recommended
manga stage2 --diagnose --quality-cut recommended
```

Stage 2 is orchestrated by `src.pipeline.stage2.run_stage2` and `src.pipeline.population.fit_m200_c_population`.

The Stage 2 input table comes from `nfw_param_cm200.csv`, optionally enriched with posterior samples from the merged NetCDF file. The default fit uses the per-galaxy `M200` and `c` estimates, plus saved log-space Gaussian-mixture summaries when present.

Quality filters are applied before population fitting. The preset thresholds are:

| Preset | Criteria |
|---|---|
| `recommended` | `redchi <= 3.0`, `0.05 <= dev_ppc_p <= 0.95`, `PPC overlap >= 0.5`, `abs(c_M200_corr) <= 0.95` |
| `strict` | `redchi <= 2.0`, `0.10 <= dev_ppc_p <= 0.90`, `PPC value coverage >= 0.80`, `PPC overlap >= 0.60`, `abs(c_M200_corr) <= 0.90` |

The population model fits a relation of the form:

```text
log10(c200) = log10(c0) + alpha * log10(M200 / M_pivot)
```

and saves the population fit results for later diagnostics and plotting. `--diagnose` runs PSIS diagnostics using the saved population fit and the merged per-galaxy posterior samples.

## 9. Robustness Samples and Figures

Two auxiliary CLI paths use the same result products:

```bash
manga sample --n 60
manga figures --ifu 8994-12701 7977-3704
```

`manga sample` draws robustness subsamples from successful Stage 1/NFW results. Available internal modes include random, stellar-mass matched, Sersic-index matched, and PSIS-k-hat ranked sampling.

`manga figures` builds velocity-field, rotation-curve, and `M200-c` summary figures from the stored result files.

## 10. Current Limitations

- Stage 1 skips galaxies whose MAPS files are not already local when running without the selection/download step.
- The rotation-curve quality gate is intentionally conservative; failed rows remain in the CSV so screening decisions are auditable.
- The NFW model uses a pure NFW halo plus baryonic components and does not model adiabatic contraction explicitly.
- Inner velocity points are down-weighted rather than beam-smearing corrected.
- Stage 2 quality depends on Stage 1 posterior diagnostics and saved sample files; `--diagnose` requires the merged posterior NetCDF.
