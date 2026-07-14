# Inputs and outputs

The pipeline separates external MaNGA products, compact result tables, and posterior sample files. Paths are resolved by `src.config.settings`; generated content should remain under the selected result directory.

## Main inputs

| Input | Role | Distribution note |
|---|---|---|
| DRPALL catalog | Global galaxy metadata and photometric anchors | Referenced, not redistributed here |
| DAP MAPS products | Emission-line velocity, inverse variance, SNR, geometry, and dispersion maps | Raw FITS are not published by this site |
| Plate-IFU list | Selects targets for Stage 1 or merge | Plain text, one identifier per workflow entry |
| `config.toml` | Paths and thresholds | Local run configuration |

## Stage 1 outputs

- `rc_param.csv`: empirical rotation-curve parameters and screening metrics.
- `nfw_param_cm200.csv`: compact single-galaxy NFW summaries.
- Per-IFU `*_nfw_param_cm200_samples.nc`: full `log10_M200` and `log10_c` posterior arrays.
- Optional case figures generated through `src.viz`.

The public [case downloads](/case-studies/downloads) expose four complete per-galaxy NetCDF examples. Their minimal schema is:

| Variable | Shape | Meaning |
|---|---:|---|
| `log10_M200_samples` | `(sample,)` | Posterior draws of halo mass in base-10 log space |
| `log10_c_samples` | `(sample,)` | Posterior draws of concentration in base-10 log space |
| `sample` | `(sample,)` | Sample coordinate |
| `sample_count` | scalar | Number of stored draws |

Each example contains 4,000 aligned draws. The archive attribute `plate_ifu` identifies the object.

## Merge and Stage 2 outputs

The merge step consolidates selected per-IFU samples for population inference. Stage 2 writes fit state, scalar summaries, and diagnostic figures under the result directory. Those aggregate artifacts are intentionally not distributed on this site.

## Provenance expectations

For reproducible work, retain the configuration profile, likelihood mode, sampler settings, code revision, input catalog version, and result path alongside each fit. Current saved outputs do not yet encode all of this provenance automatically; see [implementation status](/project/implementation-status).
