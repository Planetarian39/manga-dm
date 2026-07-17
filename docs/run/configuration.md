# Configuration

Configuration is centralized in `src.config.settings`. Business logic should receive resolved settings rather than opening `config.toml` directly.

## Lookup and override order

1. An explicit `--config` path. A missing explicit path is an error.
2. `config.toml` in the current directory.
3. `config.toml` at the detected project root.
4. Built-in fallback values when no file exists.

Global `--data-dir` and `--result-dir` override configured directories. Relative input and result paths resolve against the project root.

## Important sections

| Section | Purpose | Examples |
|---|---|---|
| `[file]` | Data/result roots and output filenames | `data_directory`, `result_directory` |
| `[thresholds]` | Spaxel and fit screening | SNR, azimuth, inclination, HDI coverage |
| `[rc]` | Rotation-curve preparation | radius floor, intrinsic axis ratio, velocity error |
| `[plateifus]` | Plate-IFU and inclination selection | list paths and fallback bounds |

## Manuscript-aligned values are not current fallbacks

| Decision | Manuscript-aligned method | Current fallback or preset |
|---|---:|---:|
| Major-axis azimuth | at most 60° | 45° fallback |
| Predictive HDI | 0.9545 (rounded to 95% in prose) | 0.95 fallback |
| Reduced chi-squared | at most 2.0 | 3.0 in `recommended` |
| Posterior-predictive p-value | 0.1 to 0.9 | 0.05 to 0.95 in `recommended` |
| Absolute posterior correlation | at most 0.85 | 0.95 in `recommended` |

<MethodStatus status="paper">

Manuscript reproduction requires one pinned profile containing the manuscript
screening values, quality equation, likelihood mode, and sampler settings.

</MethodStatus>

<MethodStatus status="implementation">

Neither `recommended` nor `strict` is a manuscript-aligned preset. Until a versioned manuscript profile exists, record the full configuration used for every run and keep the method/implementation distinction visible.

</MethodStatus>

## Path helpers

Use `settings.resolve_input_path()` for input files and `settings.resolve_result_dir()` for result roots. These helpers keep CLI overrides and project-relative paths consistent across data, pipeline, model, and visualization modules.
