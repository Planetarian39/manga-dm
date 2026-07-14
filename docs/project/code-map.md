# Method-to-code map

This map links finalized-paper concepts to the current implementation. A link means “implementation entry point,” not “numerically proven identical to the paper.” Known differences are collected in [implementation status](/project/implementation-status).

| Method concept | Current implementation entry | Responsibility |
|---|---|---|
| CLI dispatch | `src/cli/main.py` | Global options and six public subcommands |
| Settings and paths | `src/config/settings.py` | Config lookup, thresholds, and path resolution |
| Catalog selection | `src/data/catalog.py` | DRPALL loading and catalog filters |
| Velocity-map screening | `RotCurve._build_vel_quality_mask` in `src/models/rotation_curve.py` | IVAR, SNR, azimuth, and optional dispersion filtering |
| Empirical rotation curve | `RotCurve._fit_vel_rot` | Tanh-plus-linear curve and robust fit |
| Empirical quality | `RotCurve.evaluate_fit_quality` | Coverage and predictive checks |
| Single-galaxy NFW inference | `src/models/dm_nfw.py` | Stellar, halo, pressure support, priors, and PyMC sampling |
| Posterior persistence | `store_posterior_samples_file` in `src/data/results.py` | Per-IFU NetCDF output |
| Posterior merge | `merge_posterior_samples_file` in `src/data/results.py` | Common Stage 2 sample product |
| Population input preparation | `_prepare_sample_posterior_tensors` in `src/models/population.py` | Sample-based tensors and masks |
| GMM alternative | `_prepare_gmm_tensors` and `src/stats/gmm.py` | Approximate posterior representation |
| Stage 2 orchestration | `src/pipeline/stage2.py` and `src/pipeline/population.py` | Fit and diagnostic dispatch |
| Importance diagnostics | `compute_psis_importance_diagnostics` in `src/stats/psis.py` | Weight stability and Pareto-k diagnostics |
| Case figures | `src/viz/figure_panels.py` and `src/viz/posterior.py` | Velocity, component, and posterior plots |

## Reading a status callout

- **Aligned:** the documented setting or behavior is visible in current code.
- **Implementation alternative:** code provides a useful path that is not the finalized-paper path.
- **Alignment required:** the paper and current fallback/preset differ or enforcement is incomplete.

No scientific code was changed to create this documentation.
