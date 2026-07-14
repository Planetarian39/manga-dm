---
title: Implementation Status
description: Explicit boundary between the finalized paper method and current public defaults.
---

# Implementation status

The repository contains the main scientific components of the finalized analysis, but its current default CLI configuration is not yet a versioned paper-reproduction profile.

## Alignment matrix

| Area | Finalized paper method | Current implementation | Status |
|---|---|---|---|
| Major-axis spaxel cut | $|\Delta\phi|\le60^\circ$ | `PHI_DEG_THRESHOLD` fallback is `45.0` in `src/config/settings.py` | Not aligned by default |
| Predictive interval | Exact profile `0.9545`; paper prose rounds to 95% | `HDI_PROB2` fallback is `0.95` | Not aligned by default |
| Empirical convergence | $\hat R\le1.05$, ESS $\ge200$ required | `RotCurve.evaluate_fit_quality` omits them from its pass boolean; sampler warnings remain | Enforcement gap |
| NFW retention | $\hat R\le1.05$, bulk ESS $\ge200$, $\chi^2_\nu\le2.0$, $0.1\le p_{\mathrm{PPC}}\le0.9$, $|\rho|\le0.85$ | `recommended` and `strict` use different equations | No paper preset |
| NFW sampling | 500 tune, 1000 draws, up to four chains, target 0.95, `nutpie` | `src/models/dm_nfw.py` uses these values | Aligned |
| Stage 2 likelihood | Prior-corrected posterior samples with per-galaxy truncation | `src/pipeline/population.py` defaults to GMM inputs | Paper path not default |
| Provenance | Profile, likelihood mode, sampler settings, code version | Saved fits do not carry the complete versioned record | Incomplete |

::: warning Paper method and current implementation
No combination of the current fallback $45^\circ$, `0.95`, `recommended`/`strict`, warning-only empirical $\hat R$/ESS handling, and default GMM Stage 2 path should be described as reproducing the paper. The Methods pages state the finalized science; this page states current executable behavior.
:::

## Stable code map

- `src/config/settings.py`: active configuration and fallbacks
- `src/config/constants.py`: physical/prior constants and quality presets
- `src/models/rotation_curve.py`: empirical model, masks, and predictive checks
- `src/models/dm_nfw.py`: single-galaxy model and posterior generation
- `src/data/results.py`: posterior and summary I/O
- `src/pipeline/selection.py`: selection and quality-filter orchestration
- `src/pipeline/population.py`: Stage 2 dispatch and likelihood defaults
- `src/models/population.py`: population likelihood modes
- `src/stats/psis.py`: importance-weight diagnostics

## Confirmed future alignment scope

The approved design reserves a separate scientific-code alignment Sprint. Its bounded scope is:

1. add a versioned profile with the $60^\circ$ cut and `0.9545` predictive probability;
2. add a paper quality preset matching the complete retention equation, including $\hat R$ and ESS;
3. expose the prior-corrected sample likelihood as the paper-aligned Stage 2 CLI path;
4. keep GMM as a clearly named alternative unless numerical review supports another decision;
5. add provenance for profile, likelihood mode, sampler settings, and code version;
6. run numerical regressions against the four approved single-galaxy references and an approved population reference before accepting conclusion-affecting changes.

This documentation work does not create a Sprint file, change `src/`, select tolerances, or execute those changes.

## Future acceptance evidence

- CLI help, configuration, preset, and dispatch tests
- Likelihood-selection tests
- Deterministic data-preparation checks
- Four-case posterior and summary tolerances
- An approved population reference comparison
- Provenance assertions on saved outputs

Until that evidence exists, use “paper method” and “current implementation” as distinct labels.
