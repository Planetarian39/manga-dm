# src-orig Runtime Delegation Inventory

> Generated: 2026-05-30  
> Command: `rg -n "src-orig|import main|import m200|import plates|import figure|import dm" src`

This inventory separates runtime legacy delegations from historical comments.
Runtime entries are the ones to remove in Phase 2 and Phase 4 of the refactor plan.

## Pipeline Runtime Delegations

None remaining. Current scan only finds historical provenance comments in
`src/pipeline/stage1.py`, `src/pipeline/stage2.py`, and
`src/pipeline/selection.py`.

## CLI Runtime Delegations

None remaining. `manga figures` now dispatches through `src.viz.paper`,
`src.viz.rc_curves`, and `src.viz.velocity_maps`.

## Viz Runtime Delegations

None remaining. Current scan only finds historical provenance comments in
`src/viz/paper.py`, `src/viz/rc_curves.py`, and `src/viz/velocity_maps.py`.

## Historical Comments Only

These references document provenance and do not create runtime dependencies:

- `src/data/catalog.py`
- `src/data/fits.py`
- `src/data/firefly.py`
- `src/data/maps.py`
- `src/data/results.py`
- `src/models/dm_nfw.py`
- `src/models/population.py`
- `src/models/rotation_curve.py`
- `src/stats/arviz_compat.py`
- `src/stats/gmm.py`
- `src/stats/intervals.py`
- `src/stats/psis.py`
- `src/viz/figure_panels.py`
- `src/viz/posterior.py`
- `src/viz/utils.py`
