# Architecture

The repository uses a thin CLI and responsibility-oriented Python packages. Scientific behavior should remain in the model and pipeline layers, while I/O, statistics, and plotting stay independently testable.

## Package boundaries

```text
manga CLI
  └─ src/cli          parse and dispatch
      └─ src/pipeline orchestrate selection, Stage 1, merge, and Stage 2
          ├─ src/data   external data and result I/O
          ├─ src/models scientific model implementations
          ├─ src/stats  model-independent statistical utilities
          ├─ src/viz    figures and visual diagnostics
          └─ src/config settings, paths, and constants
```

The official entry point is `src.cli.main.main`, exposed through `manga` and `python -m src`.

## Data flow

1. `src.data` loads catalogs and observational products.
2. `src.pipeline.selection` prepares and screens target records.
3. `src.models.rotation_curve.RotCurve` fits the empirical velocity model.
4. `src.models.dm_nfw` fits the single-galaxy dynamical model.
5. `src.data.results` stores and merges posterior samples.
6. `src.models.population` evaluates population likelihoods and sampling.
7. `src.stats` and `src.viz` provide diagnostics and presentation.

## Design invariants

- CLI modules parse arguments and forward calls; they do not own scientific logic.
- Configuration is resolved through `src.config.settings`.
- Output schemas and filenames remain stable unless a migration is planned.
- Scientific model changes require numerical validation, not only unit tests.
- `var/` is runtime-generated and volatile; public documentation assets are curated under `docs/public/`.

Continue to the [method-to-code map](/project/code-map) for symbol-level entry points.

## Documentation route migration

The VitePress rebuild removes stale duplicate routes instead of preserving the
old Jekyll structure. These source paths map to the maintained pages:

| Legacy source path | Current destination | Action |
|---|---|---|
| `docs/index.md` | `/` | Rewritten as the method-first home page |
| `docs/Data-Processing-Pipeline.md` | `/run/cli-workflow.html` | Replaced by the CLI workflow and input/output pages |
| `docs/future/manga-dm-rc-shapes.md` | `/project/future-research.html` | Missing legacy target replaced by a boundary-safe future-research note |
| `docs/mcmc/how-and-why-to-use-mcmc.md` | `/background/mcmc/` | Split into the preserved MCMC learning sequence |
