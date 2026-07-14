# Project overview

`manga-dm` is a Python research pipeline for building and screening MaNGA gas rotation curves, fitting single-galaxy stellar-plus-NFW models, preserving posterior samples, and preparing population-level inference.

## What the pipeline connects

| Layer | Responsibility | Primary package |
|---|---|---|
| Observational inputs | Catalogs, MAPS products, downloads, and result I/O | `src/data` |
| Scientific models | Empirical rotation curves, NFW dynamics, and the population model | `src/models` |
| Workflow | Selection, Stage 1, merge, Stage 2, and diagnostics | `src/pipeline` |
| Statistics | Intervals, GMM utilities, PSIS, and ArviZ compatibility | `src/stats` |
| Figures | Case, diagnostic, and paper-oriented plotting | `src/viz` |
| Interface | Argument parsing and dispatch through `manga` | `src/cli` |

The [architecture page](/project/architecture) expands this map. The [method pages](/methods/) describe the finalized-paper analysis; [implementation status](/project/implementation-status) identifies places where current defaults differ.

## Public interface

After installation, use the `manga` command or the equivalent module entry point:

```powershell
manga --help
python -m src --help
```

The CLI exposes `select`, `stage1`, `merge`, `stage2`, `figures`, and `sample`. Begin with the [end-to-end workflow](/run/cli-workflow).

## What this site does not claim

The documentation explains a research method and its implementation. It does not release the paper, aggregate sample products, population findings, or a claim that the current default CLI reproduces every finalized-paper choice. See [limitations](/project/limitations) before interpreting a fit.
