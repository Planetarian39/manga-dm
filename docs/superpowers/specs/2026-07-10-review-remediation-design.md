# Review Remediation Design

## Goal

Resolve the eight findings in `.review/code-review-main-01e96da.md` without changing scientific model behavior or output formats.

## Scope

- Fix the seven confirmed code defects in catalog selection, Stage 1 result handling, Stage 2 merging, FITS downloads, and PSIS diagnostics.
- Resolve the rotation-curve solver discrepancy by documenting the existing MCMC default. Do not change the runtime default.
- Add focused regression coverage and preserve the existing `manga` CLI surface.

## Design

### Input and download boundaries

`manga stage1 --ifu all` will read `settings.data_dir / PLATES_FILENAME`. The catalog helper will return test galaxies only when `test=True`; a missing ordinary list will no longer silently select test data. Firefly initialization will create its directory, and a failed lazy download will raise `FileNotFoundError`. MAPS files will be accepted without a checksum sidecar when checksum verification is disabled.

### Stage 1 completion and persistence

RC-only runs will skip only successful RC rows. NFW runs will skip only IFUs with a successful NFW row and the expected per-IFU posterior sample file. Failed or incomplete rows remain retryable.

Parallel workers will retain the current computation flow. The shared CSV read-modify-write section will receive one process-safe lock from `run_stage1`, and CSV replacement will use a temporary file followed by `os.replace()`. This fixes the documented multiprocessing path without restructuring the scientific fitting function or changing CSV fields.

### Stage 2 allowlist

`merge_samples()` will require an existing IFU list, load its identifiers, and pass an allowlist into `merge_posterior_samples_file()`. The data helper will ignore per-IFU sample files outside that allowlist while preserving its existing behavior when called without one.

### PSIS distribution consistency

PSIS population log density will use SciPy's Student-t log density for shifted halo mass and conditional concentration with `nu_pop_median`, matching the factored Student-t parameterization in `fit_m200_c_mcmc()`. The PyMC population model itself will not change.

### Documentation

`docs/Data-Processing-Pipeline.md` will state that the current default rotation-curve solver is MCMC with its existing inclination and position-angle priors. No solver option or runtime behavior will be added.

## Error Handling

- Missing all-sample input produces the existing clear Stage 1 message and schedules no work.
- Missing merge allowlist raises a clear `FileNotFoundError` rather than merging unrelated samples.
- Failed Firefly download raises `FileNotFoundError` rather than returning a nonexistent path.
- Existing failure records remain visible in CSV output but do not block retries.

## Verification

Focused tests will cover each corrected behavior, including concurrent unique CSV updates, NFW retry selection, merge filtering, download failure, MAPS-without-sidecar handling, and sensitivity of PSIS results to `nu_pop_median`. Final verification will run the complete unit suite plus `manga --help` and relevant subcommand help.

## Non-goals

- No changes to PyMC model internals, priors, output schemas, or scientific formulas beyond making PSIS evaluate the already-fitted Student-t distribution.
- No new dependency or result-storage abstraction.
- No parent-process rewrite of the full Stage 1 orchestration.
