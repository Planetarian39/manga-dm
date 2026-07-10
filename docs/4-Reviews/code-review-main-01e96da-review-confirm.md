# main@01e96da Code Review Confirm

**Reviewed Inputs**

- `.review/code-review-main-01e96da.md`
- `docs/superpowers/specs/2026-07-10-review-remediation-design.md`
- `AGENTS.md` and the referenced implementation files

**Review Date**

- 2026-07-10

## Overall Conclusion

The eight findings are supported by current code. Seven require code fixes; the solver discrepancy requires documentation only because the approved scope preserves the current MCMC runtime behavior. The concurrency finding is partially accepted only with respect to its proposed parent-aggregation solution: the data-loss defect is real, but a process-safe critical section plus atomic replacement is the smaller compatible fix.

The reviewed revision should not be accepted until the actions below are implemented and verified.

## Decision Table

| No. | Severity | Type | Review Source | Original Comment Summary | Decision | Evidence | Follow-up Plan / Rejection Reason |
|---|---|---|---|---|---|---|---|
| 1 | High | Correctness | `Report P1-1` | `stage1 --ifu all` reads a working-directory file and can silently fall back to test IFUs. | Accept | `run_stage1()` passes bare `PLATES_FILENAME`; `get_plateifu_list()` returns `TEST_PLATE_IFUS` when the path is absent, while selection writes under `settings.data_dir`. | Resolve `settings.data_dir / PLATES_FILENAME`, remove the implicit test fallback, and add a regression test. |
| 2 | High | Correctness | `Report P1-2` | Parallel Stage 1 workers lose CSV rows through unsynchronized read-modify-write. | Partial | Every worker calls `store_params_file()`, which reads and rewrites the shared CSV without a lock or atomic replacement. The defect is valid, but refactoring the 344-line fitting flow to return all writes to the parent is unnecessary. | Pass one process-safe lock to workers, protect the shared update, write a temporary CSV, atomically replace it, and test concurrent unique rows. |
| 3 | High | Correctness | `Report P1-3` | An RC row suppresses requested NFW work and failed rows are not retryable. | Accept | `run_stage1()` always filters with `settings.rc_param_filename`; `get_processed_plate_ifus()` treats every row as complete regardless of result. | Select the stage-specific CSV, require successful rows, require the NFW sample file for NFW completion, and add retry tests. |
| 4 | High | Correctness | `Report P1-4` | `merge --ifu-file` ignores its IFU allowlist. | Accept | The CLI forwards `ifu_file`, but `merge_samples()` neither reads it nor passes an allowlist to `merge_posterior_samples_file()`. | Validate and load the file, pass a set of IDs, filter merged rows, and test exclusion of an unrelated sample. |
| 5 | High | Correctness | `Report P1-5` | Fresh data roots cannot lazily download Firefly and failed downloads return nonexistent paths. | Accept | `FitsUtil.__init__()` does not create `firefly_dir`; `get_firefly_file()` ignores the Boolean downloader result. | Create the directory, raise `FileNotFoundError` on failure, and add fresh-root tests. |
| 6 | High | Correctness | `Report P1-6` | PSIS uses Normal population density although the fitted sample likelihood is Student-t. | Accept | `compute_psis_importance_diagnostics()` reads but does not use `nu_pop_median`; `fit_m200_c_mcmc()` evaluates two factored Student-t terms. | Use SciPy Student-t log densities with the fitted degrees of freedom and add a sensitivity test. |
| 7 | Medium | Correctness | `Report P2-1` | A valid local MAPS file is rejected without a checksum sidecar even when checksums are disabled. | Accept | `get_maps_file()` enters its existing-file path only when both the FITS file and sidecar exist. | Return an existing MAPS file immediately when `checksum=False` and add a no-sidecar test. |
| 8 | Medium | Documentation | `Report P2-2` | Documentation says the default RC solver is lmfit while runtime defaults to MCMC. | Accept | `RotCurve.fit_model` is `"mcmc"`; `set_fit_model()` has no current `src/` caller, while `docs/Data-Processing-Pipeline.md` names lmfit as the default. | Update the document to describe the existing MCMC default; do not change runtime code. |

## Needs Immediate Action

- Implement rows 1-7 before relying on the affected Stage 1, merge, or PSIS workflows.
- Correct row 8 in the operational documentation during the same change.

## Can Be Deferred

- Parent-process result aggregation can be reconsidered only if the locked atomic CSV path becomes a measured bottleneck.

## Final Status

Not accepted as-is. The minimum remaining work is the seven confirmed code fixes, the solver documentation correction, focused regression coverage, and full unit/CLI verification.
