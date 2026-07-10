# Review Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct the eight confirmed review findings while preserving scientific model behavior, CLI compatibility, and result formats.

**Architecture:** Keep fixes at existing boundaries: catalog/FITS input handling in `src/data`, orchestration choices in `src/pipeline`, and the PSIS density in `src/stats`. Reuse the current CSV and NetCDF formats; add only a process-safe lock and atomic replacement around the shared CSV update.

**Tech Stack:** Python 3.11+, stdlib `multiprocessing`/`os`/`unittest`, pandas, SciPy, xarray.

## Global Constraints

- Do not change PyMC model internals, priors, or runtime rotation-curve solver defaults.
- Do not change CSV/NetCDF field names or per-IFU filename conventions.
- Add no dependency.
- Preserve the `manga` CLI surface and the user's existing `.gitignore` edit.

---

### Task 1: Confirm every review finding

**Files:**
- Create: `docs/4-Reviews/code-review-main-01e96da-review-confirm.md`

**Interfaces:**
- Consumes: `.review/code-review-main-01e96da.md`, the approved design, and current repository evidence.
- Produces: one decision row for each of the eight findings.

- [ ] **Step 1: Write the confirmation table**

Use `Accept` for findings 1, 3, 4, 5, 6, 7, and 8. Use `Partial` for finding 2 because the concurrency defect is accepted but the approved fix is a shared lock plus atomic replacement instead of returning all scientific results to the parent process. Include a concrete follow-up for every row.

- [ ] **Step 2: Verify confirmation coverage**

Run:

```powershell
Select-String -Path docs/4-Reviews/code-review-main-01e96da-review-confirm.md -Pattern '^\| [1-8] '
```

Expected: exactly eight table rows.

### Task 2: Correct catalog and FITS input boundaries

**Files:**
- Modify: `src/data/catalog.py:265`
- Modify: `src/data/fits.py:27`
- Create: `tests/test_data_io.py`

**Interfaces:**
- Consumes: `get_plateifu_list(filepath, test=False)` and `FitsUtil(data_dir)`.
- Produces: ordinary missing lists return `[]`; Firefly failures raise `FileNotFoundError`; MAPS files are accepted without sidecars when `checksum=False`.

- [ ] **Step 1: Write failing boundary tests**

Add tests equivalent to:

```python
def test_missing_plateifu_list_does_not_fall_back_to_test_sample(self):
    self.assertEqual(get_plateifu_list("missing.txt"), [])

def test_maps_file_without_sidecar_is_valid_when_checksum_disabled(self):
    maps_file = expected_maps_path
    maps_file.touch()
    self.assertEqual(util.get_maps_file("8550-12704", checksum=False, download=False), maps_file)

def test_failed_firefly_download_raises(self):
    util.dl_firefly_mastar = lambda filename: False
    with self.assertRaises(FileNotFoundError):
        util.get_firefly_file()
```

- [ ] **Step 2: Run the tests and verify failure**

Run: `python -m unittest tests.test_data_io -v`

Expected: failures show the test-sample fallback, missing Firefly directory/failure handling, and MAPS sidecar requirement.

- [ ] **Step 3: Implement the minimum boundary fixes**

Change the missing-list branch to `return []`, create `self.firefly_dir` in `FitsUtil.__init__`, return an existing MAPS file before sidecar checks when `checksum=False`, and raise after an unsuccessful Firefly download:

```python
if not ret_path.exists() and (
    not self.dl_firefly_mastar(filename) or not ret_path.exists()
):
    raise FileNotFoundError(f"Firefly file unavailable: {ret_path}")
```

- [ ] **Step 4: Run the focused tests**

Run: `python -m unittest tests.test_data_io -v`

Expected: all tests pass.

### Task 3: Make Stage 1 selection, retry, and CSV writes correct

**Files:**
- Modify: `src/data/results.py:27`
- Modify: `src/data/results.py:88`
- Modify: `src/pipeline/stage1.py:48`
- Modify: `src/pipeline/stage1.py:393`
- Modify: `src/pipeline/stage1.py:415`
- Modify: `tests/test_data_io.py`
- Modify: `tests/test_stage1_pipeline.py`

**Interfaces:**
- Consumes: `store_params_file(..., write_lock=None)` and `get_processed_plate_ifus(..., successful_only=False, required_sample_filename=None)`.
- Produces: atomic CSV updates; the multiprocessing path supplies one shared lock; Stage 1 uses the configured all-sample path and stage-specific successful completion.

- [ ] **Step 1: Write failing result and Stage 1 tests**

Cover these assertions:

```python
self.assertEqual(get_processed_plate_ifus(
    "nfw.csv", tmp,
    successful_only=True,
    required_sample_filename="samples.nc",
), {"1000-10001"})

get_plateifu_list.assert_called_once_with(
    filepath=stage1.settings.data_dir / stage1.PLATES_FILENAME
)

get_processed_plate_ifus.assert_called_once_with(
    stage1.settings.nfw_param_cm200_filename,
    result_dir,
    successful_only=True,
    required_sample_filename=stage1.settings.nfw_param_cm200_sample_filename,
)
```

Also write 30 unique rows concurrently through `store_params_file(..., write_lock=lock)` and assert that the final CSV contains 30 indices.

- [ ] **Step 2: Run the tests and verify failure**

Run:

```powershell
python -m unittest tests.test_data_io tests.test_stage1_pipeline -v
```

Expected: failures show lost/unsupported locked writes, the wrong all-sample path, and RC-only completion filtering.

- [ ] **Step 3: Add locked atomic CSV replacement**

Wrap the current read-modify-write block with `write_lock` or `contextlib.nullcontext()`, then replace the output atomically:

```python
temp_output_file = output_file.with_name(
    f".{output_file.name}.{os.getpid()}.tmp"
)
df.to_csv(temp_output_file)
os.replace(temp_output_file, output_file)
```

Pass the optional lock through all three `store_params_file` calls in `process_plate_ifu`.

- [ ] **Step 4: Supply one process-safe lock to Stage 1 workers**

In the `n_cores > 1` branch, create a `multiprocessing.Manager().Lock()`, append it to each worker argument tuple, and have `process_plate_ifu_worker` forward it as `write_lock`.

- [ ] **Step 5: Make completion stage-specific**

For RC-only work, query successful `rc_param.csv` rows. For NFW work, query successful `nfw_param_cm200.csv` rows whose per-IFU sample file exists. Resolve `--ifu all` with:

```python
get_plateifu_list(filepath=settings.data_dir / PLATES_FILENAME)
```

- [ ] **Step 6: Run the focused tests**

Run:

```powershell
python -m unittest tests.test_data_io tests.test_stage1_pipeline -v
```

Expected: all tests pass and the concurrency test retains all 30 rows.

### Task 4: Enforce the Stage 2 IFU allowlist

**Files:**
- Modify: `src/data/results.py:512`
- Modify: `src/pipeline/stage2.py:7`
- Modify: `tests/test_stage2_pipeline.py`

**Interfaces:**
- Consumes: the CLI-required `ifu_file` and `merge_posterior_samples_file(filename, result_dir, plate_ifus=None)`.
- Produces: merged NetCDF rows only for explicitly requested IFUs.

- [ ] **Step 1: Write failing allowlist tests**

Update the existing dispatch test to create an IFU file and assert the data helper receives `plate_ifus={"1000-10001"}`. Add a real merge regression using two stored per-IFU NetCDF files and assert the merged coordinate contains only the allowed ID.

- [ ] **Step 2: Run the tests and verify failure**

Run: `python -m unittest tests.test_stage2_pipeline -v`

Expected: the current merge call does not pass or enforce the allowlist.

- [ ] **Step 3: Load and forward the required allowlist**

Resolve `ifu_file` through `settings.resolve_input_path()`, raise `FileNotFoundError` when absent, load it with `get_plateifu_list`, and pass a `set[str]` to the data helper. In the helper, skip loaded rows not present in the set.

- [ ] **Step 4: Run the focused tests**

Run: `python -m unittest tests.test_stage2_pipeline -v`

Expected: all tests pass and unrelated samples are excluded.

### Task 5: Match PSIS to the fitted Student-t population

**Files:**
- Modify: `src/stats/psis.py:145`
- Modify: `tests/test_population_pipeline.py`

**Interfaces:**
- Consumes: `nu_pop_median`, `M200_mu_median`, `M200_sigma_median`, `log10_c0_median`, `alpha_median`, and `sigma_int_median`.
- Produces: PSIS weights from the same factored Student-t density used by the population sample likelihood.

- [ ] **Step 1: Write a failing sensitivity test**

Call `compute_psis_importance_diagnostics(..., save_plots=False)` twice with identical posterior samples and only `nu_pop_median` changed from `3.0` to `100.0`. Assert the ESS arrays are not equal.

- [ ] **Step 2: Run the test and verify failure**

Run: `python -m unittest tests.test_population_pipeline -v`

Expected: the ESS arrays are currently equal because `nu_pop_median` is unused.

- [ ] **Step 3: Use SciPy Student-t log densities**

Import `t` from `scipy.stats` and replace the two Normal population terms with:

```python
log_p_pop = t.logpdf(
    c_samps, df=nu_pop, loc=mu_c_given_m, scale=sigma_int
) + t.logpdf(
    m_samps, df=nu_pop, loc=M200_mu, scale=M200_sigma
)
```

- [ ] **Step 4: Run the focused test**

Run: `python -m unittest tests.test_population_pipeline -v`

Expected: all tests pass and the diagnostic changes with `nu_pop_median`.

### Task 6: Correct solver documentation and verify the repository

**Files:**
- Modify: `docs/Data-Processing-Pipeline.md:101`

**Interfaces:**
- Consumes: `RotCurve.fit_model = "mcmc"` and current `_inf_vel_rot` behavior.
- Produces: documentation that accurately states the runtime default without changing code.

- [ ] **Step 1: Update the solver description**

Replace the `lmfit`-default text with a concise description of the MCMC default, including inferred inclination/`phi_delta`, intrinsic scatter, and Student-t likelihood. Keep the `lmfit` path documented only as an available internal alternative.

- [ ] **Step 2: Run fresh verification**

Run:

```powershell
python -m unittest discover -v
python -m src --help
python -m src stage1 --help
python -m src merge --help
git diff --check
```

Expected: all unit tests pass, all help commands exit 0, and `git diff --check` reports no whitespace errors.

- [ ] **Step 3: Inspect final scope**

Run: `git status --short` and `git diff --stat HEAD`

Expected: only the review confirmation, plan, approved source/tests/docs changes, and the pre-existing `.gitignore` modification are present; `.gitignore` remains unmodified by this work.
