# AGENTS.md

This file provides operating rules for future automated agents working in this repository. The goal is to keep changes aligned with the current code structure and avoid breaking the scientific pipeline.

## Project Overview

- Repository name: `manga-dm`
- Language: Python
- Domain: MaNGA dark matter analysis and fitting pipeline
- Current official entry point: `manga` CLI

## Directory Responsibilities

- `src/`: current main implementation
- `src/cli/`: CLI dispatch layer, argument parsing only
- `src/config/`: configuration and constants
- `src/data/`: data loading, download, and result I/O
- `src/models/`: scientific models and PyMC-related implementation
- `src/pipeline/`: workflow orchestration
- `src/stats/`: statistical utilities
- `src/viz/`: visualization
- `docs/`: VitePress public site source, project notes, and technical articles
- `docs/.vitepress/`: public-site configuration and theme
- `package.json` and `package-lock.json`: Node 20 documentation toolchain only; Python packaging remains in `pyproject.toml`
- `project/superpowers/`: local Superpowers specifications and implementation plans
- `var/`: runtime-generated content or temporary data, treated as volatile

## Superpowers Document Location

- Store all Superpowers-generated design specifications under `project/superpowers/specs/`
- Store all Superpowers-generated implementation plans under `project/superpowers/plans/`
- Do not create or retain Superpowers workflow documents under `docs/superpowers/`
- Keep the existing `project/` ignore policy unless the user explicitly requests these local workflow documents to be versioned

## Key Entry Points

- CLI: `python -m src` or `manga` after installation
- Unified help: `manga --help`
- Subcommands:
  - `manga select`
  - `manga stage1`
  - `manga stage2`
  - `manga figures`
  - `manga merge`
  - `manga sample`

## Configuration Rules

- Prefer `src.config.settings`; do not read `config.toml` directly from business logic
- Follow the lookup order defined in `src/config/settings.py`
- Resolve paths through `settings.resolve_input_path()`, `settings.resolve_result_dir()`, and related helpers
- Do not scatter hard-coded `data/`, `results/`, or relative path logic across multiple modules

## Code Organization Principles

- Keep business logic out of `cli/`; it should only parse arguments and forward calls
- Keep workflow orchestration in `pipeline/`; do not place core model logic there
- Keep scientific model implementations in `models/`, especially PyMC code
- Keep external data and result file I/O in `data/`
- Keep model-independent statistical tools in `stats/`
- Keep plotting logic and shared visualization helpers in `viz/`
- When adding new functionality, place it in the module that matches its responsibility best instead of piling it into one large file

## Modification Priority

1. Preserve behavior first, then improve structure
2. Keep the official `manga` CLI runnable
3. Extract reusable logic into new modules before rewriting callers
4. Avoid changing scientific model internals unless explicitly requested and numerically validated

## Explicit Do Nots

- Do not casually change internal PyMC implementations unless the user explicitly asks and numerical consistency has been verified
- Do not change output file formats, field names, or naming conventions unless a migration plan exists
- Do not introduce global state, singleton caches, or hidden side effects unless necessary
- Do not keep adding data processing, statistics, and plotting logic to the CLI layer

## Coding Conventions

- Use ASCII by default
- Prefer clear module boundaries and short functions for new files
- Keep comments focused on why something exists, not on obvious line-by-line behavior
- Avoid unnecessary refactors and large formatting-only edits unless the task requires them
- Preserve existing signatures when a legacy API already exists

## Dependencies and Environment

- The project uses `pyproject.toml`
- Runtime core dependencies include `numpy`, `scipy`, `matplotlib`, `astropy`, `pandas`, `xarray`, `pymc`, `arviz`, and `lmfit`
- The documented development environment is Windows + Anaconda
- If a change seems to require a new dependency, first confirm whether it can be solved with the current dependency set

## Common Commands

```bash
pip install -e .
manga --help
manga select --download
manga stage1 --ifu test --nfw
manga stage2 --fit --quality-cut recommended
manga figures --ifu 8994-12701 7977-3704
```

## Verification Requirements

- After changing the CLI, at minimum check that `manga --help` and the relevant subcommand help still work
- After changing data or pipeline code, run the smallest practical smoke test
- After changing visualization code, confirm the output files can be generated
- If the change touches numeric results, prefer comparing against existing test samples or minimal samples for stability
- If full validation is not possible, state clearly what was not verified

## Working Style

- Read the relevant modules first, then edit
- Touch only files directly related to the task
- If the repository already contains undocumented changes, do not revert them blindly
- If the task is a refactor, preserve compatibility first and migrate incrementally

## Reference Documents

- `README.md`
- `pyproject.toml`
- `src/cli/main.py`
- `src/config/settings.py`

## Shortest Principles for Future Agents

- Preserve compatibility first, then improve structure
- Extract repeated logic before changing imports
- Validate the smallest useful slice before declaring completion
