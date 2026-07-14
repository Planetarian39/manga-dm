# Public Documentation Site Design

## Purpose and Audience

Build an English-language GitHub Pages site for research readers who want to understand the MaNGA dark-matter analysis method before using the code. The site explains the finalized paper methodology, maps each method to the repository implementation, and provides a secondary path for installation, CLI use, configuration, and output interpretation.

The finalized paper at `D:/code/manga-dev/docs/latex/thesis.tex` is the scientific source of truth. Current CLI defaults must not be used to rewrite the paper method. Where the implementation differs from the paper, the public site identifies the paper-aligned method and links it to an implementation-status note; scientific-code corrections are deferred to the separate Sprint summary below.

## Publication Boundary

The public site may include:

- the complete method, including equations, priors, likelihoods, selection rules, sampler settings, diagnostics, and limitations needed to understand the code;
- code architecture, CLI workflow, configuration behavior, file formats, and reproducibility instructions;
- one deep case study for MaNGA 11743-9102 and three supporting cases (8994-12701, 7977-3704, and 9493-6101);
- complete single-galaxy posterior NetCDF files for those four cases, plus their individual fit and diagnostic figures.

The public site must exclude:

- novelty or priority claims;
- aggregate sample counts, population-fit values, headline findings, comparisons with simulations or published results, physical interpretation, discussion, and conclusions;
- aggregate paper result figures, attrition plots with unpublished counts, sensitivity results, and population residual plots;
- the paper PDF, LaTeX source, bibliography database, raw MaNGA FITS products, and full-sample result tables or posterior collections.

Every page derived from the paper is rewritten as project documentation rather than copied as a paper section. A publication-boundary check scans the built source for forbidden aggregate-result language and validates that only allowlisted case-study assets are published.

## Site Architecture and Visual System

Keep GitHub Pages as the only public deployment target. Preserve the existing `docs/` source and Jekyll-compatible build, replacing the minimal `minima` presentation with a repository-owned static layout, styles, includes, and navigation. Do not add Sites hosting or a server runtime.

Use the approved **Astral Ledger** direction:

- warm paper backgrounds, deep navy text and surfaces, muted stellar-gold accents, and restrained rust highlights;
- serif editorial display headings, readable sans-serif body text, and monospace scientific labels and commands;
- an asymmetrical research-focused homepage, generous whitespace, thin technical dividers, and compact scientific figures;
- responsive navigation, visible focus states, reduced-motion support, WCAG AA contrast, and layouts that work from 320 px mobile width through 1600 px desktop width;
- CSS and real project figures only; no generated decorative SVGs, browser frames, or unrelated imagery.

The homepage leads with the scientific workflow rather than installation. It presents the four-stage path—data and selection, empirical rotation curves, single-galaxy NFW inference, and population inference—then routes readers to methods, the 11743-9102 case study, and the reproducibility guide.

## Information Architecture

Organize the public site into six navigation groups:

1. **Overview**
   - Documentation home: research purpose, four-stage workflow, disclosure notice, and reader paths.
   - Project overview: capabilities, repository status, and architecture at a conceptual level.
2. **Methods**
   - Data products and sample selection.
   - Empirical rotation-curve screening.
   - Single-galaxy dynamical and NFW model.
   - Population model and prior-corrected posterior-sample likelihood.
   - Diagnostics and quality gates.
3. **Case Studies**
   - Deep walkthrough for 11743-9102: inputs, velocity field, rotation curve, component decomposition, posterior geometry, convergence, and interpretation limits.
   - Supporting gallery for 8994-12701, 7977-3704, and 9493-6101, limited to method demonstrations without scientific comparison.
   - Download page for the four complete per-galaxy posterior files with provenance and schema notes.
4. **Run the Pipeline**
   - Installation and environment setup.
   - End-to-end CLI workflow.
   - Configuration and paper-aligned parameter profile.
   - Inputs, outputs, and result-file schemas.
5. **Background**
   - The existing MCMC guide, edited into a concise Bayesian background page and a diagnostics-reading page; its 11743-9102 material moves to the case-study section.
6. **Project**
   - Module architecture and code-to-method map.
   - Limitations and implementation status.
   - The inner rotation-curve-shape document retained under a clearly labeled “Future research” section.

The README remains a compact repository landing page and links to the GitHub Pages home, quick start, method overview, and case study. It does not duplicate the full site.

## Content and Asset Flow

For each method page:

1. Extract the relevant finalized-paper method statements and equations.
2. Remove paper narrative, unpublished counts, result interpretation, and novelty framing.
3. Cross-check terminology, symbols, constants, and parameter values against the finalized paper.
4. Map the method to current `src/data`, `src/models`, `src/pipeline`, `src/stats`, and `src/viz` implementation entry points.
5. Add an implementation-status callout when the public CLI or defaults do not reproduce the paper configuration.

Case-study assets come from the finalized paper result set. Extract only the four allowlisted per-galaxy posterior NetCDF files from `D:/code/manga-dev/data/results-20260326-final.zip`. Use individual-galaxy figures from the finalized result/LaTeX figure directories, convert PDF-only figures to web PNG/WebP where necessary, and record source path, SHA-256 digest, galaxy ID, artifact type, and public destination in a provenance manifest. Do not publish the aggregate archive.

## Confirmed Code-Alignment Sprint Summary

A scientific-code alignment Sprint is required before the site can claim that the default public CLI reproduces the finalized paper. The Sprint is a future code task only; this documentation project must not create its full Sprint file or execute its changes.

The Sprint should cover:

- add a versioned paper configuration profile with the finalized screening values, including the 60-degree azimuthal cut and 0.9545 predictive-HDI probability, instead of relying on the current 45-degree and 0.95 fallbacks;
- add an explicit paper quality preset matching the finalized retention equation, including reduced chi-squared at most 2.0, posterior-predictive p-value from 0.1 through 0.9, and absolute concentration–mass posterior correlation at most 0.85, while retaining the finalized R-hat and ESS gates;
- expose the prior-corrected posterior-sample likelihood as the paper-aligned Stage 2 CLI path and prevent the current default GMM path from being presented as the paper method;
- preserve the existing GMM path as an explicitly named alternative unless numerical review shows it should be removed;
- add provenance metadata to saved fits so outputs record configuration profile, likelihood mode, sampler settings, and code version;
- run numerical regression checks against the finalized four single-galaxy cases and an approved population-level reference result before accepting any conclusion-affecting change.

Minimum Sprint verification should include CLI help tests, configuration/preset unit tests, Stage 1 and Stage 2 dispatch tests, likelihood-selection tests, deterministic data-preparation checks, four-case posterior/summary tolerances, and an approved full reference comparison for the population result. Exact tolerances must be defined from the archived finalized outputs before implementation.

## Validation and Acceptance

The documentation implementation is accepted when:

- Jekyll builds the site without broken internal links, missing front matter, or missing assets;
- the homepage and navigation render correctly at desktop, tablet, and mobile breakpoints;
- equations, tables, code blocks, figures, keyboard focus, contrast, and reduced-motion behavior are usable;
- all method values match the finalized paper and all implementation differences are explicitly identified;
- only the four allowlisted complete posterior files and individual-case figures are present under public assets;
- the publication-boundary check finds no aggregate results, conclusions, novelty claims, or non-allowlisted paper assets;
- `README.md`, `manga --help`, all relevant subcommand help, and the existing unit test suite still pass;
- direct HTTP checks confirm the deployed GitHub Pages URL and key routes return successfully after publication.

## Assumptions

- The GitHub Pages repository/branch setting will be configured to publish from `docs/`; both currently inferred Pages URLs return 404, so activation is part of the deployment handoff.
- GitHub Pages is the canonical public site; Sites is used only for design guidance.
- Scientific methods follow the finalized paper even where current defaults differ.
- Aggregate conclusions remain private until the user explicitly changes the publication boundary.
- Existing unrelated working-tree deletions under `docs/superpowers/` are user-owned and must not be restored or staged as part of this work.
