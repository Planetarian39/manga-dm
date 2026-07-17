---
title: Methods
description: Manuscript-aligned methodology for the MaNGA dark-matter inference pipeline.
---

# Methods

This section documents the analysis model implemented by `manga-dm`, using the current manuscript methodology as the scientific reference. It explains how spatially resolved MaNGA gas kinematics are screened, modeled galaxy by galaxy, and propagated into a population model without reducing each posterior to a point estimate.

::: info What this section demonstrates
- **Research decision:** preserve the complete chain from observational screening to population inference.
- **My implementation contribution:** I built the Bayesian galaxy-level and population-level workflow and connected it to diagnostics, provenance, and public documentation.
- **Main limitation:** the current CLI contains the scientific components but does not yet expose one versioned manuscript-reproduction profile.
:::

The documentation deliberately separates two questions:

1. **What method defines the manuscript analysis?** The equations, priors, selection rules, sampler configuration, and diagnostic requirements on these pages answer this question.
2. **What does the current public command line run by default?** Implementation notes identify places where today's defaults are not yet a manuscript-reproduction profile.

## Analysis path

1. [Data products and sample selection](./data-and-selection.md) prepares MaNGA DR17 velocity-field and photometric inputs, applies spaxel masks, and defines the screening population.
2. [Empirical rotation curves](./empirical-rotation-curves.md) fit a robust, low-dimensional velocity model to decide whether a galaxy has sufficiently ordered disk-like kinematics.
3. [Single-galaxy NFW inference](./single-galaxy-nfw.md) decomposes rotational support into stellar, dark-matter, and pressure-support terms while retaining the complete posterior.
4. [Population modeling](./population-model.md) reuses those posterior samples in a prior-corrected importance-sampling likelihood.
5. [Diagnostics and quality gates](./diagnostics-and-quality-gates.md) combines convergence, posterior-predictive, fit-quality, degeneracy, and importance-weight checks.

The model is intentionally conditional: it describes galaxies that pass the documented kinematic and inferential gates. It is not an unconditional model for every MaNGA galaxy or for the halo population as a whole.

<MethodStatus status="paper">

The manuscript method uses a major-axis cut of 60°, predictive interval
probability `0.9545`, the full quality equation including R-hat and ESS,
and a prior-corrected posterior-sample Stage 2 likelihood.

</MethodStatus>

<MethodStatus status="implementation">

The public CLI has no single versioned manuscript profile. Current fallbacks use
45°, `0.95`, non-manuscript `recommended`/`strict` presets, and a GMM Stage
2 path by default.

</MethodStatus>

## What is included

The public method record includes:

- observational inputs and their use;
- projection and rotation-curve equations;
- all single-galaxy and population priors;
- robust likelihoods and inner-radius weighting;
- the single-galaxy NUTS configuration;
- the final retention equations;
- the prior correction used when posterior samples become population inputs;
- diagnostic interpretation and model limitations.

## Publication boundary

These pages do not report sample attrition, aggregate posterior values, population figures, sensitivity-test outcomes, comparisons with simulations or other studies, physical interpretation, novelty claims, or conclusions from the unpublished analysis. The equations and rules are sufficient to understand and audit the pipeline without disclosing those results.

For the code-facing view, continue to [Implementation status](../project/implementation-status.md). For scientific scope and assumptions, see [Limitations](../project/limitations.md).
