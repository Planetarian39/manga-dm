---
title: Lessons from Developing the Inference Workflow
description: Practical, non-result lessons from moving from point fits to Bayesian posterior inference.
---

# Lessons from developing the inference workflow

This page records method-development lessons that help readers understand the
repository. It is not a population result, an originality claim, or a claim
that one tool is universally superior.

## Limits encountered with point estimation

Early rotation-curve experiments used least-squares style point fitting. A
simple stellar model could reproduce broad trends, but adding a dark-matter
component exposed weak identifiability: different starting values could lead
to different parameter combinations with similar fitted velocities.

The central problem was not that optimization was incapable of fitting the
data. It was that a single optimum did not communicate the uncertainty and
curved $M_{200}$-$c$ degeneracy needed by later analysis. Repeated starts and
curvature diagnostics remained useful for debugging, but a distributional
representation was needed for the scientific workflow.

## Moving to posterior sampling

Bayesian modelling made assumptions explicit as priors and a likelihood, and
MCMC provided samples that could represent skewness and correlation. The
change also introduced new responsibilities:

- prior predictive checks;
- careful parameterization;
- multiple-chain and HMC diagnostics;
- posterior predictive checks;
- reproducible sampler and backend settings.

The $M_{200}$-$c$ ridge remained present. Posterior sampling revealed it; it
did not make the underlying data more informative.

## Prior-design trade-offs

The practical goal was to stabilize weakly identified directions without
inserting the desired relation into each galaxy fit.

- Photometric information anchors stellar mass and size.
- A stellar-to-halo mass relation regularizes halo mass.
- A wider concentration prior allows the rotation-curve shape to update
  concentration.

If both halo mass and concentration priors are extremely broad, the sampler
can spend substantial time along physically implausible parts of the ridge.
If both are very tight, the posterior mostly reproduces the prior. Prior
predictive checks and sensitivity runs are therefore part of model
development, not optional presentation.

The canonical numerical priors belong in
[Single-galaxy NFW inference](/methods/single-galaxy-nfw#priors);
this page preserves the design reasoning rather than a second copy of values.

## Iterative model checking

Model development proceeded through repeated cycles:

1. fit a small diagnostic case;
2. inspect R-hat, ESS, divergences, traces, and posterior geometry;
3. inspect predictive residuals and physical plausibility;
4. revise priors, parameterization, or likelihood weighting;
5. rerun the same checks.

This process motivated the robust Student-t likelihood, explicit residual
scatter, radial weighting, and the separation between single-galaxy and
population inference. Each choice should still be judged against the
paper-aligned specification and reproducibility tests rather than accepted
because it appears in the current implementation.

## Sampler and backend evaluation

The project evaluated more than one MCMC implementation. PyMC supplies the
probability-model representation and ArviZ-compatible diagnostics; the chosen
NUTS backend also has to support automatic differentiation through the
modified Bessel functions used by the exponential-disk velocity term.

Backend choice is therefore part of the model's reproducibility surface:

- record PyMC, PyTensor, backend, NumPy, and ArviZ versions;
- record draws, warmup, chains, target acceptance, and random seed;
- compare posterior shapes on fixed diagnostic cases after backend changes;
- retain divergences and sampler statistics with the posterior output.

The public documentation should describe only comparisons that can be
reproduced from retained artifacts.

## Lessons

The theory of Bayes' theorem is the shortest part of a reliable application.
Most work lies in:

- deciding what information a prior may legitimately encode;
- finding parameterizations that a gradient sampler can explore;
- separating measurement error from model inadequacy;
- checking whether a posterior feature is data-informed or prior-sensitive;
- carrying uncertainty into downstream inference without double-counting the
  first-stage prior.

Those lessons explain the repository's emphasis on full posterior files,
diagnostic figures, quality gates, configuration provenance, and the
prior-corrected Stage 2 likelihood.
