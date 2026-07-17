---
title: Two-Minute Project Overview
description: A concise account of the research question, method, contribution, evidence, and limits of manga-dm.
---

# Two-minute project overview

`manga-dm` is my computational astrophysics project on what spatially resolved galaxy kinematics can tell us about dark-matter halo mass and concentration. I built a Python/PyMC 5 workflow that connects MaNGA velocity-field screening, galaxy-level Bayesian inference, posterior diagnostics, and population-level uncertainty propagation.

I am **Hongyi Xu**, an undergraduate student in the Department of Physics at the University of Toronto. This public site accompanies a first-author manuscript in preparation for arXiv and subsequent journal submission.

## The scientific question

MaNGA DR17 provides integral-field spectroscopy across nearby galaxies. Its ionized-gas velocity fields reveal ordered rotation, but they cover only the inner part of a dark-matter halo. Within this limited radial range, a larger halo mass $M_{200}$ can be offset by a lower concentration $c$, producing a curved and correlated posterior rather than one uniquely determined best fit.

The project asks:

> How well can spatially resolved ionized-gas kinematics constrain dark-matter halo mass and concentration when selection, baryonic modeling, radial coverage, and posterior uncertainty are treated explicitly?

## The inference chain

1. **Select.** Prepare MaNGA velocity-field and photometric inputs, mask unreliable spaxels, and screen for disk-like kinematics.
2. **Resolve.** Fit a robust empirical rotation curve to check posterior stability and predictive adequacy before physical mass decomposition.
3. **Infer.** Sample a stellar-plus-NFW dynamical model with NUTS, retaining the correlated $M_{200}$-$c$ posterior for each galaxy.
4. **Propagate.** Reuse full posterior samples in a prior-corrected population likelihood rather than compressing each galaxy to a point estimate.

[Follow the complete method →](/methods/)

## My contribution

| Area | Work completed | Public evidence |
|---|---|---|
| Data and screening | Connected MaNGA inputs, projected geometry, masks, and quality gates in a reproducible pipeline | [Data and selection](/methods/data-and-selection) |
| Bayesian modeling | Built galaxy-level stellar-plus-NFW and population-level inference workflows in Python/PyMC 5 | [Single-galaxy model](/methods/single-galaxy-nfw) · [Population model](/methods/population-model) |
| Uncertainty quantification | Preserved full posterior geometry and documented why median-only compression is inadequate | [Worked galaxy](/case-studies/11743-9102) |
| Diagnostics | Evaluated convergence, effective sample size, posterior geometry, predictive adequacy, and implementation differences | [Diagnostics](/methods/diagnostics-and-quality-gates) |
| Scientific software | Organized a thin CLI, responsibility-oriented modules, provenance records, and automated documentation checks | [Architecture](/project/architecture) |
| Public record | Prepared the research narrative, method documentation, reviewable artifacts, and explicit limitations | [Downloads and provenance](/case-studies/downloads) |

## Evidence available for review

- four allowlisted single-galaxy examples with fit figures and complete NetCDF posterior files;
- a representative case with 4,000 retained posterior draws and directly generated quantiles;
- convergence, effective-sample-size, posterior-geometry, and prior-to-posterior diagnostic records;
- byte counts, SHA-256 digests, schemas, and source provenance for downloadable artifacts;
- a tested `manga` CLI, modular Python architecture, documentation checks, and deployment-level route checks.

The [11743-9102 worked example](/case-studies/11743-9102) is the fastest way to inspect the scientific and computational evidence together.

## What the results mean—and do not mean

The inference is conditional on the selected data, quality gates, stellar model, NFW halo profile, fixed photometric inclination, pressure-support approximation, and available radial range. The public examples demonstrate the method and posterior diagnostics; they are not a released population result.

The repository contains the core scientific components of the manuscript method. Its current fallback thresholds and default Stage 2 route are not yet one versioned manuscript-reproduction profile. The [implementation status](/project/implementation-status) states the exact differences and required validation.

## Continue reading

- [Research methods](/methods/)
- [Worked example: 11743-9102](/case-studies/11743-9102)
- [Run the pipeline](/run/)
- [About Hongyi Xu](/about/)
