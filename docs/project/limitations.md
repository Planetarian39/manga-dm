---
title: Limitations
description: Scientific scope, assumptions, and interpretation boundaries.
---

# Limitations

`manga-dm` is a controlled dynamical baseline, not a complete reconstruction of galaxy mass distributions. These assumptions should accompany any use of its outputs.

## Selection conditioning

The empirical screen favors velocity fields with ordered disk-like rotation, adequate radial coverage, stable sampling, and predictive agreement. The population model is conditional on passing those gates and should not be generalized to all MaNGA galaxies or all dark-matter halos without an explicit selection model. No aggregate selection counts are needed to state this boundary.

## Limited radial leverage

MaNGA velocity fields cover only a fraction of $R_{200}$. Consequently, $M_{200}$ and $c$ can trade off along curved posterior ridges. The SHMR prior helps identify a halo-mass scale, but also makes the result sensitive to its calibration and width. Prior-corrected population reweighting preserves the sampled ridge; it cannot create radial information that was not observed.

## Stellar and gas model

The baryonic force is approximated by a Hernquist bulge plus an exponential stellar disk. The baseline omits explicit gas gravity, bars, nuclear components, thick disks, mass-to-light-ratio gradients, and non-axisymmetric streaming. Unmodeled baryonic support can be absorbed by fitted halo parameters.

## Pressure support

$$
V_{\mathrm{drift}}^2=2\sigma_0^2(r/R_d).
$$

This single fitted scale does not model radial dispersion gradients, shocks, or bar-driven flows, and is not calibrated point by point to the observed dispersion map.

## Beam smearing

The likelihood down-weights the inner region with a logistic function centered at $0.3r_{\max}$. It does not convolve the model with the instrumental PSF. Because concentration is informed by the inner rotation-curve rise, residual beam smearing can affect $c$.

## Fixed inclination

The baseline fixes inclination to the photometric value. This avoids a strong degeneracy with velocity amplitude and central weighting, but conditions posterior uncertainty on that estimate. Unpublished sensitivity-test values are not part of this public statement.

## Halo profile

NFW is a fiducial, literature-comparable profile with a fixed inner cusp. The pipeline does not establish that it is unique. Cores, contraction, feedback-modified structure, or other departures may map into effective NFW parameters.

## Robust residual model

Student-t residuals limit the influence of outlying spaxels and absorb extra mismatch through $\sigma_{\mathrm{int}}$. Robustness does not explain the physical origin of residuals or validate axisymmetry.

## Importance support

Stage 2 depends on overlap between each single-galaxy posterior and the candidate population density. Large Pareto $\hat k$ or small importance ESS means that few draws dominate. Truncation controls variance but introduces finite-sample bias; severe mismatch can require iterated importance sampling or a joint hierarchical model.

## Interpretation boundary

Outputs are constraints under the selected data, priors, NFW-equivalent profile, fixed inclination, stellar-only gravitational baryon model, pressure-support approximation, and quality gates. They are not assumption-free halo measurements.

::: warning Paper method and current implementation
Scientific limitations and implementation differences are separate. Current defaults do not instantiate the complete paper configuration: the paper profile requires $60^\circ$, `0.9545`, the full $\hat R$/ESS and NFW quality equations, and the prior-corrected sample likelihood. Current fallbacks use $45^\circ$, `0.95`, non-paper presets, warning-only empirical convergence enforcement, and GMM Stage 2 inputs by default. See [Implementation status](./implementation-status.md); do not fold these differences into scientific uncertainty.
:::

For the equations, return to [Methods](../methods/index.md).
