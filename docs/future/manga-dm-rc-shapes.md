---
title: "Research Option: Inner Rotation-Curve Shapes"
---

# Research Option: Inner Rotation-Curve Shapes and Dark-Matter-Relevant Central Concentration

This note is a public-facing research direction for extending `manga-dm`. It is not part of the current production CLI workflow; it frames a possible next project around inner rotation-curve morphology, baryonic decomposition, and dark-matter-relevant candidate discovery.

## 1. Working Title

**Identifying central dynamical concentration types from MaNGA inner rotation-curve shapes and their connection to baryon- and dark-matter-dominated regimes**

Short title:

**MaNGA inner rotation-curve shapes and dark matter dominance**


## 2. Core Idea

This project uses the inner shapes of MaNGA galaxy rotation curves to identify different types of central dynamical concentration, then tests how those types relate to baryon-dominated and dark-matter-dominated dynamical regimes.

The central idea is not to infer dark matter directly from the raw inner slope alone. The inner rotation curve is shaped by stars, gas pressure support, inclination, beam smearing, non-circular motion, and dark matter at the same time. Therefore, the scientifically safer framing is:

**Use inner rotation-curve morphology as an observational entry point, then combine it with baryonic mass modeling and Bayesian dynamical inference to assess whether a galaxy is centrally baryon dominated, dark-matter supported, diffuse/core-like, or dynamically mismatched.**


## 3. Scientific Motivation

The inner rise of a rotation curve traces how quickly the enclosed gravitational support grows toward the galaxy center. A steep inner rise usually indicates a centrally concentrated mass distribution, while a slowly rising inner curve may indicate a more diffuse central potential, a core-like mass distribution, strong beam-smearing effects, or disturbed kinematics.

For dark matter studies, this is useful because the inner rotation-curve shape can help identify galaxies where:

- The stellar component already explains most of the central gravitational support.
- A significant dark matter contribution is required even in the inner region.
- The central mass distribution appears unusually diffuse.
- Standard halo models fail to reproduce the observed central kinematics.

This creates a direct bridge between MaNGA resolved kinematics, baryon-dark matter decomposition, and machine-learning-based sample discovery.


## 4. Main Scientific Questions

1. Can inner rotation-curve shape features classify galaxies into physically meaningful central dynamical concentration types?
2. Are these concentration types correlated with baryon-dominated or dark-matter-dominated dynamical states?
3. Do some inner-shape classes systematically correspond to poor NFW fits, broad halo posteriors, strong parameter degeneracy, or possible preference for core-like halo models?
4. Can machine learning recover these classes from MaNGA velocity fields and related maps more robustly than hand-crafted rotation-curve features alone?


## 5. Proposed Concentration Types

The project should avoid vague anomaly labels. A useful starting taxonomy is:

- **Baryon-concentrated systems**: The inner rotation curve rises rapidly, and the stellar bulge/disk contribution can explain most of the central support.
- **Dark-matter-supported inner-rise systems**: The inner curve rises strongly, but the baryonic model is insufficient, requiring a significant dark matter contribution at small radii.
- **Diffuse or core-like candidates**: The inner curve rises slowly after controlling for resolution and baryonic structure, suggesting a less concentrated central mass distribution.
- **Model-mismatch or disturbed systems**: The inner shape is unusual, but residuals, non-circular motions, geometry, or posterior predictive checks indicate that simple mass decomposition is unreliable.

The second and third classes are the most relevant for dark matter science. The fourth class is still important because it marks the boundary where dark matter inference should be treated cautiously.


## 6. Data Inputs

The first version should use MaNGA MAPS and catalog-level products rather than full spectral datacubes.

Recommended inputs:

- Halpha gas velocity map.
- Gas velocity dispersion map.
- Halpha flux or equivalent-width map.
- Stellar velocity and stellar dispersion maps, if available at sufficient quality.
- Measurement uncertainty and quality maps such as IVAR, SNR, and valid-spaxel masks.
- Global catalog features such as stellar mass, effective radius, Sersic index, axis ratio, inclination proxy, and redshift.
- Existing empirical rotation-curve outputs and Bayesian dynamical-model diagnostics when available.

Full datacube modeling should be deferred until the project has a working map-level baseline.


## 7. Feature Baselines

The first technical step should be a transparent feature-based baseline. Candidate inner-shape features include:

- Inner logarithmic slope, such as `d log V / d log r`.
- Central velocity gradient.
- `V(0.5 Re) / V(2 Re)`.
- `V(1 kpc) / Vmax`, where physical resolution allows it.
- Turnover radius or normalized transition radius.
- Inner curvature of the rotation curve.
- Residual asymmetry between approaching and receding sides.
- Posterior predictive coverage or residual structure in the inner region.

These features provide an interpretable baseline before training a neural representation model.


## 8. Physical Normalization

Inner rotation-curve shape alone is not enough. The same observed shape can have different physical meanings depending on the baryonic mass distribution.

The key physical quantities should be defined in terms of velocity contributions:

```text
f_baryon(r) = V_baryon(r)^2 / V_rot(r)^2
f_dm(r)     = V_dm(r)^2     / V_rot(r)^2
```

Useful evaluation radii include:

- `0.5 Re`
- `1 Re`
- `2 Re`
- fixed physical radii such as `1 kpc`, only when spatial resolution allows meaningful comparison

The central scientific classification should combine inner shape and dynamical decomposition:

- Steep inner rise plus high `f_baryon`: baryon-concentrated.
- Steep inner rise plus high `f_dm`: dark-matter-supported inner rise.
- Slow inner rise plus low central concentration after resolution checks: diffuse/core-like candidate.
- Poor posterior predictive performance: model-mismatch or disturbed.


## 9. Machine Learning Role

Machine learning should enter after the interpretable baseline is established.

Recommended model path:

1. **Baseline models**: XGBoost, random forest, or shallow MLP using hand-crafted features and catalog parameters.
2. **Map-level encoders**: CNN or compact vision encoder using velocity, dispersion, flux, and quality maps.
3. **Multimodal encoder**: One branch for 2D maps and one branch for catalog/summary features, fused into a galaxy representation.
4. **Downstream tasks**: Predict concentration type, identify dark-matter-relevant candidates, and perform similarity search in embedding space.

The ML model should not be trained primarily to predict `M200` and `c` in the first version. A better target is the central dynamical concentration type or the reliability/instability of dark matter inference.


## 10. Connection to Bayesian MCMC

Bayesian MCMC is a natural calibration and interpretation layer for this project.

MCMC can provide:

- Posterior distributions of baryonic and dark matter velocity contributions.
- `f_dm(r)` and `f_baryon(r)` with uncertainty.
- NFW fit quality and posterior predictive checks.
- Strength of the `M200-c` degeneracy.
- Evidence for whether alternative halo models such as Burkert or Einasto are preferred in selected subclasses.

The recommended coupling is:

**Machine learning discovers candidate concentration classes at scale; Bayesian dynamical modeling explains whether those classes correspond to baryon dominance, dark matter dominance, diffuse central mass structure, or model failure.**

This avoids treating current MCMC point estimates as ground-truth labels while still using Bayesian inference as the physical anchor.


## 11. Evaluation Strategy

The project should be evaluated at three levels:

- **Feature validity**: Do hand-crafted inner-shape metrics correlate with physically meaningful baryon/dark matter fractions?
- **Model validity**: Does the ML representation recover concentration types more robustly than simple feature thresholds?
- **Physical validity**: Do the discovered classes show distinct MCMC posterior behavior, fit quality, residual structure, or halo-model preference?

Important controls:

- Match or control for stellar mass and redshift.
- Account for inclination uncertainty.
- Exclude or separately label low-resolution inner regions.
- Track beam-smearing sensitivity.
- Separate physical core-like candidates from poor-quality or disturbed systems.


## 12. Main Risks

The main risk is over-interpreting the inner rotation-curve shape as a direct dark matter signature. The inner curve is strongly affected by baryons and observational systematics.

Specific risks:

- Beam smearing can flatten the inner velocity gradient.
- Inclination errors can distort deprojected velocities.
- Bars, warps, and non-circular motions can mimic unusual central mass profiles.
- Stellar mass-to-light assumptions can change the inferred baryon fraction.
- MaNGA radial coverage is limited, so halo cusp/core conclusions must be phrased cautiously.

The project should therefore use language such as "central dynamical concentration" and "dark-matter-relevant candidates" rather than claiming direct dark matter core/cusp identification from the inner curve alone.


## 13. Expected Outputs

Expected scientific outputs include:

- A catalog of MaNGA galaxies classified by central dynamical concentration type.
- A set of interpretable inner rotation-curve shape metrics.
- A machine-learning representation for identifying dark-matter-relevant central structures.
- A comparison of baryon-dominated and dark-matter-dominated inner regimes.
- A candidate list for follow-up Bayesian halo model comparison.


## 14. Recommended Scope

A practical first paper should focus on:

**Can MaNGA inner rotation-curve shapes, combined with baryonic decomposition, identify distinct central dynamical concentration regimes, and are those regimes linked to dark matter dominance or halo-model mismatch?**

The initial version should not promise definitive dark matter core/cusp measurement. It should instead deliver a reliable classification and candidate-selection framework.


## 15. One-Sentence Summary

This project uses MaNGA inner rotation-curve shapes and learned galaxy representations to classify central dynamical concentration types, then connects those types to baryon-dominated, dark-matter-supported, diffuse, or model-mismatched inner dynamical regimes through Bayesian mass decomposition.
