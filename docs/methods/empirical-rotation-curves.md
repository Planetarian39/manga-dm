---
title: Empirical Rotation Curves
description: Robust Bayesian screening of disk-like velocity fields before physical mass decomposition.
---

# Empirical rotation-curve screening

The empirical stage asks a deliberately narrow question: can a compact disk-rotation model describe the usable velocity field with a stable posterior and adequate predictive coverage? It does not infer a stellar or dark-matter mass decomposition.

::: info What this page demonstrates
- **Scientific purpose:** use a robust low-dimensional model as a quality screen before physical mass decomposition.
- **Key modeling decision:** combine a tanh-plus-linear rotation curve with a Student-t observation model and predictive checks.
- **My implementation contribution:** I integrated Bayesian empirical fitting, posterior-predictive evaluation, and the handoff to the physical model.
- **Main limitation:** passing this screen establishes model adequacy for the observed field, not a unique physical interpretation.
:::

## Rotation-curve parameterization

The intrinsic empirical curve is

$$
V_{\mathrm{rot}}(r)
=V_c\tanh\!\left(\frac{r}{R_t}\right)
+s_{\mathrm{out}}r.
$$

- $V_c$ controls the characteristic velocity scale.
- $R_t$ controls the inner turnover.
- $s_{\mathrm{out}}$ permits a residual outer slope.

After geometric projection,

$$
V_{\mathrm{obs,model}}(r,\phi)
=V_{\mathrm{sys}}
+V_{\mathrm{rot}}(r)\sin i\cos(\phi-\phi_0).
$$

The fitted vector contains $V_c$, $R_t$, $s_{\mathrm{out}}$, $V_{\mathrm{sys}}$, geometric offsets in inclination and position angle, an intrinsic-scatter term $\sigma_{\mathrm{int}}$, and Student-t degrees of freedom $\nu$.

## Robust observation model

Each retained spaxel contributes a Student-t likelihood,

$$
V_{\mathrm{obs}}
\sim \operatorname{StudentT}\!\left(
\nu,
\mu=V_{\mathrm{obs,model}},
\sigma=\sqrt{\sigma_{\mathrm{meas}}^2+\sigma_{\mathrm{int}}^2}
\right),
$$

with

$$
\sigma_{\mathrm{meas}}=\frac{1}{\sqrt{\mathrm{IVAR}}}.
$$

The heavy-tailed likelihood reduces the leverage of isolated disturbed or poorly modeled spaxels while preserving sensitivity to the coherent rotation pattern. Posterior sampling uses NUTS in PyMC 5. The current manuscript specifies the sampler family here but does not define a separate empirical-stage draw count; current implementation settings should therefore be labeled as implementation behavior rather than silently promoted to manuscript parameters.

## Posterior-predictive screening

The screen evaluates two complementary quantities at predictive probability `0.9545`:

- **Value coverage**, $f_{\mathrm{HDI}}$: the fraction of observed velocities lying inside the posterior-predictive interval.
- **Measurement overlap**, $g_{\mathrm{HDI}}$: the fraction of spaxels for which the measurement-uncertainty interval intersects the posterior-predictive interval.

The manuscript thresholds are

$$
f_{\mathrm{HDI}}>0.60,
\qquad
g_{\mathrm{HDI}}>0.80.
$$

Predictive adequacy is combined with inclination, spaxel-count, radial-extent, $\hat R$, and ESS requirements from [Data and selection](./data-and-selection.md). No single residual statistic determines acceptance.

## What the screen establishes

A passing fit establishes only that the velocity field supports stable inference under this empirical disk-like model over the observed radial range. It does not establish that:

- all motions are circular;
- the galaxy is free of bars, warps, or local asymmetries;
- a particular halo profile is preferred;
- the later mass decomposition is uniquely determined.

This distinction matters because the empirical model is a quality gate, not a physical interpretation layer.

<MethodStatus status="paper">

The manuscript screen uses spaxels within 60° of the major axis,
predictive probability `0.9545`, R-hat ≤ 1.05, and ESS ≥ 200 for
empirical retention.

</MethodStatus>

<MethodStatus status="implementation">

Current fallbacks are 45° and `0.95`. `RotCurve.evaluate_fit_quality`
does not include R-hat or ESS in its pass boolean, named presets are not
the manuscript equation, and Stage 2 defaults to GMM inputs.

</MethodStatus>

## Code map

- Input preparation and masks: `src/models/rotation_curve.py`
- Selection orchestration: `src/pipeline/selection.py`
- Active thresholds: `src/config/settings.py`
- Shared interval utilities: `src/stats/intervals.py`

Next: [Single-galaxy NFW inference](./single-galaxy-nfw.md).
