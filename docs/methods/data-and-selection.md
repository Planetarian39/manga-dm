---
title: Data and Selection
description: MaNGA inputs, projected geometry, spaxel screening, and the paper-aligned selection rules.
---

# Data products and sample selection

The pipeline combines spatially resolved MaNGA emission-line measurements with catalog-level photometric and structural quantities. The velocity-field model uses the MaNGA DR17 Data Analysis Pipeline (DAP) MAPS products, while galaxy-level anchors come from DRPALL.

## Observational inputs

| Quantity | Source | Role |
|---|---|---|
| $V_{\mathrm{obs}}$ | MAPS | H-alpha line-of-sight velocity field |
| IVAR | MAPS | Formal velocity uncertainty, $\sigma_{\mathrm{meas}}=1/\sqrt{\mathrm{IVAR}}$ |
| SNR | MAPS | Spaxel-quality screening |
| $r,\phi$ | MAPS | Elliptical polar coordinates; radius is converted to physical kpc |
| Position angle and $b/a$ | MAPS | Major-axis geometry and photometric inclination estimate |
| $\sigma_{\mathrm{gas}}$ | MAPS | Gas-dispersion quality information and pressure-support context |
| $R_e$ | DRPALL | Structural prior scale for disk and bulge components |
| $M_\star$ | DRPALL | Stellar-mass prior anchor |
| Sersic $n$ | DRPALL | Morphology proxy for the bulge-fraction prior |
| $z$ | DRPALL | Angular-distance conversion and $H(z)$ |

The public project does not redistribute raw MAPS FITS products. Data acquisition and I/O live under `src/data/`; path resolution goes through `src/config/settings.py`.

## Projected velocity geometry

For an axisymmetric thin disk, the line-of-sight velocity model is

$$
V_{\mathrm{obs}}(r,\phi)
=V_{\mathrm{sys}}
+V_{\mathrm{rot}}(r)\sin i\cos(\phi-\phi_0),
$$

where $V_{\mathrm{sys}}$ is the systemic velocity, $i$ is inclination, and $\phi_0$ is the projected major-axis position angle. The geometric projection is shared by the empirical screen and the physical mass model.

## Spaxel-level screen

The finalized method retains a velocity spaxel when the required values are finite and:

- H-alpha velocity SNR is at least 10;
- its azimuthal offset from the major axis is at most $60^\circ$;
- its inverse variance is positive.

The major-axis restriction avoids giving minor-axis spaxels, where the rotational projection is weak, disproportionate influence over the inferred intrinsic velocity. The code that constructs the current mask is `RotCurve._build_vel_quality_mask` in `src/models/rotation_curve.py`.

## Galaxy-level empirical screen

A galaxy proceeds from empirical fitting only when all paper gates are satisfied:

$$
\left\{
\begin{aligned}
25^\circ &\le i \le 70^\circ,\\
N_{\mathrm{valid}} &\ge 150,\\
R_{\mathrm{out}}/R_t &\ge 2,\\
\hat R &\le 1.05,\\
\mathrm{ESS} &\ge 200,\\
f_{\mathrm{HDI}} &> 0.60,\\
g_{\mathrm{HDI}} &> 0.80.
\end{aligned}
\right.
$$

Here $R_t$ is the empirical turnover radius, $R_{\mathrm{out}}$ is the outer usable radius, $f_{\mathrm{HDI}}$ is the fraction of observed velocities inside the posterior-predictive interval, and $g_{\mathrm{HDI}}$ is the fraction whose measurement interval overlaps that predictive interval. The interval probability is `0.9545`; paper prose describes it as 95%.

These rules define a kinematically selected analysis population. They should be published without reporting how many galaxies pass or fail any stage.

<MethodStatus status="paper">

The paper-aligned azimuthal limit is 60°, and the exact predictive
probability is `0.9545`. The paper quality equation enforces its stated
diagnostic gates before posterior samples enter Stage 2.

</MethodStatus>

<MethodStatus status="implementation">

`src/config/settings.py` falls back to `PHI_DEG_THRESHOLD = 45.0` and
`HDI_PROB2 = 0.95`. Current masks add IVAR-percentile and optional
gas-dispersion checks, while empirical R-hat and ESS are warnings rather
than enforced pass conditions. Named presets and the default GMM Stage 2 path
do not reproduce the paper configuration.

</MethodStatus>

## Reproducibility notes

- Keep angles in radians inside model code and report thresholds in degrees.
- Record the interval probability alongside coverage values; “95%” alone is not enough to distinguish `0.95` from `0.9545`.
- Record all active mask settings because a valid-spaxel count is configuration-dependent.
- Treat selection as part of the statistical model when interpreting any later population fit.

Next: [Empirical rotation curves](./empirical-rotation-curves.md).
