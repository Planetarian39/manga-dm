---
title: Sampling and Diagnostics
description: HMC, NUTS, PyMC, convergence checks, and the MaNGA NFW application.
---

# Sampling and diagnostics

## NUTS and PyMC

### NUTS

Random-walk Metropolis-Hastings can be inefficient when parameters are
numerous or strongly correlated. Hamiltonian Monte Carlo (HMC) uses gradients
of the log posterior to propose long trajectories through parameter space.
Instead of repeatedly taking short random steps, it introduces an auxiliary
momentum and numerically follows Hamiltonian dynamics.

The No-U-Turn Sampler (NUTS) adapts the HMC trajectory length. It stops
expanding a trajectory when continuing would begin to double back, and it
adapts step size during warmup. This removes two important hand-tuning
decisions while preserving HMC's gradient-informed exploration.

The physical analogy is a particle moving over the potential-energy surface
$-\log p(\theta \mid D)$. Momentum carries it across a high-probability region;
a Metropolis correction accounts for numerical integration error. The analogy
is useful, but the implementation still requires a continuous,
differentiable target and a parameterization with manageable curvature.

### A minimal PyMC example

This self-contained linear model uses a fixed seed for both simulated data and
sampling:

```python
import arviz as az
import numpy as np
import pymc as pm

rng = np.random.default_rng(42)
x = np.linspace(0, 10, 50)
y_obs = 2.5 * x + 1.0 + rng.normal(0, 1, x.size)

with pm.Model() as model:
    slope = pm.Normal("slope", mu=0, sigma=10)
    intercept = pm.Normal("intercept", mu=0, sigma=10)
    sigma = pm.HalfNormal("sigma", sigma=1)

    mu = slope * x + intercept
    pm.Normal("y", mu=mu, sigma=sigma, observed=y_obs)

    inference_data = pm.sample(
        draws=1000,
        tune=500,
        chains=4,
        random_seed=42,
        return_inferencedata=True,
    )

print(az.summary(inference_data, var_names=["slope", "intercept", "sigma"]))
```

PyMC selects NUTS for compatible continuous variables. The stored
`InferenceData` contains posterior draws and sampler statistics used by
ArviZ diagnostics.

## Practical guidelines

### Warmup and discarded draws

“Burn-in” historically describes early states discarded before a chain
reaches its stationary regime. For HMC and NUTS, **warmup** is more precise:
PyMC's `tune` draws are used to adapt step size, mass matrix, and related
sampler state, and are discarded by default. Increasing warmup can help an
adaptation problem, but it cannot repair a misspecified model or pathological
parameterization.

### Convergence diagnostics

No single diagnostic proves convergence. Use them together:

| Diagnostic | What it checks | Desired behavior |
|---|---|---|
| Rank-normalized split $\hat R$ | Agreement across split chains | Close to 1; project gates use an explicit threshold |
| Bulk ESS | Information for central posterior summaries | Large enough for stable location/scale estimates |
| Tail ESS | Information for interval endpoints and tails | Large enough for the reported credible intervals |
| Trace plot | Mixing, drift, sticking, chain disagreement | Stationary-looking, overlapping chains |
| Divergences | Numerical failure in high-curvature regions | None after warmup |
| Tree depth and energy diagnostics | HMC exploration and adaptation | No persistent saturation or pathological energy behavior |

The familiar “fuzzy caterpillar” trace is a visual shorthand, not a formal
test. Always connect diagnostics to the estimand: stable medians do not
guarantee stable extreme-tail probabilities.

### Choosing priors

- Use weakly informative priors that encode scale and physical support.
- Inspect prior predictive draws before fitting.
- Avoid extremely diffuse priors that put mass on nonsensical regions.
- Avoid unjustifiably narrow priors that suppress information in the data.
- Repeat the analysis with defensible alternatives when conclusions may be
  prior-sensitive.

See [Priors, data, and posterior updating](./priors-and-data.md) for the
analytical example.

### Common sampling problems

| Symptom | Possible cause | First response |
|---|---|---|
| $\hat R$ materially above 1 | Poor mixing, separated modes, too little warmup | Inspect chains, reparameterize, and run longer only after understanding the geometry |
| Low bulk or tail ESS | Autocorrelation, funnels, strong correlations | Reparameterize, improve priors, then increase draws |
| Divergences | High curvature or a problematic boundary | Inspect divergent regions and reparameterize; a higher `target_accept` is a secondary mitigation, not a universal fix |
| Maximum tree depth | Long difficult trajectories | Inspect scaling/correlations and adaptation |
| Chains agree but predictions fail | Model misspecification | Revise the likelihood or scientific model |

## Workflow

1. Define the likelihood, parameters, and priors.
2. Inspect prior predictive behavior.
3. Run multiple NUTS chains with warmup.
4. Inspect R-hat, bulk/tail ESS, divergences, traces, and energy behavior.
5. Run posterior predictive checks and scientific plausibility checks.
6. Report posterior summaries together with the interval definition and
   diagnostic evidence.
7. Preserve full samples when a downstream model needs their geometry.

MCMC turns integration into a sampling problem. It does not remove the need to
validate the probability model or the numerical exploration.

## MaNGA dark-matter halo inference

This section preserves the project-specific teaching material from the
original guide. The [Methods section](/methods/single-galaxy-nfw) is the
canonical source for the manuscript-aligned method and any implementation-status
differences.

::: warning Manuscript and implementation profiles
The parameter roles and equations below explain the model structure. Public
reproduction must use the manuscript-aligned profile documented in Methods rather
than inferring settings from this background page or from current CLI
defaults.
:::

### Inference task

MaNGA integral-field spectroscopy measures an H-alpha line-of-sight velocity
for each spatial pixel, or **spaxel**. A two-dimensional velocity field
contains information about the galaxy's rotating stellar and dark-matter mass
components.

For an NFW halo, two key parameters are:

- $M_{200}$, the mass enclosed within the radius whose mean density is 200
  times the cosmological critical density;
- $c=R_{200}/r_s$, the concentration, where $r_s$ is the NFW scale radius.

The NFW density profile has an inner $r^{-1}$ cusp and an outer $r^{-3}$
decline. MaNGA observes only an inner portion of the halo, so different
combinations of $M_{200}$ and $c$ can produce similar velocities. That
non-linear degeneracy motivates full posterior inference.

### Dynamical model

The circular speed is decomposed into stellar and dark-matter gravity, with an
asymmetric-drift correction for gas pressure support:

$$
V_{\mathrm{rot}}^2(r)
= V_\star^2(r) + V_{\mathrm{dm}}^2(r) - V_{\mathrm{drift}}^2(r).
$$

The stellar term contains a Hernquist-like bulge and an exponential disk:

$$
V_\star^2(r)=V_{\mathrm{bulge}}^2(r)+V_{\mathrm{disk}}^2(r),
$$

$$
V_{\mathrm{bulge}}^2(r)
= \frac{G M_{\mathrm{bulge}}r}{(r+a)^2},
\qquad
M_{\mathrm{bulge}}=f_{\mathrm{bulge}}M_\star,
\qquad
a=\frac{R_e}{1.8153},
$$

$$
V_{\mathrm{disk}}^2(r)
= \frac{2GM_{\mathrm{disk}}}{R_d}
   y^2\!\left[I_0(y)K_0(y)-I_1(y)K_1(y)\right],
\quad
M_{\mathrm{disk}}=(1-f_{\mathrm{bulge}})M_\star,
\quad
y=\frac{r}{2R_d},
\quad
R_d=\frac{R_e}{1.678}.
$$

$I_n$ and $K_n$ are modified Bessel functions. The NFW contribution is

$$
V_{\mathrm{dm}}^2(r)
= \frac{V_{200}^2}{x}
  \frac{\ln(1+cx)-cx/(1+cx)}
       {\ln(1+c)-c/(1+c)},
\qquad x=\frac{r}{R_{200}},
$$

with

$$
V_{200}=\left(10GH(z)M_{200}\right)^{1/3},
\qquad
H(z)=H_0\sqrt{\Omega_m(1+z)^3+\Omega_\Lambda}.
$$

The gas-pressure term used by this model family is

$$
V_{\mathrm{drift}}^2(r)=2\sigma_0^2\frac{r}{R_d}.
$$

Projection converts the intrinsic speed to a model prediction for the
line-of-sight velocity:

$$
V_{\mathrm{los,model}}(r,\phi)
= V_{\mathrm{sys}}
  + V_{\mathrm{rot}}(r)\sin i\cos\phi.
$$

Writing “model” explicitly distinguishes this prediction from the measured
$V_{\mathrm{los,data}}$.

### Parameters and priors

The teaching model contains stellar mass, halo mass, concentration, gas
dispersion, systemic velocity, effective radius, bulge fraction, intrinsic
scatter, and Student-t degrees of freedom. Photometric quantities anchor the
stellar structure; a stellar-to-halo mass relation stabilizes the weakly
identified halo-mass direction; the concentration prior remains wider so the
rotation-curve shape can update it.

| Parameter | Prior role in the teaching model |
|---|---|
| $\log_{10}M_\star$ | Normal prior anchored to photometric stellar mass |
| $\log_{10}M_{200}$ | Truncated normal centered on a stellar-to-halo mass estimate |
| $\log_{10}c$ | Broad normal prior, independent of $M_{200}$ in the single-galaxy fit |
| $\log\sigma_0$ | Lognormal scale prior for pressure support |
| $V_{\mathrm{sys}}$ | Truncated normal around the measured systemic velocity |
| $\log R_e$ | Lognormal prior anchored to photometry |
| $\operatorname{logit}(f_{\mathrm{bulge}})$ | Normal prior informed by Sersic index |
| $\sigma_{\mathrm{int}}$ | Exponential prior for additional residual scatter |
| $\nu-2$ | Gamma prior, enforcing $\nu>2$ |

For every lognormal, Gamma, or Exponential distribution, the Methods page
states the sampled variable, log base, units, and rate-versus-scale convention.
Those details are essential because shorthand such as
`Gamma(2, 0.1)` is otherwise ambiguous.

The prior-design lesson is structural: tightening both $M_{200}$ and $c$ would
impose the answer, while making both extremely broad can let the chain wander
along a weakly identified ridge. Sensitivity checks are needed to determine
what the data update.

### Likelihood

Measurement error and residual model scatter enter a robust Student-t
likelihood:

$$
V_{\mathrm{los,data},i}
\sim
\operatorname{StudentT}\!\left(
  \nu,\,
  \mu=V_{\mathrm{los,model},i},\,
  \sigma=\sqrt{\sigma_{\mathrm{meas},i}^2+\sigma_{\mathrm{int}}^2}
\right).
$$

The Student-t tails reduce the leverage of spaxels affected by bars,
non-circular motion, or other outliers. As $\nu$ increases, the distribution
approaches a normal distribution.

The model family also supports smooth radial weighting:

$$
w(r)
= w_{\min}
  + \frac{1-w_{\min}}
         {1+\exp[-(r-r_0)/\delta r]}.
$$

Applying the weight through a PyMC `Potential` changes the total
log-likelihood to

$$
\log\mathcal L_{\mathrm{weighted}}
= \sum_i w_i
  \log p\!\left(
    V_{\mathrm{los,data},i}\mid\boldsymbol{\Theta}
  \right).
$$

This can reduce the influence of inner spaxels affected by beam smearing, but
the manuscript-aligned values and rationale must be read from the canonical Methods
page.

### Abridged PyMC implementation

The following block is deliberately abridged. Helper functions and arrays are
defined in `src.models.dm_nfw.DmNfw`; this is a map of probability-model
structure, not a copy-paste replacement for repository code.

```python
with pm.Model() as model:
    log10_Mstar = pm.Normal(
        "log10_Mstar", mu=log10_Mstar_photometry, sigma=mstar_prior_dex
    )
    Mstar = pm.Deterministic("Mstar", 10.0**log10_Mstar)

    log10_M200 = pm.TruncatedNormal(
        "log10_M200",
        mu=log10_M200_shmr,
        sigma=m200_prior_dex,
        lower=m200_lower,
        upper=m200_upper,
    )
    M200 = pm.Deterministic("M200", 10.0**log10_M200)

    log_c = pm.Normal("log_c", mu=np.log(c_reference), sigma=c_prior_ln)
    c = pm.Deterministic("c", pm.math.exp(log_c))

    sigma_int = pm.Exponential("sigma_int", lam=sigma_int_rate)
    nu = pm.Deterministic(
        "nu", pm.Gamma("nu_minus_two", alpha=2.0, beta=0.1) + 2.0
    )

    v_model = project_velocity_field(
        radius, azimuth, Mstar, M200, c, other_parameters
    )
    sigma_obs = pm.math.sqrt(sigma_meas**2 + sigma_int**2)
    point_logp = pm.logp(
        pm.StudentT.dist(nu=nu, mu=v_model, sigma=sigma_obs),
        velocity_data,
    )
    pm.Potential("weighted_likelihood", pm.math.sum(weight * point_logp))

    inference_data = pm.sample(
        draws=draws,
        tune=tune,
        chains=chains,
        target_accept=target_accept,
        random_seed=random_seed,
        return_inferencedata=True,
    )
```

Modified Bessel functions in the disk term must remain differentiable through
the selected PyTensor/backend path. Backend support and package versions are
part of the reproducibility profile.

### Posterior diagnostics and quality gates

Each single-galaxy fit is evaluated with sampling and predictive diagnostics:

| Diagnostic | What failure indicates |
|---|---|
| Rank-normalized $\hat R$ | Chains have not reached consistent distributions |
| Bulk and tail ESS | Too few equivalent independent draws for reported summaries |
| Reduced $\chi^2$ | Median model residuals are too large relative to the error model |
| Posterior predictive $p$-value | Replicated data are inconsistent with the observation |
| $|\rho(c,M_{200})|$ | Halo parameters remain excessively degenerate for the downstream use |

The manuscript-aligned quality preset, including exact thresholds, belongs on
[Diagnostics and quality gates](/methods/diagnostics-and-quality-gates).
This background page intentionally reports no aggregate pass counts.

### Why full posterior samples are needed

Within the observed radial range, the single-galaxy joint posterior can form a
curved $M_{200}$-$c$ ridge. A mean and covariance cannot necessarily preserve
that geometry. Stage 2 therefore works with Stage 1 samples.

There is an important correction to the original guide: a Stage 1 posterior
cannot be inserted as if it were a likelihood. Let
$q_i(\theta_i)=p(\theta_i\mid D_i)$ be the single-galaxy posterior,
$\pi_i(\theta_i)$ its Stage 1 prior, and $p(\theta_i\mid\eta)$ the population
model. Up to factors independent of the population parameters $\eta$,

$$
p(\{D_i\}\mid\eta)
\propto
\prod_i
\int
\frac{q_i(\theta_i)}{\pi_i(\theta_i)}
p(\theta_i\mid\eta)\,d\theta_i.
$$

For draws $\theta_i^{(k)}\sim q_i$,

$$
p(\{D_i\}\mid\eta)
\propto
\prod_i
\left[
\frac{1}{K_i}
\sum_{k=1}^{K_i}
\frac{p(\theta_i^{(k)}\mid\eta)}
     {\pi_i(\theta_i^{(k)})}
\right].
$$

Dividing by $\pi_i$ removes the prior already used in Stage 1 and avoids
double-counting it. Numerical implementations evaluate the expression in log
space and must check importance-weight stability.

See [11743-9102](/case-studies/11743-9102#joint-posterior-geometry) for a
single-galaxy visualization of the curved posterior geometry.
