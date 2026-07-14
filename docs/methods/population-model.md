---
title: Population Model
description: Factored Student-t concentration-mass model and prior-corrected posterior-sample likelihood.
---

# Population model and prior-corrected likelihood

The population stage treats each accepted single-galaxy posterior as a sampled representation of that galaxy's likelihood geometry. It does not replace a curved $M_{200}$-$c$ posterior with a median, covariance matrix, or Gaussian approximation.

## Concentration-mass parameterization

$$
\log_{10}c
=\log_{10}c_0
+\alpha\log_{10}\!\left(\frac{M_{200}}{M_{\mathrm{pivot}}}\right).
$$

The pivot convention is

$$
M_{\mathrm{pivot}}=\frac{M_{\mathrm{pivot},h^{-1}}}{h},
\qquad
M_{\mathrm{pivot},h^{-1}}=10^{12}h^{-1}M_\odot.
$$

For galaxy $i$,

$$
m_i=\log_{10}M_{200,i}-\log_{10}M_{\mathrm{pivot}},
\qquad
\ell_i=\log_{10}c_i.
$$

## Factored Student-t population

$$
m_i\sim t_\nu(\mu_M,\sigma_M^2),
\qquad
\ell_i\mid m_i
\sim t_\nu\!\left(\log_{10}c_0+\alpha m_i,\sigma_{\mathrm{int}}^2\right).
$$

The global parameter vector is

$$
\Phi=\{\mu_M,\sigma_M,c_0,\alpha,\sigma_{\mathrm{int}},\nu\}.
$$

The factors share $\nu$, allowing the selected halo-mass distribution and deviations about the mean relation to have heavier tails than a Normal model.

## Population priors

| Parameter | Finalized prior | Role |
|---|---|---|
| $\mu_M$ | $\mathcal N(\bar m_{\mathrm{obs}},1.0^2)$ | Broad center for the selected mass distribution |
| $\sigma_M$ | $\mathcal{HN}(1.0)$ | Positive mass-distribution width |
| $\log_{10}c_0$ | $\mathcal N(0.905,0.5^2)$ | Broad normalization prior |
| $\alpha$ | $\mathcal N(-0.101,0.3^2)$ | Broad slope prior |
| $\ln\sigma_{\mathrm{int}}$ | $\mathcal N(\ln0.15,0.8^2)$ | Positive intrinsic-width parameterization |
| $\nu$ | $\Gamma(2.0,0.1)$ | Shared tail parameter; second value is a rate, giving prior mean about 20 |

## Posterior samples as an importance proposal

For data $D_i$ and latent parameters $\boldsymbol\theta_i=(m_i,\ell_i)$,

$$
\mathcal L_i(\Phi)=
\int p(D_i\mid\boldsymbol\theta_i)
p_{\mathrm{pop}}(\boldsymbol\theta_i\mid\Phi)
\,d\boldsymbol\theta_i.
$$

Because

$$
p(\boldsymbol\theta_i\mid D_i)
\propto
p(D_i\mid\boldsymbol\theta_i)
p_{\mathrm{stage1}}(\boldsymbol\theta_i),
$$

the stored posterior can be reused as a proposal. The raw log weight of draw $s$ is

$$
\log w_{is}
=\log p_{\mathrm{pop}}(\boldsymbol\theta_{is}\mid\Phi)
-\log p_{\mathrm{stage1}}(\boldsymbol\theta_{is}).
$$

The denominator removes the single-galaxy prior before the candidate population density is applied. Omitting it would count the Stage 1 prior as new population evidence.

## Truncated estimator

Extreme weights can destabilize a finite-sample estimate. The finalized method caps weights at a galaxy-specific threshold $\tau_i$:

$$
\log\mathcal L_i(\Phi)
\approx
\operatorname{logsumexp}_s
\left[\min(\log w_{is},\log\tau_i)\right]
-\log S_i.
$$

$S_i$ is the number of retained draws. Truncation controls weight variance while introducing bias of order $S_i^{-1/2}$. Pareto $\hat k$ and importance-sampling ESS diagnose support mismatch.

## Conditioning

This density is conditional on every upstream selection and quality gate. It describes the retained kinematic population under the adopted NFW-equivalent model. This page intentionally reports no fitted hyperparameter values, aggregate plots, comparisons, or physical conclusions.

<MethodStatus status="paper">

The finalized Stage 2 method is the prior-corrected posterior-sample
likelihood above, using only galaxies that pass the finalized upstream gates.

</MethodStatus>

<MethodStatus status="implementation">

`fit_m200_c_population` defaults to `use_gmm=True` and `use_samples=False`.
The sample-aware implementation exists, but an ordinary `manga stage2 --fit
--quality-cut recommended` run is not the paper path, and upstream profile
differences remain.

</MethodStatus>

## Code map

- Population orchestration: `src/pipeline/population.py`
- Population and likelihood modes: `src/models/population.py`
- Posterior loading and prior metadata: `src/data/results.py`
- PSIS diagnostics: `src/stats/psis.py`

Next: [Diagnostics and quality gates](./diagnostics-and-quality-gates.md).
