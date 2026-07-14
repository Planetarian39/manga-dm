---
title: Optimization and Posterior Sampling
description: How point estimation and MCMC answer different inference questions.
---

# Optimization and posterior sampling

Optimization and posterior sampling are complementary tools. The right choice
depends on the estimand, posterior geometry, accuracy requirement, and
available compute.

## Different inference targets

Maximum likelihood estimation returns a parameter value that maximizes the
likelihood:

$$
\hat{\theta}_{\mathrm{MLE}}
= \arg\max_\theta p(D\mid\theta).
$$

Maximum a posteriori estimation includes the prior:

$$
\hat{\theta}_{\mathrm{MAP}}
= \arg\max_\theta p(D\mid\theta)p(\theta).
$$

MCMC targets the distribution rather than only its mode:

$$
\{\theta^{(1)},\theta^{(2)},\ldots,\theta^{(K)}\}
\sim p(\theta\mid D).
$$

The samples support medians, credible intervals, correlations, posterior
predictive draws, and downstream uncertainty propagation. An optimizer can be
the right tool when the mode itself is the target or when an approximation
around the mode is demonstrably adequate.

## Comparison

| Property | Optimization | MCMC |
|---|---|---|
| Primary output | One or more optima | Correlated posterior draws |
| Prior support | MAP includes a prior; MLE does not | Included in the target posterior |
| Uncertainty | Requires Hessian/Laplace, profile likelihood, bootstrap, or another approximation | Estimated from draws, subject to Monte Carlo and exploration error |
| Non-Gaussian geometry | Can be explored with profiling or repeated optimization, but is not in one point | Represented when the chain explores it successfully |
| Compute | Usually cheaper | Usually more expensive |
| Diagnostics | Gradient, termination, curvature, restart stability | R-hat, ESS, divergences, trace/energy checks, predictive checks |
| Downstream propagation | Requires an uncertainty representation | Samples can be transformed or reweighted directly |

Neither column guarantees a correct scientific model. Both depend on the
likelihood, parameterization, and data.

## Parameter degeneracy

Degeneracy occurs when different parameter combinations produce similar
predictions. In an NFW rotation curve observed at $r\ll R_{200}$,

$$
V_{\mathrm{dm}}^2(r)\approx f(M_{200},c,r),
$$

and increasing $M_{200}$ while decreasing $c$ can leave the predicted velocity
nearly unchanged.

An optimizer returns a point on, or near, this ridge. Multiple starts, profile
likelihoods, and curvature checks can reveal the direction, but the optimum
alone does not show how posterior mass is distributed along it. Successful
MCMC sampling places draws across the ridge, making curved and skewed
structure visible.

That geometry is why `manga-dm` preserves single-galaxy samples for the
population stage. Inspect the
[11743-9102 joint posterior](/case-studies/11743-9102#joint-posterior-geometry)
for the concrete example.

## Uncertainty from curvature and samples

A common optimization-based approximation uses the inverse Fisher information
or inverse Hessian:

$$
\operatorname{Cov}(\hat\theta)
\approx \mathcal I(\hat\theta)^{-1},
\qquad
\mathcal I_{ij}(\theta)
=-\mathbb E\!\left[
\frac{\partial^2\log p(D\mid\theta)}
     {\partial\theta_i\,\partial\theta_j}
\right].
$$

The Laplace approximation treats the log posterior as locally quadratic. It
can be accurate for a near-Gaussian, well-identified posterior and misleading
for curved ridges, boundaries, heavy tails, or multiple modes.

MCMC estimates quantiles from samples without imposing a Gaussian shape:

```python
import numpy as np

samples = np.asarray(
    inference_data.posterior["M200"].stack(sample=("chain", "draw"))
)
samples = samples[np.isfinite(samples)]
q16, q50, q84 = np.percentile(samples, [15.865, 50.0, 84.135])
```

These are Monte Carlo estimates, not exact intervals. Their reliability
depends on effective sample size, exploration, and the interval definition.
For a skewed posterior, an equal-tailed interval (ETI) and highest-density
interval (HDI) answer different questions and must not be mixed silently.

## Initialization and multimodality

An optimizer may converge to different local extrema from different starts:

```python
from scipy.optimize import minimize

# Conceptual example: neg_log_posterior is defined by the model.
result_a = minimize(neg_log_posterior, x0=[1e12, 5])
result_b = minimize(neg_log_posterior, x0=[1e13, 20])
```

Restart stability is therefore an important optimization diagnostic.

Multiple MCMC chains also do not guarantee global exploration. Widely
separated modes can trap all chains, especially if initialization places them
in the same basin. Initializing chains broadly, examining marginal and joint
plots, and using model-specific knowledge remain necessary.

## MAP estimation

MAP uses the same prior-times-likelihood target as Bayesian sampling but
returns only its mode. It can provide:

- a quick model and scale check;
- a useful point summary when the posterior is near Gaussian;
- a starting region for some algorithms after validation.

MAP is not invariant under parameter transformation: the MAP of $\theta$ does
not generally transform into the MAP of $g(\theta)$. Posterior draws transform
by applying $g$ to every draw.

Do not assume `pm.find_MAP()` is a universally good NUTS initializer. A mode
can lie near a boundary or in geometry that is poor for adaptation. Use
PyMC's supported initialization strategies and validate them for the model.

## Choosing an approach

| Scenario | Useful approach | Reason |
|---|---|---|
| Rapid scale or model check | Optimization | Low computational cost |
| Near-Gaussian posterior, mode and local covariance sufficient | MAP plus validated curvature approximation | The local approximation can be adequate |
| Strong non-linear degeneracy | MCMC or another validated distributional approximation | A point alone loses geometry |
| Credible intervals and predictive distributions required | MCMC | Direct sample-based summaries |
| Single-galaxy uncertainty passed to a population model | Prior-corrected posterior samples | Preserves non-Gaussian geometry |
| Very high dimensional model or strict latency budget | Optimization, Laplace, or variational approximation, validated against references | Full MCMC may be too costly |

A practical workflow may use optimization for model debugging and scale
checks, then posterior sampling for the reported inference. The two stages
must share a clearly defined probability model.

## Summary

> Optimization locates parameter values that optimize an objective. MCMC
> estimates how posterior probability is distributed across plausible
> parameter values.

The contrast is about outputs and approximation assumptions, not a claim that
one method always succeeds.
