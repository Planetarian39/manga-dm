---
title: Priors, Data, and Posterior Updating
description: A Beta-binomial example showing how prior information and data combine.
---

# Priors, data, and posterior updating

## How priors and data interact

Bayesian updating is often summarized as

$$
\underbrace{p(\theta \mid D)}_{\text{posterior}}
\propto
\underbrace{p(D \mid \theta)}_{\text{likelihood}}
\times
\underbrace{p(\theta)}_{\text{prior}}.
$$

The posterior is not a mechanical compromise with a fixed weighting. Its
balance depends on the amount and precision of the data, the strength of the
prior, parameter identifiability, and whether the model can describe the
observations.

As a useful first approximation, more informative observations usually reduce
the influence of reasonable alternative priors:

| Data information | Typical posterior behavior | Differences across priors |
|---|---|---|
| Very weak | Prior structure can be prominent | Often large |
| Moderate | Prior and likelihood both matter | Visible but reduced |
| Strong and identifiable | Likelihood often dominates locally | Often small |

The qualification “identifiable” is essential. A large data set can still
leave a parameter prior-sensitive if the measured quantity barely changes
with that parameter, or if another parameter can compensate for it.

## Coin-bias example

Suppose a coin has an unknown heads probability $\mu$. After $n$ flips with
$h$ heads, use a Beta prior:

$$
p(\text{heads} \mid \mu)=\mu,
\qquad
p(\mu)=\operatorname{Beta}(\alpha_0,\beta_0).
$$

The Beta distribution is defined on $[0,1]$. Its mean is
$\alpha/(\alpha+\beta)$. It is conjugate to a binomial likelihood, so the
posterior remains in the same family:

$$
p(\mu \mid n,h)
= \operatorname{Beta}\!\left(
    \alpha_0+h,\,
    \beta_0+n-h
  \right).
$$

The posterior mean is

$$
\bar{\mu}_{\mathrm{post}}
= \frac{\alpha_0+h}{\alpha_0+\beta_0+n}.
$$

This analytical example does not require MCMC. That is why it is useful:
prior influence can be studied without mixing it up with sampling error.

The figures use a simulated true value $\mu_{\mathrm{true}}=0.7$ and three
deliberately different priors:

- **Prior A:** $\operatorname{Beta}(2,2)$, a weak symmetric prior with mean
  $0.50$.
- **Prior B:** $\operatorname{Beta}(1,8)$, concentrated toward small values,
  with mean about $0.11$.
- **Prior C:** $\operatorname{Beta}(8,1)$, concentrated toward large values,
  with mean about $0.89$.

The same base seed is used for every maintained figure. Each figure derives an
independent random stream, so regenerating one figure cannot change another.

## Posterior evolution with sample size

![Six panels show that three different Beta priors produce distinct posteriors for small samples and increasingly similar posteriors as the sample grows.](/assets/mcmc/posterior-by-sample-size.png)

*Posterior distributions for $n=3$, $10$, $30$, $100$, $300$, and $1000$.
Color and line style both identify the three priors; the dotted vertical line
marks $\mu_{\mathrm{true}}=0.7$.*

At very small $n$, the priors retain visibly different shapes and peak
locations. By $n=30$ the likelihood has shifted all three distributions toward
the observed fraction of heads, but the prior choices still matter. At
$n=100$ and beyond, the distributions become narrow and increasingly
similar. The exact curves remain conditional on this simulated sequence; the
general lesson is the reduction in prior sensitivity, not a special threshold
at one value of $n$.

## Posterior mean as data accumulate

![Three posterior-mean paths begin near their prior means and move toward the true heads probability as observations accumulate.](/assets/mcmc/posterior-mean-by-sample-size.png)

*Posterior mean under the three priors as $n$ increases from 1 to 200. The
paths fluctuate because each new flip changes the observed heads fraction.*

The initially separated paths move toward one another as the common data
sequence grows. This is **posterior updating with more data**, not Markov-chain
convergence. Chain convergence asks whether a sampling algorithm has explored
a fixed posterior; the figure changes the posterior itself at every step.

## Small and large data

![Dashed low-opacity priors and marked posterior curves differ strongly at n equals 2, while the three posteriors cluster near 0.7 at n equals 200.](/assets/mcmc/prior-posterior-small-large-data.png)

*Prior and posterior densities for a small and a larger simulated data set.
Line style identifies the prior family; low-opacity lines are priors and
thicker marked lines are posteriors.*

With $n=2$, the posterior is only a modest update of each prior, so the three
answers remain different. With $n=200$, the three posterior densities are
concentrated in a similar region even though their prior densities remain
very different.

## Practical prior guidance

### When data are weak

A strong but poorly justified prior can dominate a weak likelihood. Prefer a
prior that:

- encodes known physical bounds and plausible scales;
- rules out numerical pathologies;
- is wide enough to let the measured quantity update the parameter;
- is checked with prior predictive simulation.

“Weakly informative” does not mean flat over an enormous range. Extremely
diffuse priors can place most probability on implausible values and create
geometry that is difficult for a sampler.

### When data are strong

Reasonable priors may produce similar posteriors, but the prior still:

- defines support and regularizes poorly identified directions;
- influences extrapolation beyond the observed range;
- changes the marginal likelihood;
- can improve or degrade sampling geometry.

A Gaussian prior corresponds to an L2 penalty only in a MAP or penalized
optimization objective. That equivalence does not apply to every prior and
does not reduce full Bayesian inference to regularized optimization.

### Sensitivity is an empirical question

Compare more than one defensible prior, examine prior and posterior predictive
distributions, and report parameters that remain prior-sensitive. The
[11743-9102 case study](/case-studies/11743-9102#prior-to-posterior-update)
shows the comparison for a real single-galaxy fit without duplicating its
case-specific values here.

## Reproduce the figures

From the repository root:

```text
python docs/scripts/mcmc/gen_prior_figures.py \
  --output-dir docs/public/assets/mcmc \
  --seed 42
```

The script uses NumPy, SciPy, and Matplotlib, writes no files at import time,
and gives each figure its own deterministic random stream.
