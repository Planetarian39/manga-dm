---
title: MCMC and Bayesian Inference
description: Why manga-dm uses posterior samples and how to read the Bayesian background sequence.
---

# MCMC and Bayesian inference

This sequence introduces the Bayesian and Markov chain Monte Carlo concepts
used by `manga-dm`. It is written for readers who want to understand why the
pipeline keeps posterior samples instead of reporting only a best-fitting
point.

## Why use MCMC?

Scientific inference often asks:

> Given observations $D$, what parameter values $\theta$ remain plausible,
> and how uncertain are they?

Bayesian inference answers with the posterior distribution
$p(\theta \mid D)$. Bayes' theorem writes that distribution as

$$
p(\theta \mid D)
= \frac{p(D \mid \theta)\,p(\theta)}{p(D)}.
$$

The denominator is the marginal likelihood, also called the model evidence:

$$
Z = p(D) = \int p(D \mid \theta)\,p(\theta)\,d\theta.
$$

For realistic scientific models this integral is often analytically
intractable and expensive to approximate on a grid. MCMC takes a different
route. It constructs correlated draws from the posterior without requiring
the numerical value of $Z$. Given a well-explored chain, sample averages and
quantiles estimate posterior expectations, credible intervals, correlations,
and other summaries.

MCMC does not make a difficult model automatically trustworthy. The result
still depends on the likelihood, priors, parameterization, and whether the
sampler explored the relevant posterior regions. That is why this guide treats
diagnostics as part of inference rather than as an optional final plot.

## Learning path

1. [Bayesian foundations](./bayesian-foundations.md) derives Bayes' theorem,
   Monte Carlo estimates, Markov chains, and Metropolis-Hastings.
2. [Priors and data](./priors-and-data.md) uses a Beta-binomial coin example
   to show how prior information and observations combine.
3. [Sampling and diagnostics](./sampling-and-diagnostics.md) introduces HMC,
   NUTS, PyMC, convergence checks, and the MaNGA NFW application.
4. [Optimization and MCMC](./optimization-vs-mcmc.md) compares point
   estimation with sample-based uncertainty propagation.
5. [Project lessons](./project-lessons.md) records practical lessons from
   developing the inference workflow without presenting population results.

For a concrete output-reading exercise, see the
[11743-9102 posterior diagnostics](/case-studies/11743-9102#posterior-diagnostics).
That page is the single source for all values and figures specific to that
galaxy.

## What posterior samples make possible

A point estimate can locate a mode or optimum. Posterior samples add the
geometry around and beyond that point:

- marginal uncertainty for each parameter;
- non-linear correlations and degeneracy ridges;
- skewness, heavy tails, and multiple modes;
- posterior predictive checks;
- propagation of single-galaxy uncertainty into the population model.

The last item is especially important for `manga-dm`. A Stage 1 posterior is
not itself a likelihood. When it is reused in Stage 2, the Stage 1 prior must
be divided out. The [MaNGA application](./sampling-and-diagnostics.md#why-full-posterior-samples-are-needed)
derives that prior-corrected form.

## Scope

These pages explain inference mechanics and the repository's modelling
choices. They intentionally do not publish aggregate sample counts,
population-fit values, headline findings, or scientific conclusions.
