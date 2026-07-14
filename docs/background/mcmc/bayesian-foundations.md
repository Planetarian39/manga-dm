---
title: Bayesian Foundations
description: Bayes' theorem, Monte Carlo integration, Markov chains, and Metropolis-Hastings.
---

# Bayesian foundations

## Bayes' theorem

### Derivation

Bayes' theorem follows from the multiplication rule for conditional
probability. For events or random variables $A$ and $B$, the joint probability
can be decomposed in either order:

$$
p(A,B) = p(A \mid B)\,p(B) = p(B \mid A)\,p(A).
$$

Substitute parameters $\theta$ for $A$ and observations $D$ for $B$:

$$
p(\theta,D)
= p(\theta \mid D)\,p(D)
= p(D \mid \theta)\,p(\theta).
$$

Dividing by $p(D)$ gives

$$
\boxed{
p(\theta \mid D)
= \frac{p(D \mid \theta)\,p(\theta)}{p(D)}
}.
$$

The denominator marginalizes over the parameter space so that the posterior
integrates to one:

$$
p(D) = \int p(D \mid \theta)\,p(\theta)\,d\theta \equiv Z.
$$

### Terms in Bayes' theorem

| Symbol | Name | Meaning |
|---|---|---|
| $p(\theta \mid D)$ | posterior | Parameter distribution after conditioning on the observations |
| $p(D \mid \theta)$ | likelihood | Probability model for the observations at a parameter value |
| $p(\theta)$ | prior | Parameter distribution before conditioning on these observations |
| $p(D)=Z$ | marginal likelihood or evidence | Normalizing integral over the parameter space |

Even when $Z$ is unavailable, the unnormalized posterior can often be
evaluated point by point:

$$
f(\theta) = p(D \mid \theta)\,p(\theta).
$$

Because $p(\theta \mid D)=f(\theta)/Z$, a ratio at two parameter values does not
contain $Z$:

$$
\frac{p(\theta' \mid D)}{p(\theta \mid D)}
= \frac{f(\theta')/Z}{f(\theta)/Z}
= \frac{f(\theta')}{f(\theta)}.
$$

This cancellation is the key to many MCMC transition rules. A sampler can
compare relative posterior density without evaluating the global normalizing
integral.

## Monte Carlo integration

### Sample averages

If $\theta_1,\ldots,\theta_K$ are independent draws from a normalized density
$p(\theta)$, then

$$
\mathbb{E}_{p(\theta)}[g(\theta)]
= \int g(\theta)\,p(\theta)\,d\theta
\approx \frac{1}{K}\sum_{k=1}^{K}g(\theta_k).
$$

The law of large numbers explains why the approximation improves as $K$
grows. For independent draws, Monte Carlo error generally decreases as
$K^{-1/2}$.

### Unnormalized targets

When $p(\theta)=f(\theta)/Z$, the expectation can also be written

$$
\mathbb{E}_{p(\theta)}[g(\theta)]
= \frac{\int g(\theta)\,f(\theta)\,d\theta}
       {\int f(\theta)\,d\theta}.
$$

If a sampler produces draws from the normalized target $p$, the ordinary
sample average still estimates this quantity; the sampler need not report
$Z$. The identity does not by itself describe how to obtain those draws.
Metropolis-Hastings is one construction.

MCMC draws are correlated rather than independent. Under stationarity and
ergodicity, a Markov-chain law of large numbers still supports sample
averages, but autocorrelation reduces the information in $K$ stored draws.
Effective sample size estimates that reduction.

## Markov chains

A Markov chain is a stochastic sequence in which the next state depends on
the current state, not on the complete path:

$$
p\!\left(\theta^{(t+1)}
  \mid \theta^{(t)},\theta^{(t-1)},\ldots\right)
= p\!\left(\theta^{(t+1)} \mid \theta^{(t)}\right).
$$

MCMC constructs a chain whose stationary distribution is the target posterior
$p(\theta \mid D)$. After warmup, and provided the chain can reach all relevant
regions and has mixed adequately, retained states behave as correlated
posterior draws.

Three qualifications matter:

- **Stationarity:** the distribution of chain states must no longer depend
  materially on the initial state.
- **Ergodicity:** the transition must be able to explore the target rather
  than remain trapped in a disconnected subset.
- **Finite-run error:** no diagnostic proves convergence in finite time;
  multiple diagnostics and domain checks provide evidence, not certainty.

## The Metropolis-Hastings algorithm

Metropolis-Hastings (MH) is a foundational MCMC method. It makes the
normalization cancellation concrete.

### Metropolis-Hastings steps

Given the current state $\theta^{(t)}$:

1. Propose $\theta'$ from $q(\theta' \mid \theta^{(t)})$.
2. Compute

   $$
   \alpha
   = \min\!\left(
       1,
       \frac{f(\theta')\,q(\theta^{(t)} \mid \theta')}
            {f(\theta^{(t)})\,q(\theta' \mid \theta^{(t)})}
     \right).
   $$

3. Accept the proposal with probability $\alpha$; otherwise repeat the
   current state.
4. Iterate until the retained chain is long enough for the intended
   summaries.

For a symmetric proposal,
$q(\theta' \mid \theta)=q(\theta \mid \theta')$, so

$$
\alpha
= \min\!\left(1,\frac{f(\theta')}{f(\theta^{(t)})}\right).
$$

A higher-density proposal is accepted. A lower-density proposal can still be
accepted, which prevents the chain from acting like a local optimizer. The
proposal scale remains important: steps that are too small mix slowly, while
steps that are too large are rejected frequently. Modern HMC and NUTS replace
the random-walk proposal with gradient-informed trajectories, but the goal is
the same: leave the desired posterior invariant while exploring it
efficiently.

## Continue

The [priors and data](./priors-and-data.md) page applies these ideas to a model
with an analytical posterior before returning to MCMC.
