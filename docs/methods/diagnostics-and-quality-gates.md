---
title: Diagnostics and Quality Gates
description: Convergence, predictive, fit-quality, degeneracy, and importance-sampling checks.
---

# Diagnostics and quality gates

No single diagnostic is sufficient. A fit can mix well but describe the velocity field poorly; it can pass residual checks while retaining an unusably degenerate halo posterior; and a valid single-galaxy posterior can still be a poor importance proposal for a candidate population model.

## Empirical-stage gates

| Diagnostic | Paper requirement | Purpose |
|---|---:|---|
| Inclination | $25^\circ\le i\le70^\circ$ | Avoid weak face-on projection and highly inclined geometry |
| Usable spaxels | $N_{\mathrm{valid}}\ge150$ | Require spatial information |
| Radial extent | $R_{\mathrm{out}}/R_t\ge2$ | Observe beyond the empirical turnover |
| Convergence | $\hat R\le1.05$ | Check between-chain agreement |
| Effective samples | ESS $\ge200$ | Require adequate posterior information |
| Predictive coverage | $f_{\mathrm{HDI}}>0.60$ | Observations fall inside the predictive interval |
| Predictive overlap | $g_{\mathrm{HDI}}>0.80$ | Measurement intervals overlap predictive intervals |

Coverage and overlap use predictive probability `0.9545`, described as 95% in paper prose.

## Single-galaxy NFW retention

$$
\left\{
\begin{aligned}
\hat R &\le 1.05,\\
\mathrm{ESS}_{\mathrm{bulk}} &\ge 200,\\
\chi^2_\nu &\le 2.0,\\
0.1\le p_{\mathrm{PPC}} &\le 0.9,\\
|\rho(c,M_{200})| &\le 0.85.
\end{aligned}
\right.
$$

### Convergence

- **$\hat R$** compares within-chain and between-chain variation.
- **Bulk ESS** estimates effectively independent draws in the central posterior.
- **Trace plots** should show stationary, overlapping chains without persistent trends or mode separation.
- **Divergences** indicate integration difficulty and require investigation even when summaries pass.

### Fit and predictive adequacy

- **Reduced chi-squared** evaluates residuals of the posterior-median rotation curve relative to the formal observation scale. It is a diagnostic, not the robust sampling likelihood.
- **Posterior-predictive $p$-value** checks whether replicated-data discrepancies place the observation in an extreme tail. Both extremes are rejected.
- **Coverage and overlap** test local velocity-field agreement rather than only a global residual.

### Posterior geometry

- **$\rho(c,M_{200})$** is computed from log-concentration and log-halo-mass draws. The gate rejects an almost one-dimensional degeneracy ridge.
- **Pair plots** reveal curvature, connectedness, and trade-offs with stellar mass, bulge fraction, effective radius, and pressure support.
- **Marginals** should be checked for truncation, unresolved multimodality, and prior domination.

The 11743-9102 case study may use its trace and pair plots as a worked inspection, without quoting individual posterior values as findings.

## Importance-sampling diagnostics

For

$$
w_{is}=
\frac{p_{\mathrm{pop}}(\boldsymbol\theta_{is}\mid\Phi)}
{p_{\mathrm{stage1}}(\boldsymbol\theta_{is})},
$$

the generic Pareto-$\hat k$ interpretation is:

| Pareto $\hat k$ | Interpretation |
|---:|---|
| $<0.5$ | Reliable finite-variance behavior |
| $0.5\le\hat k<0.7$ | Caution; increasing tail sensitivity |
| $\ge0.7$ | Unreliable raw importance-weight behavior |

The importance ESS fraction, $\mathrm{ESS}/S$, measures how many draws materially contribute after reweighting. Low values mean that a few draws carry most of the likelihood estimate. These checks are per galaxy; unpublished aggregate counts and fractions are outside the public boundary.

## Decision checklist

1. Record the exact screen configuration and interval probability.
2. Apply every empirical gate.
3. Inspect warnings, divergences, and trace behavior.
4. Apply the complete NFW retention equation.
5. Confirm that prior metadata accompanies posterior samples.
6. Inspect Pareto $\hat k$ and importance ESS during Stage 2.
7. Record implementation fallbacks as provenance, not equivalent paper settings.

<MethodStatus status="paper">

The paper uses the complete quality equation on this page together with
`60°`/`0.9545` screening and a posterior-sample Stage 2 path.

</MethodStatus>

<MethodStatus status="implementation">

`recommended` uses reduced chi-squared `3.0`, PPC `0.05`--`0.95`, and
correlation `0.95`. `strict` reaches `2.0` and `0.10`--`0.90` but uses
correlation `0.90` and adds different coverage gates. Empirical R-hat/ESS
warn rather than determine the pass boolean, and neither preset is the paper
profile.

</MethodStatus>

See [Implementation status](../project/implementation-status.md) and [Limitations](../project/limitations.md).
