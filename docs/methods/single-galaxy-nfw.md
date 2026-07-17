---
title: Single-Galaxy NFW Model
description: Stellar, NFW halo, pressure-support, prior, likelihood, and sampling model for one galaxy.
---

# Single-galaxy dynamical and NFW model

For each galaxy that passes empirical screening, the physical model decomposes ordered rotational support into stellar, dark-matter, and pressure-support terms. It retains the complete posterior because the halo parameters are correlated and can have non-Gaussian geometry.

::: info What this page demonstrates
- **Scientific purpose:** infer stellar and NFW halo contributions while retaining correlated parameter uncertainty.
- **Key modeling decision:** sample the full stellar-plus-NFW model with NUTS instead of reporting only a best fit.
- **My implementation contribution:** I built the PyMC 5 posterior-inference and uncertainty-quantification workflow used for the public galaxy cases.
- **Main limitation:** limited radial coverage, fixed inclination, baryonic assumptions, and beam smearing can all affect the inferred halo parameters.
:::

## Force budget

$$
V_{\mathrm{rot}}^2(r)
=V_{\mathrm{baryon}}^2(r)
+V_{\mathrm{dm}}^2(r)
-V_{\mathrm{drift}}^2(r).
$$

The baseline approximates $V_{\mathrm{baryon}}\simeq V_\star$: stellar gravity is explicit, gas gravity is not a separate term, and gas turbulence enters through pressure support.

## Stellar model

$$
V_\star^2=V_{\mathrm{bulge}}^2+V_{\mathrm{disk}}^2,
\quad
M_{\mathrm{bulge}}=f_{\mathrm{bulge}}M_\star,
\quad
M_{\mathrm{disk}}=(1-f_{\mathrm{bulge}})M_\star.
$$

For a Hernquist bulge,

$$
V_{\mathrm{bulge}}^2(r)=\frac{GM_{\mathrm{bulge}}r}{(r+a)^2},
\qquad a=\frac{R_e}{1.8153}.
$$

For a thin exponential disk,

$$
V_{\mathrm{disk}}^2(r)
=\frac{2GM_{\mathrm{disk}}}{R_d}y^2
\left[I_0(y)K_0(y)-I_1(y)K_1(y)\right],
$$

where $y=r/(2R_d)$ and $R_d=R_e/1.678$.

## NFW halo and cosmology

With $x=r/R_{200}$,

$$
V_{\mathrm{dm}}^2(r)=V_{200}^2
\frac{\ln(1+cx)-cx/(1+cx)}
{x[\ln(1+c)-c/(1+c)]}.
$$

$$
V_{200}=[10GH(z)M_{200}]^{1/3},
\qquad
H(z)=H_0\sqrt{\Omega_m(1+z)^3+\Omega_\Lambda}.
$$

The adopted constants are $H_0=67.4\ \mathrm{km\,s^{-1}\,Mpc^{-1}}$, $\Omega_m=0.315$, and $\Omega_\Lambda=0.685$.

## Pressure support

$$
V_{\mathrm{drift}}^2(r)=2\sigma_0^2(r/R_d).
$$

The factor of two corresponds to the adopted thin-disk Jeans approximation with gas-dispersion scale length $R_d/2$. The fitted $\sigma_0$ is an empirical characteristic scale, not a point-by-point calibration to the observed dispersion map.

## SHMR anchor

$$
\frac{M_\star}{M_{\mathrm{halo}}}
=2N\left[
\left(\frac{M_{\mathrm{halo}}}{M_1}\right)^{-\beta}
+\left(\frac{M_{\mathrm{halo}}}{M_1}\right)^\gamma
\right]^{-1}.
$$

The Moster et al. (2013) constants are defined in `src/config/constants.py`. This relation anchors the halo-mass prior. The concentration prior remains independent of halo mass, so Stage 1 does not impose a population $c$-$M$ slope.

## Priors

| Parameter | Manuscript prior | Purpose |
|---|---|---|
| $\log_{10}M_\star$ | $\mathcal N(\log_{10}M_{\star,\mathrm{NSA}},0.05\,\mathrm{dex})$ | Photometric mass anchor |
| $\log_{10}M_{200}$ | $\mathcal{TN}(\mu_{\mathrm{SHMR}},0.15\,\mathrm{dex};\pm3\sigma)$ | Halo-mass anchor |
| $\log_{10}c$ | $\mathcal N(\log_{10}9.0,0.50\,\mathrm{dex})$ | Broad mass-independent concentration prior |
| $\log_{10}\sigma_0$ | log-normal centered on $10\,\mathrm{km\,s^{-1}}$, width $0.20\,\mathrm{dex}$ | Pressure-support scale |
| $V_{\mathrm{sys}}$ | truncated Normal about the NSA value, $\sigma=5\,\mathrm{km\,s^{-1}}$, bounds $\pm20\,\mathrm{km\,s^{-1}}$ | Systemic velocity |
| $i$ | $\delta(i-i_{\mathrm{NSA}})$ | Fixed photometric inclination |
| $\log_{10}R_e$ | log-normal about $R_{e,\mathrm{NSA}}$, width $0.05\,\mathrm{dex}$ | Structural scale |
| $\operatorname{logit}(f_{\mathrm{bulge}})$ | $\mathcal N(1.2(n-2.5),0.2)$ | Sersic morphology anchor |
| $\sigma_{\mathrm{int}}$ | Exponential with manuscript-stated scale $2\bar\sigma_{\mathrm{meas}}$ | Scatter beyond IVAR |
| $\nu-2$ | $\Gamma(2.0,0.1)$ | Student-t tail weight |

Reproducible metadata should record rate-versus-scale conventions for Gamma and Exponential APIs. The table preserves the manuscript parameterization; `src/models/dm_nfw.py` provides the executable mapping.

## Likelihood and inner weighting

$$
V_{\mathrm{obs}}\sim\operatorname{StudentT}\!\left(
\nu,V_{\mathrm{obs,model}},
\sqrt{\sigma_{\mathrm{meas}}^2+\sigma_{\mathrm{int}}^2}
\right).
$$

The log-likelihood has a radial logistic weight with half-weight radius $0.3r_{\max}$, transition width $0.1r_{\max}$, and minimum central weight `0.3`. This mitigates central beam smearing but is not explicit PSF convolution.

## Sampling and output

| Setting | Manuscript value |
|---|---:|
| Framework | PyMC 5 |
| Sampler | NUTS with `nutpie` |
| Tuning steps | 500 |
| Posterior draws | 1000 per chain |
| Chains | Up to 4 |
| Target acceptance | 0.95 |
| Summaries and diagnostics | ArviZ |

The pipeline stores raw posterior draws rather than only medians or equal-tailed intervals. Output I/O is in `src/data/results.py`.

<MethodStatus status="paper">

The manuscript requires reduced chi-squared ≤ 2.0, PPC p-value from 0.1 through
0.9, absolute concentration–mass correlation ≤ 0.85, R-hat ≤ 1.05, and bulk
ESS ≥ 200. Stage 2 uses prior-corrected
posterior samples.

</MethodStatus>

<MethodStatus status="implementation">

The NFW sampler settings match, but upstream `60°`/`0.9545` screening is not a
versioned profile. Neither `recommended` nor `strict` matches the complete
manuscript equation, and the public population path defaults to GMM inputs.

</MethodStatus>

Next: [Population model](./population-model.md) and [Diagnostics](./diagnostics-and-quality-gates.md).
