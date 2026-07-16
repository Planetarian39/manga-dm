---
layout: home

hero:
  name: "Dark matter, read from rotation"
  text: "A method-first guide to the MaNGA inference pipeline"
  tagline: "Trace the path from emission-line velocity maps to single-galaxy NFW posteriors and a prior-corrected population model. Every scientific step is paired with its implementation status."
  actions:
    - theme: brand
      text: Explore the methods
      link: /methods/
    - theme: alt
      text: Inspect 11743-9102
      link: /case-studies/11743-9102
    - theme: alt
      text: Image credits
      link: /#image-credits

features:
  - title: Finalized-paper method
    details: Equations, priors, likelihoods, diagnostics, and quality gates are taken from the finalized method—not inferred from CLI defaults.
  - title: Code-aware documentation
    details: Each method page points to the current modules and labels known differences instead of hiding them.
  - title: Reviewable case data
    details: Four allowlisted single-galaxy posterior files and fit figures demonstrate the workflow without exposing aggregate findings.
---

<section id="image-credits" class="home-image-credits" aria-labelledby="image-credits-title">
  <p class="section-kicker">Hero image credits</p>
  <h2 id="image-credits-title">MaNGA telescope</h2>
  <div class="home-image-credits__grid">
    <p>
      <strong>MaNGA telescope</strong><br>
      Image credit: Sloan Digital Sky Survey (SDSS), CC BY.<br>
      <a href="https://www.sdss4.org/wp-content/uploads/2021/05/manga_4.png">Original image</a>
      · <a href="https://www.sdss.org/collaboration/image-use-policy/">SDSS image-use policy</a>
    </p>
  </div>
</section>

<WorkflowMap />

## Choose a reader path

<div class="reader-paths">

### Understand the method

Start with [data and selection](/methods/data-and-selection), then follow the inference chain through the [population likelihood](/methods/population-model).

### Inspect a worked case

Use [11743-9102](/case-studies/11743-9102) to connect a fit figure, posterior geometry, diagnostics, and a complete per-galaxy NetCDF file.

### Reproduce the pipeline

Move from [installation](/run/installation) to the [CLI workflow](/run/cli-workflow), configuration, and output schemas.

</div>

::: warning Publication boundary
This site documents methods and software. It intentionally excludes unpublished aggregate results, scientific interpretation, novelty claims, discussion, and conclusions. The four case studies are method demonstrations, not a population comparison.
:::

## A transparent implementation status

The finalized paper and the current CLI are not identical in every default. The public method uses a 60° azimuthal cut, an exact predictive-HDI probability of 0.9545, the finalized quality equation, and a prior-corrected posterior-sample likelihood. The current implementation has known differences in each area.

[Read the implementation status →](/project/implementation-status)
