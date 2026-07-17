<script setup lang="ts">
import { withBase } from 'vitepress'

const evidence = [
  { value: 'MaNGA DR17', label: 'Spatially resolved galaxy kinematics' },
  { value: 'PyMC 5 + NUTS', label: 'Bayesian posterior inference' },
  { value: '4 cases', label: 'Reviewable single-galaxy artifacts' },
  { value: '4,000 draws', label: 'Pinned representative posterior' },
  { value: 'NetCDF + SHA-256', label: 'Downloadable, verified provenance' },
]

const contributions = [
  ['Data and screening', 'Connected MaNGA velocity-field inputs to reproducible kinematic quality gates.'],
  ['Bayesian modeling', 'Built galaxy-level stellar-plus-NFW and population-level inference workflows in Python/PyMC 5.'],
  ['Uncertainty', 'Retained correlated posterior samples instead of reducing each galaxy to a point estimate.'],
  ['Validation', 'Documented convergence, posterior geometry, predictive checks, and known implementation differences.'],
]
</script>

<template>
  <div class="application-home">
    <section class="evidence-rail" aria-labelledby="evidence-title">
      <div class="home-section-heading">
        <p class="section-kicker">Evidence at a glance</p>
        <h2 id="evidence-title">A research project with inspectable outputs</h2>
      </div>
      <dl class="evidence-rail__grid">
        <div v-for="item in evidence" :key="item.value">
          <dt>{{ item.value }}</dt>
          <dd>{{ item.label }}</dd>
        </div>
      </dl>
    </section>

    <section class="home-research" aria-labelledby="research-question-title">
      <div class="home-research__question">
        <p class="section-kicker">Research question</p>
        <h2 id="research-question-title">What can inner-galaxy kinematics tell us about a dark-matter halo?</h2>
        <p>
          MaNGA maps ionized-gas motion across nearby galaxies, providing a resolved view of how stars and dark matter
          contribute to rotation. The challenge is that the data cover only the inner halo, where halo mass
          <span class="no-break"><i>M</i><sub>200</sub></span> and concentration <i>c</i> can trade off along a curved
          posterior ridge.
        </p>
        <p>
          This project screens velocity fields, fits a physical stellar-plus-NFW model, and carries complete
          single-galaxy posteriors into population inference so that this uncertainty is not erased.
        </p>
        <a class="text-link" :href="withBase('/overview/project-overview.html')">Read the two-minute overview</a>
      </div>

      <div class="home-contribution" aria-labelledby="contribution-title">
        <p class="section-kicker">My contribution</p>
        <h2 id="contribution-title">From physical question to auditable computation</h2>
        <p>
          I built the Bayesian inference and uncertainty-quantification workflow and connected its scientific
          decisions to public case artifacts, diagnostics, and documentation.
        </p>
        <dl>
          <div v-for="item in contributions" :key="item[0]">
            <dt>{{ item[0] }}</dt>
            <dd>{{ item[1] }}</dd>
          </div>
        </dl>
      </div>
    </section>

    <WorkflowMap />

    <section class="featured-case" aria-labelledby="featured-case-title">
      <div class="featured-case__media">
        <img
          :src="withBase('/assets/case-studies/11743-9102/nfw-fit.png')"
          alt="Component rotation-curve fit for MaNGA galaxy 11743-9102, showing the total, stellar, and dark-matter contributions."
          loading="lazy"
        >
      </div>
      <div class="featured-case__body">
        <p class="section-kicker">Featured worked galaxy</p>
        <h2 id="featured-case-title">MaNGA 11743-9102</h2>
        <p class="featured-case__lede">
          A complete walkthrough from velocity-field context to mass decomposition, posterior geometry, convergence
          checks, and a downloadable posterior artifact.
        </p>
        <dl class="featured-case__metrics">
          <div>
            <dt>log<sub>10</sub>(<i>M</i><sub>200</sub>/<i>M</i><sub>☉</sub>)</dt>
            <dd>12.5738 <span>+0.1434 / −0.1506</span></dd>
          </div>
          <div>
            <dt>log<sub>10</sub> <i>c</i></dt>
            <dd>0.9814 <span>+0.0770 / −0.0801</span></dd>
          </div>
          <div>
            <dt>Posterior geometry</dt>
            <dd>ρ = −0.6888 <span>4,000 retained draws</span></dd>
          </div>
        </dl>
        <p class="featured-case__note">
          The diagnostic record lists a maximum R-hat of 1.007 and ESS above 800. Limited radial coverage and fixed
          photometric inclination remain important interpretation limits.
        </p>
        <div class="featured-case__actions">
          <a class="home-button home-button--brand" :href="withBase('/case-studies/11743-9102.html')">Read the worked example</a>
          <a class="home-button" :href="withBase('/case-studies/downloads.html#files')">Inspect the posterior</a>
        </div>
      </div>
    </section>

    <section class="validation-ledger" aria-labelledby="validation-title">
      <div class="home-section-heading">
        <p class="section-kicker">Validation and limits</p>
        <h2 id="validation-title">Clear about what works—and what remains approximate</h2>
      </div>
      <div class="validation-ledger__grid">
        <article>
          <p class="validation-ledger__number">01</p>
          <h3>What is validated</h3>
          <p>Four pinned case artifacts, posterior summaries generated from source files, documentation boundary checks, and deployment-level route and byte-size checks.</p>
        </article>
        <article>
          <p class="validation-ledger__number">02</p>
          <h3>Known differences</h3>
          <p>The public CLI contains the core scientific components, but its fallback thresholds and default Stage 2 path do not form one versioned manuscript profile.</p>
        </article>
        <article>
          <p class="validation-ledger__number">03</p>
          <h3>What comes next</h3>
          <p>A versioned profile, complete provenance, and numerical regression against approved single-galaxy and population references before reproduction claims.</p>
        </article>
      </div>
      <a class="text-link" :href="withBase('/project/implementation-status.html')">See the complete implementation status</a>
    </section>

    <section class="reproducibility-band" aria-labelledby="reproducibility-title">
      <div>
        <p class="section-kicker">Reproducibility</p>
        <h2 id="reproducibility-title">Scientific software as part of the evidence</h2>
      </div>
      <ul>
        <li><strong>One CLI</strong><span><code>manga</code> connects selection, Stage 1, merge, Stage 2, figures, and sampling.</span></li>
        <li><strong>Responsibility-oriented modules</strong><span>Data, models, pipeline, statistics, visualization, configuration, and a thin CLI remain independently inspectable.</span></li>
        <li><strong>Versioned artifacts</strong><span>NetCDF posterior files include exact byte counts, SHA-256 digests, schemas, and source provenance.</span></li>
        <li><strong>Automated checks</strong><span>Documentation tests, Python extraction tests, built-route checks, and post-deployment HTTP checks guard the public release.</span></li>
      </ul>
      <div class="reproducibility-band__links">
        <a class="home-button home-button--brand" :href="withBase('/run/')">Run the pipeline</a>
        <a class="home-button" :href="withBase('/project/architecture.html')">View code and architecture</a>
      </div>
    </section>

    <section class="researcher-band" aria-labelledby="researcher-title">
      <div>
        <p class="section-kicker">About the researcher</p>
        <h2 id="researcher-title">Hongyi Xu</h2>
      </div>
      <div>
        <p class="researcher-band__identity">Undergraduate student in the Department of Physics at the University of Toronto.</p>
        <p>
          My research interests center on computational physics: using statistical inference, machine learning, and
          reproducible computational experiments to extract physical insight from complex data.
        </p>
        <a class="text-link" :href="withBase('/about/')">Research background and contribution statement</a>
      </div>
    </section>

    <section id="image-credits" class="home-credits" aria-labelledby="image-credits-title">
      <p class="section-kicker">Credits and references</p>
      <h2 id="image-credits-title">MaNGA telescope</h2>
      <p>
        Hero image: Sloan Digital Sky Survey (SDSS), CC BY.
        <a href="https://www.sdss4.org/wp-content/uploads/2021/05/manga_4.png">Original image</a>
        · <a href="https://www.sdss.org/collaboration/image-use-policy/">SDSS image-use policy</a>
      </p>
    </section>
  </div>
</template>
