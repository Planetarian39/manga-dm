<script setup lang="ts">
import { withBase } from 'vitepress'

const manuscriptPages = [
  {
    page: '01',
    title: 'Title and abstract',
    src: '/assets/manuscript-preview/manuscript-page-01.webp',
  },
  {
    page: '04',
    title: 'Analytic mass model',
    src: '/assets/manuscript-preview/manuscript-page-04.webp',
  },
  {
    page: '06',
    title: 'Priors and population model',
    src: '/assets/manuscript-preview/manuscript-page-06.webp',
  },
]
</script>

<template>
  <section class="manuscript-preview" aria-labelledby="manuscript-preview-title">
    <div class="manuscript-preview__heading">
      <div>
        <p class="section-kicker">Manuscript preview · draft</p>
        <h2 id="manuscript-preview-title">A concise view of the research manuscript</h2>
      </div>
      <p class="manuscript-preview__notice">
        Protected research content. Read and cite with attribution; do not reproduce, adapt, submit, or publish it
        without written permission. See the
        <a :href="withBase('/research-content-license.html')">Research Content License</a>.
      </p>
    </div>
    <div class="manuscript-preview__grid">
      <figure v-for="item in manuscriptPages" :key="item.page" class="manuscript-preview__card">
        <a :href="withBase('/paper-preview/')" :aria-label="`Open manuscript preview: page ${item.page}`">
          <img :src="withBase(item.src)" :alt="`Draft manuscript page ${item.page}: ${item.title}.`" loading="lazy">
        </a>
        <figcaption><span>Page {{ item.page }}</span>{{ item.title }} · © 2026 Hongyi Xu</figcaption>
      </figure>
    </div>
    <a class="home-button home-button--brand" :href="withBase('/paper-preview/')">Open manuscript preview</a>
  </section>
</template>
