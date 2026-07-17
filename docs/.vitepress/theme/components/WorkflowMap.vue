<script setup lang="ts">
import { withBase } from 'vitepress'

const stages = [
  { number: '01', title: 'Select', text: 'Prepare MaNGA products and apply kinematic screening.', href: '/methods/data-and-selection.html' },
  { number: '02', title: 'Resolve', text: 'Fit an empirical rotation curve before mass decomposition.', href: '/methods/empirical-rotation-curves.html' },
  { number: '03', title: 'Infer', text: 'Sample the stellar, NFW, and pressure-support model.', href: '/methods/single-galaxy-nfw.html' },
  { number: '04', title: 'Propagate', text: 'Carry full posteriors into a prior-corrected population model.', href: '/methods/population-model.html' },
]
</script>

<template>
  <section class="workflow-map" aria-labelledby="workflow-map-title">
    <header class="workflow-map__header">
      <p class="section-kicker">Research workflow</p>
      <h2 id="workflow-map-title">One inference chain, four auditable stages</h2>
      <p>Each stage links a modeling decision to the public method record and its current implementation.</p>
    </header>
    <ol class="workflow-map__grid">
      <li v-for="stage in stages" :key="stage.number">
        <a :href="withBase(stage.href)">
          <span class="workflow-map__number">{{ stage.number }}</span>
          <strong>{{ stage.title }}</strong>
          <span>{{ stage.text }}</span>
        </a>
      </li>
    </ol>
  </section>
</template>
