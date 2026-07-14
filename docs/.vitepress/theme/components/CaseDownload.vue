<script setup lang="ts">
import { withBase } from 'vitepress'

defineProps<{
  galaxy: string
  href: string
  size: string
  sha: string
}>()
</script>

<template>
  <article class="case-download">
    <div>
      <p class="case-download__eyebrow">Posterior archive</p>
      <h3>{{ galaxy }}</h3>
      <p>{{ size }} · NetCDF · SHA-256 <code>{{ sha }}</code></p>
    </div>
    <a class="case-download__action" :href="withBase(href)" download>Download posterior</a>
  </article>
</template>
