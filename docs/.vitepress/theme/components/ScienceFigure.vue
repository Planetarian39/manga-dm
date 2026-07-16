<script setup lang="ts">
import { withBase } from 'vitepress'

defineProps<{
  src: string
  alt: string
  caption: string
  source?: string
  sourceHref?: string
}>()
</script>

<template>
  <figure class="science-figure">
    <a :href="withBase(src)" target="_blank" rel="noopener" aria-label="Open the full-resolution figure">
      <img :src="withBase(src)" :alt="alt" loading="lazy" />
    </a>
    <figcaption>
      <span>{{ caption }}</span>
      <small v-if="source">
        <a v-if="sourceHref" :href="sourceHref" target="_blank" rel="noopener">{{ source }}</a>
        <template v-else>{{ source }}</template>
      </small>
    </figcaption>
  </figure>
</template>
