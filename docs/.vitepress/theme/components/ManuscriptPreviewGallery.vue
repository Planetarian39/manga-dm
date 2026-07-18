<script setup lang="ts">
import { withBase } from 'vitepress'

const pages = Array.from({ length: 14 }, (_, index) => {
  const number = String(index + 1).padStart(2, '0')
  return {
    number,
    src: `/assets/manuscript-preview/manuscript-page-${number}.webp`,
    isFullPreview: number === '01',
  }
})
</script>

<template>
  <div class="manuscript-page-gallery" aria-label="Draft manuscript page previews">
    <figure v-for="page in pages" :key="page.number" class="manuscript-page-gallery__item">
      <img
        :src="withBase(page.src)"
        :alt="`Draft manuscript page ${page.number}. ${page.isFullPreview ? 'Full preview.' : 'Only the top third is readable; the remainder is irreversibly blurred.'}`"
        loading="lazy"
      >
    </figure>
  </div>
</template>
