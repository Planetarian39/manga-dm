<script setup lang="ts">
import DefaultTheme from 'vitepress/theme'
import { useData } from 'vitepress'
import ManuscriptPreview from './components/ManuscriptPreview.vue'

const { frontmatter } = useData()
</script>

<template>
  <DefaultTheme.Layout>
    <template #home-features-before>
      <ManuscriptPreview v-if="frontmatter.layout === 'home'" />
    </template>
  </DefaultTheme.Layout>
</template>
