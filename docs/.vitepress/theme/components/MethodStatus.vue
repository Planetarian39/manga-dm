<script setup lang="ts">
import { computed } from 'vue'
import { withBase } from 'vitepress'

const props = defineProps<{
  status: 'paper' | 'implementation'
}>()

const label = computed(() => (props.status === 'paper' ? 'Manuscript method' : 'Current implementation'))
</script>

<template>
  <aside class="method-status" :data-status="status" :aria-label="label">
    <p class="method-status__label">{{ label }}</p>
    <div class="method-status__body"><slot /></div>
    <p v-if="status === 'implementation'" class="method-status__link">
      <a :href="withBase('/project/implementation-status.html')">Implementation status</a>
    </p>
  </aside>
</template>
