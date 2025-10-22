<script setup>

const props = defineProps({
  scores: {
    required: true,
  },
});

</script>

<template>
  <div class="heatmap-wrapper">
    <div v-for="score in scores" class="heatmap-block" :style="{backgroundColor: `rgb(${score * 215}, ${(1 - score) * 215}, 0)`}"></div>
  </div>
</template>

<style scoped>

.heatmap-wrapper {
  display: flex;
  position: relative;
  width: 100%;
  height: 100%;
  margin: 0;
  border: 0;
  padding: 0;
  flex-direction: row;
}

.heatmap-block {
  position: relative;
  height: 100%;
  margin: 0;
  border: 0;
  padding: 0;
  flex-grow: 1;
}

</style>
