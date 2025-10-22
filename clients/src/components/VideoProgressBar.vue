<script setup>

import AnomalyHeatmap from './AnomalyHeatmap.vue';

const props = defineProps({
  scores: {
    required: true,
  },
});

const progress = defineModel({
  required: true,
});

const pointerDragging = ref(false);
const pointerBoundingComponent = useTemplateRef('bounding');

const eventEmitter = defineEmits(['pause', 'change']);

const setPointerDragging = () => {
  pointerDragging.value = true;
};

const unsetPointerDragging = () => {
  pointerDragging.value = false;
};

const setupMouseMoveListener = () => {
  document.addEventListener('mousemove', mouseMoveCallback);
};

const removeMouseMoveListener = () => {
  document.removeEventListener('mousemove', mouseMoveCallback);
};

const progressBoundingLimit = (unboundProgress) => {
  if (unboundProgress >= 1) {
    return 1;
  }
  if (unboundProgress <= 0) {
    return 0;
  }
  return unboundProgress;
};

const mouseMoveCallback = (event) => {
  if (pointerDragging.value && pointerBoundingComponent.value) {
    const boundingRect = pointerBoundingComponent.value.getBoundingClientRect();

    const rangeLeft = boundingRect.left;
    const rangeWidth = boundingRect.width;

    progress.value = progressBoundingLimit((event.clientX - rangeLeft) / rangeWidth);
  }
};

const mouseUpCallback = () => {
  if (pointerDragging.value) {
    unsetPointerDragging();

    removeMouseUpListener();
    removeMouseMoveListener();

    eventEmitter('change');
  }
};

const setupMouseUpListener = () => {
  document.addEventListener('mouseup', mouseUpCallback);
};

const removeMouseUpListener = () => {
  document.removeEventListener('mouseup', mouseUpCallback);
};

const pointerDragStart = () => {
  eventEmitter('pause');

  setupMouseUpListener();
  setupMouseMoveListener();

  setPointerDragging();
};

const onProgressClick = (event) => {
  if (pointerBoundingComponent.value) {
    const boundingRect = pointerBoundingComponent.value.getBoundingClientRect();

    const rangeLeft = boundingRect.left;
    const rangeWidth = boundingRect.width;

    progress.value = progressBoundingLimit((event.clientX - rangeLeft) / rangeWidth);
  }
  eventEmitter('change');
};

onBeforeUnmount(() => {
  removeMouseUpListener();
  removeMouseMoveListener();
});

</script>

<template>
  <div class="progress-bar-wrapper" @click="onProgressClick">
    <div class="heatmap-wrapper">
      <AnomalyHeatmap :scores="scores"/>
    </div>
    <div class="pointer-wrapper" ref="bounding">
      <div :class="['pointer', {'pointer-active': pointerDragging}]" :style="{left: `${progress * 100}%`}" @mousedown="pointerDragStart"></div>
    </div>
  </div>
</template>

<style scoped>

.progress-bar-wrapper {
  position: relative;
  width: 100%;
  height: 20px;
  margin: 0;
  border: 0;
  padding: 0;
}

.heatmap-wrapper {
  position: absolute;
  overflow: hidden;
  width: 100%;
  height: 100%;
  margin: 0;
  border: 0;
  padding: 0;
}

.pointer-wrapper {
  position: absolute;
  overflow: visible;
  width: 100%;
  margin: 0;
  border: 0;
  padding: 10px 0 10px 0;
}

.pointer {
  position: absolute;
  width: 10px;
  height: 28px;
  margin: 0;
  border: 0;
  padding: 0;
  transform: translate(-50%, -50%);
  background-color: var(--color-primary-dark-8);
}

.pointer-active {
  width: 12px;
  height: 32px;
}

</style>
