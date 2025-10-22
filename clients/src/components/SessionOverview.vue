<script setup>

import ImageSkeleton from './ImageSkeleton.vue';

const props = defineProps({
  session: {
    required: true,
  },
  selectable: {
    required: false,
  },
});

const sessionLoaded = ref(false);
const checkboxComponent = useTemplateRef('checkbox');

const sessionUrl = computed(() => {
  return `${window.location.origin}/api/realtimeinference/session/${props.session.sessionId}`;
});

const emitEvents = defineEmits(['enter']);

const setSessionLoaded = () => {
  sessionLoaded.value = true;
};

const simulateCheckboxClick = () => {
  const checkboxes = checkboxComponent.value.children;

  if (checkboxes.length > 0) {
    checkboxes[0].click();
  }
};

const onBackgroundClick = () => {
  if (props.selectable) {
    simulateCheckboxClick();
  } else {
    emitEvents('enter', props.session.sessionId)
  }
};

const stopPropagation = (event) => {
  event.stopPropagation();
}

</script>

<template>
  <ElCard class="overview-wrapper" shadow="hover" @click="onBackgroundClick">
    <div class="session-name">
      <ElPopover placement="bottom-start" trigger="hover">
        <template #reference>
          <ElText truncated>{{session.name}}</ElText>
        </template>
        <template #default>
          <ElText size="small">{{session.note}}</ElText>
        </template>
      </ElPopover>
    </div>
    <div class="session-body">
      <div v-if="!sessionLoaded" class="skeleton-wrapper">
        <ImageSkeleton/>
      </div>
      <img v-show="sessionLoaded" class="session-content" draggable="false" :src="sessionUrl" @load="setSessionLoaded">
    </div>
    <div class="session-source">
      <ElText size="small" class="source-content">{{session.source}}</ElText>
    </div>
    <div class="checkbox-wrapper" ref="checkbox">
      <ElCheckbox v-if="selectable" size="large" :value="session.sessionId" @click="stopPropagation"></ElCheckbox>
    </div>
  </ElCard>
</template>

<style scoped>

.overview-wrapper {
  position: relative;
  width: 286px;
  height: 240px;
  margin: 9.2px;
  padding: 0;
  background-color: var(--color-background-light-4);
}

.checkbox-wrapper {
  position: absolute;
  top: 8px;
  right: 14px;
  margin: 0;
  border: 0;
  padding: 0;
}

.session-name {
  position: relative;
  width: 206px;
  height: 20px;
  margin: 0;
  border: 0;
  padding: 0 0 16px 0;
  line-height: 20px;
}

.session-body {
  position: relative;
  width: 246px;
  height: 138px;
  border: 0;
  margin: 0;
  padding: 0;
  user-select: none;
}

.skeleton-wrapper {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  border: 0;
  margin: 0;
  padding: 0;
}

.session-content {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  margin: 0;
  border: 0;
  padding: 0;
  user-select: none;
}

.session-source {
  position: relative;
  width: 246px;
  height: 12px;
  margin: 0;
  border: 0;
  padding: 16px 0 0 0;
  user-select: none;
}

.source-content {
  position: relative;
  margin: 0;
  border: 0;
  padding: 0;
  user-select: none;
}

</style>
