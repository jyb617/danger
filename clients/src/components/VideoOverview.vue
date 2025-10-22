<script setup>

import ImageSkeleton from './ImageSkeleton.vue';

const props = defineProps({
  video: {
    required: true,
  },
  selectable: {
    required: false,
  },
});

const coverLoaded = ref(false);
const checkboxComponent = useTemplateRef('checkbox');

const coverUrl = computed(() => {
  return `${window.location.origin}/api/videoinference/cover/${props.video.videoId}`;
});

const emitEvents = defineEmits(['enter']);

const setCoverLoaded = () => {
  coverLoaded.value = true;
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
    emitEvents('enter', props.video.videoId);
  }
};

const stopPropagation = (event) => {
  event.stopPropagation();
}

</script>

<template>
  <ElCard class="overview-wrapper" shadow="hover" @click="onBackgroundClick">
    <div class="video-name">
      <ElPopover placement="bottom-start" trigger="hover">
        <template #reference>
          <ElText truncated>{{video.name}}</ElText>
        </template>
        <template #default>
          <ElText size="small">{{video.note}}</ElText>
        </template>
      </ElPopover>
    </div>
    <div class="video-cover">
      <div v-if="!coverLoaded" class="skeleton-wrapper">
        <ImageSkeleton/>
      </div>
      <img v-show="coverLoaded" class="cover-content" draggable="false" :src="coverUrl" @load="setCoverLoaded">
    </div>
    <div class="video-time">
      <ElText class="time-content" size="small" >{{video.time}}</ElText>
    </div>
    <div class="checkbox-wrapper" ref="checkbox">
      <ElCheckbox v-if="selectable" size="large" :value="video.videoId" @click="stopPropagation"></ElCheckbox>
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
  user-select: none;
}

.video-name {
  position: relative;
  width: 206px;
  height: 20px;
  margin: 0;
  border: 0;
  padding: 0 0 16px 0;
  line-height: 20px;
}

.video-cover {
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
  user-select: none;
}

.cover-content {
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

.video-time {
  position: relative;
  width: 246px;
  height: 12px;
  margin: 0;
  border: 0;
  padding: 16px 0 0 0;
  user-select: none;
}

.time-content {
  position: relative;
  float: right;
  margin: 0;
  border: 0;
  padding: 0;
  user-select: none;
}

</style>
