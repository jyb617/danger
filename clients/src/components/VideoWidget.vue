<script setup>

import ImageSkeleton from './ImageSkeleton.vue';
import VideoControl from './VideoControl.vue';
import VideoProgressBar from './VideoProgressBar.vue';

const props = defineProps({
  videoUrl: {
    required: true,
  },
  scores: {
    required: true,
  },
});

const controlEnable = ref(false);
const videoLoaded = ref(false);
const videoPlaying = ref(false);

const progress = ref(0);
const duration = ref(0);

const playerComponent = useTemplateRef('player');

const controlShow = computed(() => {
  if (videoPlaying.value) {
    return controlEnable.value;
  } else {
    return videoLoaded.value;
  }
});

const onTimeUpdate = () => {
  if (playerComponent.value === null) {
    return;
  }
  progress.value = playerComponent.value.currentTime / playerComponent.value.duration;
};

const onVideoLoad = () => {
  videoLoaded.value = true;

  if (playerComponent.value) {
    duration.value = playerComponent.value.duration;
  }
};

const enableControl = () => {
  controlEnable.value = true;
};

const disableControl = () => {
  controlEnable.value = false;
};

const toggleVideoPlaying = () => {
  if (playerComponent.value.paused) {
    playerComponent.value.play();
  } else {
    playerComponent.value.pause();
  }
};

const onVideoPlay = () => {
  videoPlaying.value = true;
};

const onVideoPause = () => {
  videoPlaying.value = false;
};

const onProgressChange = () => {
  if (playerComponent.value === null) {
    return;
  }
  playerComponent.value.currentTime = progress.value * playerComponent.value.duration;
};

const pauseVideo = () => {
  if (playerComponent.value === null) {
    return;
  }
  playerComponent.value.pause();
};

</script>

<template>
  <div class="video-wrapper" @mouseenter="enableControl" @mouseleave="disableControl">
    <div v-if="!videoLoaded" class="skeleton-wrapper">
      <ImageSkeleton/>
    </div>
    <div v-show="videoLoaded" class="video-content">
      <video ref="player" muted autoplay @timeupdate="onTimeUpdate" @loadedmetadata="onVideoLoad" @play="onVideoPlay" @pause="onVideoPause">
        <source :src="videoUrl">
      </video>
    </div>
    <div class="control-wrapper" @click="toggleVideoPlaying">
      <VideoControl :show="controlShow" :playing="videoPlaying" :progress="progress" :duration="duration"/>
    </div>
  </div>
  <div class="progress-wrapper">
    <VideoProgressBar v-model="progress" :scores="scores" @pause="pauseVideo" @change="onProgressChange"/>
  </div>
</template>

<style scoped>

.video-wrapper {
  position: relative;
  width: 856px;
  height: 480px;
  margin: 0;
  border: 0;
  padding: 0;
  user-select: none;
}

.skeleton-wrapper {
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

.video-content {
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

.progress-wrapper {
  position: relative;
  width: 856px;
  height: 24px;
  margin: 0;
  border: 0;
  padding: 14px 0 0 0;
}

.control-wrapper {
  position: absolute;
  width: 100%;
  height: 100%;
  margin: 0;
  border: 0;
  padding: 0;
  user-select: none;
}

</style>
