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

const videoError = ref(false);
const errorMessage = ref('');

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
  videoError.value = false;

  if (playerComponent.value) {
    duration.value = playerComponent.value.duration;
  }
};

const onVideoError = (event) => {
  videoError.value = true;
  videoLoaded.value = false;

  const error = playerComponent.value?.error;
  if (error) {
    switch (error.code) {
      case 1:
        errorMessage.value = '视频加载被中止';
        break;
      case 2:
        errorMessage.value = '网络错误，无法加载视频';
        break;
      case 3:
        errorMessage.value = '视频解码失败，可能是格式不支持';
        break;
      case 4:
        errorMessage.value = '视频格式不支持或文件损坏';
        break;
      default:
        errorMessage.value = '未知错误';
    }
  }
  console.error('视频加载错误:', errorMessage.value, error);
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
    <div v-if="!videoLoaded && !videoError" class="skeleton-wrapper">
      <ImageSkeleton/>
    </div>
    <div v-if="videoError" class="error-wrapper">
      <div class="error-content">
        <p class="error-title">视频加载失败</p>
        <p class="error-message">{{ errorMessage }}</p>
        <p class="error-hint">请尝试重新上传视频或联系管理员</p>
      </div>
    </div>
    <div v-show="videoLoaded && !videoError" class="video-content">
      <video ref="player" muted autoplay @timeupdate="onTimeUpdate" @loadedmetadata="onVideoLoad" @play="onVideoPlay" @pause="onVideoPause" @error="onVideoError">
        <source :src="videoUrl" type="video/mp4">
        <source :src="videoUrl.replace('.mp4', '.avi')" type="video/x-msvideo">
        您的浏览器不支持视频播放
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

.error-wrapper {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: #f5f5f5;
  user-select: none;
}

.error-content {
  text-align: center;
  padding: 20px;
}

.error-title {
  font-size: 18px;
  font-weight: bold;
  color: #e74c3c;
  margin-bottom: 10px;
}

.error-message {
  font-size: 14px;
  color: #555;
  margin-bottom: 8px;
}

.error-hint {
  font-size: 12px;
  color: #999;
}

</style>
