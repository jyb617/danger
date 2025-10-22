<script setup>

import axios from 'axios';

import VideoWidget from './VideoWidget.vue';

const props = defineProps({
  videoId: {
    required: true,
  },
});

const detailShow = defineModel({
  required: true,
});

const videoName = ref('');
const videoNote = ref('');
const videoTime = ref('');
const anomalyScores = ref([]);

const videoUrl = computed(() => {
  return `${window.location.origin}/api/videoinference/video/${props.videoId}`;
});

const getVideoDetail = () => {
  axios.get(`${window.location.origin}/api/videoinference/detail/${props.videoId}`).then((response) => {
    if (response.status === 200) {
      videoName.value = response.data.name;
      videoNote.value = response.data.note;
      videoTime.value = response.data.time;
      anomalyScores.value = response.data.scores;
    }
  });
};

</script>

<template>
  <ElDialog v-model="detailShow" width="900" align-center destroy-on-close @open="getVideoDetail">
    <template #header>
      <div class="header-wrapper">
        <ElPopover placement="bottom-start" trigger="hover">
          <template #reference>
            <ElText tag="b">{{videoName}}</ElText>
          </template>
          <template #default>
            <ElText size="small">{{videoNote}}</ElText>
          </template>
        </ElPopover>
        <ElText size="small">{{videoTime}}</ElText>
      </div>
    </template>
    <template #default>
      <div class="widget-wrapper">
        <VideoWidget :video-url="videoUrl" :scores="anomalyScores"/>
      </div>
    </template>
  </ElDialog>
</template>

<style scoped>

.header-wrapper {
  display: flex;
  position: relative;
  width: 100%;
  height: 100%;
  margin: 0;
  border: 0;
  padding: 0 6px 0 6px;
  gap: 16px;
  user-select: none;
}

.widget-wrapper {
  position: relative;
  width: 856px;
  height: 518px;
  margin: 0;
  border: 0;
  padding: 6px;
  overflow: hidden;
}

</style>
