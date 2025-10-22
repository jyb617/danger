<script setup>

import axios from 'axios';

import IconButton from './IconButton.vue';

const creationShow = defineModel({
  required: true,
});

const hiddenInputComponent = useTemplateRef('input');

const videoDetecting = ref(false);
const videoDetectionFail = ref(false);

const inputName = ref('');
const inputNote = ref('');
const inputVideo = ref(null);

const emitEvents = defineEmits(['success']);

const resetVideoInfo = () => {
  videoDetectionFail.value = false;

  inputName.value = '';
  inputNote.value = '';

  inputVideo.value = null;
};

const onSelectClick = () => {
  hiddenInputComponent.value.click();
};

const onSelectChange = (event) => {
  const selectedFiles = event.target.files;

  if (selectedFiles.length > 0) {
    inputVideo.value = selectedFiles[0];
  }
};

const setVideoDetecting = () => {
  videoDetecting.value = true;
  videoDetectionFail.value = false;
};

const unsetVideoDetecting = () => {
  videoDetecting.value = false;
};

const setVideoDetectionFail = () => {
  videoDetectionFail.value = true;
};

const createVideoDetection = () => {
  setVideoDetecting();

  axios.postForm(`${window.location.origin}/api/videoinference`, {
    name: inputName.value,
    note: inputNote.value,
    video: inputVideo.value,
  }).then((response) => {
    if (response.status === 200) {
      emitEvents('success', response.data.videoId);
    } else {
      setVideoDetectionFail();
    }
  }).catch(() => {
    setVideoDetectionFail();
  }).finally(() => {
    unsetVideoDetecting();
  });
};

</script>

<template>
  <ElDialog v-model="creationShow" width="480" align-center destroy-on-close @open="resetVideoInfo">
    <template #header>
      <ElText class="title-content" tag="b">创建视频检测</ElText>
    </template>
    <template #default>
      <ElContainer class="main-container">
        <ElInput v-model="inputName" type="text" size="large" placeholder="请输入视频名称" maxlength="15">
          <template #prefix>
            <IconEpCollectionTag></IconEpCollectionTag>
          </template>
        </ElInput>
        <ElInput v-model="inputNote" :autosize="{minRows: 4}" type="textarea" placeholder="请输入视频备注"></ElInput>
        <ElContainer class="select-container">
          <IconButton type="primary" @click="onSelectClick">
            <template #icon>
              <IconEpDocument></IconEpDocument>
            </template>
            <template #default>选择视频文件</template>
          </IconButton>
          <ElText v-if="inputVideo" size="small">已选择的视频文件：{{inputVideo.name}}</ElText>
          <ElText v-else size="small">未选择视频文件</ElText>
        </ElContainer>
      </ElContainer>
      <input class="hidden-input" ref="input" type="file" accept="video/mp4" @change="onSelectChange">
    </template>
    <template #footer>
      <ElContainer class="footer-container">
        <IconButton type="success" :loading="videoDetecting" :disabled="!inputVideo" @click="createVideoDetection">
          <template #icon>
            <IconEpUpload></IconEpUpload>
          </template>
          <template #loading>正在检测</template>
          <template #default>开始检测</template>
        </IconButton>
        <ElText v-if="videoDetectionFail" class="detection-fail-info" size="small" type="danger">
          <span>视频检测失败！</span>
        </ElText>
      </ElContainer>
    </template>
  </ElDialog>
</template>

<style scoped>

.main-container {
  display: flex;
  position: relative;
  width: 100%;
  margin: 0;
  border: 0;
  padding: 0;
  gap: 20px;
  flex-direction: column;
}

.title-content {
  position: relative;
  margin: 0;
  border: 0;
  padding: 0;
  user-select: none;
}

.hidden-input {
  display: none;
}

.select-container {
  display: flex;
  position: relative;
  width: 100%;
  margin: 0;
  border: 0;
  padding: 0;
  gap: 20px;
  flex-direction: row;
}

.footer-container {
  display: flex;
  position: relative;
  margin: 0;
  border: 0;
  padding: 0;
  gap: 20px;
  flex-direction: row-reverse;
  align-items: center;
}

.detection-fail-info {
  position: relative;
  margin: 0;
  border: 0;
  padding: 0;
  user-select: none;
}

</style>
