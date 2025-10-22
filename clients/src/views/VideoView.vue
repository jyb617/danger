<script setup>

import axios from 'axios';

import IconButton from '../components/IconButton.vue';
import MainFramework from '../components/MainFramework.vue';
import ConfirmOption from '../components/ConfirmOption.vue';
import PaginationLayout from '../components/PaginationLayout.vue';

import VideoCreation from '../components/VideoCreation.vue';
import VideoDetail from '../components/VideoDetail.vue';
import VideoOverview from '../components/VideoOverview.vue';

const selectionMode = ref(false);
const requestingVideos = ref(false);

const videos = ref([]);
const selectedVideoIds = ref([]);

const currentId = ref(0);
const pageNumber = ref(1);
const pageLength = ref(8);
const totalCount = ref(0);

const creationShow = ref(false);
const detailShow = ref(false);
const confirmShow = ref(false);

const setRequestingVideos = () => {
  requestingVideos.value = true;
};

const unsetRequestingVideos = () => {
  requestingVideos.value = false;
};

const updateVideoList = () => {
  setRequestingVideos();
  
  axios.post(`${window.location.origin}/api/videoinference/list`, {
    pageNumber: pageNumber.value,
    pageLength: pageLength.value,
  }).then((response) => {
    if (response.status === 200) {
      videos.value = response.data.videos;

      if (response.data.totalCount <= 0) {
        totalCount.value = 0;
      } else {
        totalCount.value = response.data.totalCount;
      }
    }
  }).finally(() => {
    unsetRequestingVideos();
  });
};

const onPageChange = (newPageNumber, newPageLength) => {
  pageNumber.value = newPageNumber;
  pageLength.value = newPageLength;
  updateVideoList();
};

const onCreationClick = () => {
  detailShow.value = false;
  creationShow.value = true;
};

const onDetectionSuccess = (videoId) => {
  updateVideoList();
  showVideoDetail(videoId);
};

const showVideoDetail = (videoId) => {
  currentId.value = videoId;
  detailShow.value = true;
  creationShow.value = false;
};

const enterSelectionMode = () => {
  selectionMode.value = true;
};

const leaveSelectionMode = () => {
  selectedVideoIds.value = [];
  selectionMode.value = false;
};

const showDeleteConfirm = () => {
  if (selectedVideoIds.value.length > 0) {
    confirmShow.value = true;
  }
};

const selectAllVideos = () => {
  videos.value.forEach((video) => {
    if (selectedVideoIds.value.includes(video.videoId)) {
      return;
    }
    selectedVideoIds.value.push(video.videoId);
  });
};

const unselectAllVideos = () => {
  selectedVideoIds.value = [];
};

const deleteSelectedVideos = () => {
  axios.post(`${window.location.origin}/api/videoinference/delete`, {
    videoIds: selectedVideoIds.value,
  }).then((response) => {
    if (response.status === 200) {
      updateVideoList();
      leaveSelectionMode();
    }
  });
};

onMounted(() => {
  updateVideoList();
});

</script>

<template>
  <MainFramework>
    <template #title>
      <ElText tag="b">视频检测列表</ElText>
    </template>
    <template #option v-if="selectionMode">
      <IconButton type="primary" plain @click="selectAllVideos">
        <template #icon>
          <IconEpCheck></IconEpCheck>
        </template>
        <template #default>全选</template>
      </IconButton>
      <IconButton type="primary" plain @click="unselectAllVideos">
        <template #icon>
          <IconEpClose></IconEpClose>
        </template>
        <template #default>取消全选</template>
      </IconButton>
      <IconButton type="danger" plain @click="showDeleteConfirm">
        <template #icon>
          <IconEpDelete></IconEpDelete>
        </template>
        <template #default>删除所选</template>
      </IconButton>
      <IconButton type="primary" plain @click="leaveSelectionMode">
        <template #icon>
          <IconEpBack></IconEpBack>
        </template>
        <template #default>退出选择</template>
      </IconButton>
    </template>
    <template #option v-else>
      <IconButton type="primary" :loading="requestingVideos" plain @click="updateVideoList">
        <template #icon>
          <IconEpRefresh></IconEpRefresh>
        </template>
        <template #loading>正在刷新</template>
        <template #default>刷新列表</template>
      </IconButton>
      <IconButton type="primary" plain @click="onCreationClick">
        <template #icon>
          <IconEpPlus></IconEpPlus>
        </template>
        <template #default>创建检测</template>
      </IconButton>
      <IconButton type="primary" plain @click="enterSelectionMode">
        <template #icon>
          <IconEpCheck></IconEpCheck> 
        </template>
        <template #default>选择视频</template>
      </IconButton>
    </template>
    <template #body>
      <ElCheckboxGroup class="checkbox-group" v-model="selectedVideoIds">
        <PaginationLayout :disabled="requestingVideos" :length="pageLength" :total="totalCount" @change="onPageChange">
          <VideoOverview v-for="video in videos" :video="video" :selectable="selectionMode" @enter="showVideoDetail"/>
        </PaginationLayout>
      </ElCheckboxGroup>
    </template>
  </MainFramework>
  <VideoCreation v-model="creationShow" @success="onDetectionSuccess"/>
  <ConfirmOption v-model="confirmShow" @confirm="deleteSelectedVideos">
    <template #icon>
      <ElIcon size="40px" color="#F56C6C">
        <IconEpWarningFilled></IconEpWarningFilled>
      </ElIcon>
    </template>
    <template #text>
      <ElText>是否确认删除所选的 {{selectedVideoIds.length}} 个视频？此操作不可撤销！</ElText>
    </template>
  </ConfirmOption>
  <VideoDetail v-model="detailShow" :video-id="currentId"/>
</template>

<style scoped>

.checkbox-group {
  position: relative;
  width: 100%;
  height: 100%;
  margin: 0;
  border: 0;
  padding: 0;
  user-select: none;
}

</style>
