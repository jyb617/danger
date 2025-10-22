<script setup>

import axios from 'axios';

import ImageSkeleton from './ImageSkeleton.vue';

const props = defineProps({
  sessionId: {
    required: true,
  },
});

const detailShow = defineModel({
  required: true,
});

const sessionLoaded = ref(false);

const sessionName = ref('');
const sessionNote = ref('');
const sessionSource = ref('');

const sessionUrl = computed(() => {
  return `${window.location.origin}/api/realtimeinference/session/${props.sessionId}`;
});

const setSessionLoaded = () => {
  sessionLoaded.value = true;
};

const unsetSessionLoaded = () => {
  sessionLoaded.value = false;
};

const getSessionDetail = () => {
  unsetSessionLoaded();

  axios.get(`${window.location.origin}/api/realtimeinference/detail/${props.sessionId}`).then((response) => {
    if (response.status === 200) {
      sessionName.value = response.data.name;
      sessionNote.value = response.data.note;
      sessionSource.value = response.data.source;
    }
  });
};

</script>

<template>
  <ElDialog v-model="detailShow" width="900" align-center destroy-on-close @open="getSessionDetail">
    <template #header>
      <div class="header-wrapper">
        <ElPopover placement="bottom-start" trigger="hover">
          <template #reference>
            <ElText tag="b">{{sessionName}}</ElText>
          </template>
          <template #default>
            <ElText size="small">{{sessionNote}}</ElText>
          </template>
        </ElPopover>
        <ElText size="small">{{sessionSource}}</ElText>
      </div>
    </template>
    <template #default>
      <div class="main-wrapper">
        <div class="session-wrapper">
          <div v-if="!sessionLoaded" class="skeleton-wrapper">
            <ImageSkeleton/>
          </div>
          <img v-show="sessionLoaded" class="session-content" draggable="false" :src="sessionUrl" @load="setSessionLoaded">
        </div>
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

.main-wrapper {
  position: relative;
  width: 856px;
  margin: 0;
  border: 0;
  padding: 6px;
}

.session-wrapper {
  position: relative;
  width: 856px;
  height: 480px;
  margin: 0;
  border: 0;
  padding: 0;
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
}

</style>
