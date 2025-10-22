<script setup>

import axios from 'axios';

import IconButton from '../components/IconButton.vue';
import MainFramework from '../components/MainFramework.vue';
import ConfirmOption from '../components/ConfirmOption.vue';
import PaginationLayout from '../components/PaginationLayout.vue';

import SessionCreation from '../components/SessionCreation.vue';
import SessionDetail from '../components/SessionDetail.vue';
import SessionOverview from '../components/SessionOverview.vue';

const requestingSessions = ref(false);
const selectionMode = ref(false);
const synchronizingSessions = ref(false);

const sessions = ref([]);
const selectedSessionIds = ref([]);

const currentId = ref(0);
const pageNumber = ref(1);
const pageLength = ref(8);
const totalCount = ref(0);

const creationShow = ref(false);
const detailShow = ref(false);
const confirmShow = ref(false);

const setRequestingSessions = () => {
  requestingSessions.value = true;
};

const unsetRequestingSessions = () => {
  requestingSessions.value = false;
};

const updateSessionList = () => {
  setRequestingSessions();

  axios.post(`${window.location.origin}/api/realtimeinference/list`, {
    pageNumber: pageNumber.value,
    pageLength: pageLength.value,
  }).then((response) => {
    if (response.status === 200) {
      sessions.value = response.data.sessions;

      if (response.data.totalCount <= 0) {
        totalCount.value = 0;
      } else {
        totalCount.value = response.data.totalCount;
      }
    }
  }).finally(() => {
    unsetRequestingSessions();
  });
};

const onPageChange = (newPageNumber, newPageLength) => {
  pageNumber.value = newPageNumber;
  pageLength.value = newPageLength;
  updateSessionList();
};

const onCreationClick = () => {
  detailShow.value = false;
  creationShow.value = true;
};

const onCreationSuccess = (sessionId) => {
  updateSessionList();
  showSessionDetail(sessionId);
};

const showSessionDetail = (sessionId) => {
  currentId.value = sessionId;
  detailShow.value = true;
  creationShow.value = false;
};

const enterSelectionMode = () => {
  selectionMode.value = true;
};

const leaveSelectionMode = () => {
  selectedSessionIds.value = [];
  selectionMode.value = false;
};

const showDeleteConfirm = () => {
  if (selectedSessionIds.value.length > 0) {
    confirmShow.value = true;
  }
};

const selectAllSessions = () => {
  sessions.value.forEach((session) => {
    if (selectedSessionIds.value.includes(session.sessionId)) {
      return;
    }
    selectedSessionIds.value.push(session.sessionId);
  });
};

const unselectAllSessions = () => {
  selectedSessionIds.value = [];
};

const setSynchronizingSessions = () => {
  synchronizingSessions.value = true;
};

const unsetSynchronizingSessions = () => {
  synchronizingSessions.value = false;
};

const synchronizeSessions = () => {
  setSynchronizingSessions();

  axios.get(`${window.location.origin}/api/realtimeinference/sync`).finally(() => {
    unsetSynchronizingSessions();
  });
};

const deleteSelectedSessions = () => {
  axios.post(`${window.location.origin}/api/realtimeinference/delete`, {
    sessionIds: selectedSessionIds.value,
  }).then((response) => {
    if (response.status === 200) {
      updateSessionList();
      leaveSelectionMode();
    }
  });
};

onMounted(() => {
  updateSessionList();
});

</script>

<template>
  <MainFramework>
    <template #title>
      <ElText tag="b">实时检测会话列表</ElText>
    </template>
    <template #option v-if="selectionMode">
      <IconButton type="primary" plain @click="selectAllSessions">
        <template #icon>
          <IconEpCheck></IconEpCheck>
        </template>
        <template #default>全选</template>
      </IconButton>
      <IconButton type="primary" plain @click="unselectAllSessions">
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
      <IconButton type="warning" :loading="synchronizingSessions" plain @click="synchronizeSessions">
        <template #icon>
          <IconEpSort></IconEpSort>
        </template>
        <template #loading>正在同步</template>
        <template #default>同步会话</template>
      </IconButton>
      <IconButton type="primary" :loading="requestingSessions" plain @click="updateSessionList">
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
        <template #default>创建会话</template>
      </IconButton>
      <IconButton type="primary" plain @click="enterSelectionMode">
        <template #icon>
          <IconEpCheck></IconEpCheck> 
        </template>
        <template #default>选择会话</template>
      </IconButton>
    </template>
    <template #body>
      <ElCheckboxGroup class="checkbox-group" v-model="selectedSessionIds">
        <PaginationLayout :disabled="requestingSessions" :length="pageLength" :total="totalCount" @change="onPageChange">
          <SessionOverview v-for="session in sessions" :session="session" :selectable="selectionMode" @enter="showSessionDetail"/>
        </PaginationLayout>
      </ElCheckboxGroup>
    </template>
  </MainFramework>
  <SessionCreation v-model="creationShow" @success="onCreationSuccess"/>
  <ConfirmOption v-model="confirmShow" @confirm="deleteSelectedSessions">
    <template #icon>
      <ElIcon size="40px" color="#F56C6C">
        <IconEpWarningFilled></IconEpWarningFilled>
      </ElIcon>
    </template>
    <template #text>
      <ElText>是否确认删除所选的 {{selectedSessionIds.length}} 个会话？此操作不可撤销！</ElText>
    </template>
  </ConfirmOption>
  <SessionDetail v-model="detailShow" :session-id="currentId"/>
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
