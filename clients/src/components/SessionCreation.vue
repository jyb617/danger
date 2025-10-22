<script setup>

import axios from 'axios';

import IconButton from './IconButton.vue';

const creationShow = defineModel({
  required: true,
});

const sessionCreating = ref(false);
const sessionCreationFail = ref(false);

const inputName = ref('');
const inputNote = ref('');
const inputSource = ref('');

const sourceEmpty = computed(() => {
  return inputSource.value.length === 0;
});

const emitEvents = defineEmits(['success']);

const resetSessionInfo = () => {
  inputName.value = '';
  inputNote.value = '';
  inputSource.value = '';
};

const setSessionCreating = () => {
  sessionCreating.value = true;
  sessionCreationFail.value = false;
};

const unsetSessionCreating = () => {
  sessionCreating.value = false;
};

const setSessionCreationFail = () => {
  sessionCreationFail.value = true;
};

const createSession = () => {
  setSessionCreating();

  axios.post(`${window.location.origin}/api/realtimeinference/create`, {
    name: inputName.value,
    note: inputNote.value,
    source: inputSource.value,
  }).then((response) => {
    if (response.status === 200) {
      emitEvents('success', response.data.sessionId);
    } else {
      setSessionCreationFail();
    }
  }).catch(() => {
    setSessionCreationFail();
  }).finally(() => {
    unsetSessionCreating();
  });
};

</script>

<template>
  <ElDialog v-model="creationShow" width="480" align-center destroy-on-close @open="resetSessionInfo">
    <template #header>
      <ElText class="title-content" tag="b">创建实时检测会话</ElText>
    </template>
    <template #default>
      <ElContainer class="main-container">
        <ElInput v-model="inputName" type="text" size="large" placeholder="请输入会话名称" maxlength="15">
          <template #prefix>
            <IconEpCollectionTag></IconEpCollectionTag>
          </template>
        </ElInput>
        <ElInput v-model="inputSource" type="text" size="large" placeholder="请输入会话视频源">
          <template #prefix>
            <IconEpLink></IconEpLink>
          </template>
        </ElInput>
        <ElInput v-model="inputNote" :autosize="{minRows: 4}" type="textarea" placeholder="请输入会话备注"></ElInput>
      </ElContainer>
    </template>
    <template #footer>
      <ElContainer class="footer-container">
        <IconButton type="success" :loading="sessionCreating" :disabled="sourceEmpty" @click="createSession">
          <template #icon>
            <IconEpPlus></IconEpPlus>
          </template>
          <template #loading>正在创建</template>
          <template #default>创建会话</template>
        </IconButton>
        <ElText v-if="sessionCreationFail" class="creation-fail-info" size="small" type="danger">
          <span>会话创建失败！</span>
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
  height: 100%;
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

.footer-container {
  display: flex;
  position: relative;
  width: 100%;
  height: 100%;
  margin: 0;
  border: 0;
  padding: 0;
  gap: 20px;
  flex-direction: row-reverse;
  align-items: center;
}

.creation-fail-info {
  position: relative;
  margin: 0;
  border: 0;
  padding: 0;
  user-select: none;
}

</style>
