<script setup>

const confirmShow = defineModel({
  required: true,
});

const emitEvents = defineEmits(['confirm']);

const onCancelClick = () => {
  if (confirmShow.value) {
    confirmShow.value = false;
  }
};

const onConfirmClick = () => {
  if (confirmShow.value) {
    confirmShow.value = false;
  }
  emitEvents('confirm');
};

</script>

<template>
  <ElDialog v-model="confirmShow" width="450" align-center>
    <template #default>
      <ElContainer class="main-container">
        <div class="icon-wrapper">
          <slot name="icon"></slot>
        </div>
        <div class="text-wrapper">
          <slot name="text"></slot>
        </div>
      </ElContainer>
    </template>
    <template #footer>
      <ElButton type="primary" plain @click="onCancelClick">
        <span>取消</span>
      </ElButton>
      <ElButton type="primary" @click="onConfirmClick">
        <span>确认</span>
      </ElButton>
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
  padding: 20px;
  gap: 20px;
  user-select: none;
}

.icon-wrapper {
  display: flex;
  position: relative;
  margin: 0;
  border: 0;
  padding: 0;
  align-items: center;
}

.text-wrapper {
  display: flex;
  position: relative;
  margin: 0;
  border: 0;
  padding: 0;
  align-items: center;
}

</style>
