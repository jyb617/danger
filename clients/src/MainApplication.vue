<script setup>

import MainMenu from './components/MainMenu.vue';

</script>

<template>
  <ElContainer class="main-container">
    <ElHeader class="app-header">
      <ElText class="app-title" size="large" tag="b">监控视频异常行为检测系统</ElText>
    </ElHeader>
    <ElContainer class="body-container">
      <ElAside class="app-aside">
        <MainMenu/>
      </ElAside>
      <ElMain class="app-main">
        <RouterView/>
      </ELMain>
    </ElContainer>
  </ElContainer>
</template>

<style scoped>

.main-container {
  display: flex;
  position: relative;
  width: 100%;
  height: 100%;
  border: 0;
  margin: 0;
  padding: 0;
  user-select: none;
}

.app-header {
  position: relative;
  width: 100%;
  height: 60px;
  border: 0;
  margin: 0;
  padding: 0 40px 0 40px;
  background-color: var(--color-primary);
}

.app-title {
  position: relative;
  width: 300px;
  height: 60px;
  border: 0;
  margin: 0;
  padding: 0;
  color: var(--color-text-dark-1);
  line-height: 60px;
}

.body-container {
  display: flex;
  position: relative;
  width: 100%;
  border: 0;
  margin: 0;
  padding: 0;
}

.app-aside {
  position: relative;
  width: 240px;
  border: 0;
  margin: 0;
  padding: 0;
  user-select: none;
}

.app-main {
  position: relative;
  border: 0;
  margin: 0;
  padding: 16px 0 0 0;
}

</style>
