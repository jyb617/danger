<template>
  <ElContainer class="main-container">
    <ElContainer class="header-container">
      <ElContainer class="title-container">
        <slot name="title"></slot>
      </ElContainer>
      <ElContainer class="option-container">
        <slot name="option"></slot>
      </ElContainer>
    </ElContainer>
    <ElContainer class="body-container">
      <slot name="body"></slot>
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
  flex-direction: column;
}

.header-container {
  display: flex;
  position: relative;
  height: 50px;
  margin: 0;
  border: 0;
  padding: 0 50px 10px 50px;
  flex-grow: 1;
  user-select: none;
}

.title-container {
  position: relative;
  height: 40px;
  margin: 0;
  border: 0;
  padding: 0;
  flex-grow: 1;
  line-height: 40px;
}

.option-container {
  display: flex;
  position: relative;
  height: 40px;
  margin: 0;
  border: 0;
  padding: 0;
  flex-grow: 0;
  gap: 12px;
  align-items: center;
}

.body-container {
  position: relative;
  width: 100%;
  height: 100%;
  border: 0;
  margin: 0;
  padding: 0;
  user-select: none;
}

</style>
