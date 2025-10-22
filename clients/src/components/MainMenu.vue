<template>
  <ElMenu class="app-aside-menu" text-color="#e5eaf3" active-text-color="#66b1ff" :default-active="$route.name" router>
    <ElMenuItem class="menu-item" index="videos" route="/">
      <ElIcon>
        <IconEpFiles></IconEpFiles>
      </ElIcon>
      <span>视频异常检测</span>
    </ElMenuItem>
    <ElMenuItem class="menu-item" index="sessions" route="/session">
      <ElIcon>
        <IconEpVideoCamera></IconEpVideoCamera>
      </ElIcon>
      <span>实时异常检测</span>
    </ElMenuItem>
  </ElMenu>
</template>

<style scoped>

.app-aside-menu {
  position: relative;
  width: 100%;
  height: 100%;
  border: 0;
  margin: 0;
  padding: 20px 0 20px 0;
  background-color: var(--color-primary-dark-8);
}

.menu-item {
  position: relative;
  width: 100%;
  margin: 0;
  border: 0;
  padding: 0;
  background-color: var(--color-primary-dark-8);
}

.menu-item:hover {
  background-color: var(--color-primary-dark-9);
}

</style>
