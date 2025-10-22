<template>
  <ElSkeleton class="image-skeleton" animated>
    <template #template>
      <ElSkeletonItem class="image-skeleton-item" variant="image"></ElSkeletonItem>
    </template>
  </ElSkeleton>
</template>

<style scoped>

.image-skeleton {
  position: relative;
  width: 100%;
  height: 100%;
  margin: 0;
  border: 0;
  padding: 0;
  user-select: none;
}

.image-skeleton-item {
  position: relative;
  width: 100%;
  height: 100%;
  margin: 0;
  border: 0;
  padding: 0;
  user-select: none;
}

</style>
