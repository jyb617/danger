<script setup>

const props = defineProps({
  length: {
    required: true,
  },
  total: {
    required: true,
  },
  disabled: {
    required: false,
  },
});

const emitEvents = defineEmits(['change']);

const onPageChange = (pageNumber, pageLength) => {
  if (pageNumber <= 0) {
    return;
  }
  if (pageLength <= 0) {
    return;
  }
  emitEvents('change', pageNumber, pageLength);
};

</script>

<template>
  <div class="list-container">
    <ElContainer class="list-center-container">
      <ElContainer class="list-content-container">
        <slot></slot>
      </ElContainer>
    </ElContainer>
    <ElContainer class="pagination-center-container">
      <div class="pagination-container">
        <ElPagination background :disabled="disabled" layout="->, prev, pager, next" :page-size="length" :total="total" @change="onPageChange"/>
      </div>
    </ElContainer>
  </div>
</template>

<style scoped>

.list-container {
  position: relative;
  width: 100%;
  height: 100%;
  min-width: 1224px;
  min-height: 588px;
  border: 0;
  margin: 0;
  padding: 0;
  overflow: scroll;
}

.list-center-container {
  display: flex;
  position: relative;
  width: 100%;
  height: 536px;
  border: 0;
  margin: 0;
  padding: 0;
  justify-content: center;
}

.list-content-container {
  display: flex;
  position: relative;
  width: 1224px;
  height: 520px;
  margin: 0;
  border: 0;
  padding: 0;
  flex-grow: unset;
  flex-wrap: wrap;
  user-select: none;
}

.pagination-center-container {
  display: flex;
  position: relative;
  width: 100%;
  height: 32px;
  margin: 0;
  border: 0;
  padding: 0;
  justify-content: center;
}

.pagination-container {
  position: relative;
  width: 1204px;
  height: 32px;
  margin: 0;
  border: 0;
  padding: 0;
  user-select: none;
}

</style>
