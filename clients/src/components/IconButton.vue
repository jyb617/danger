<script setup>

const props = defineProps({
  type: {
    required: false,
  },
  size: {
    required: false,
  },
  plain: {
    required: false,
  },
  loading: {
    required: false,
  },
  disabled: {
    required: false,
  },
});

</script>

<template>
  <div>
    <ElButton :type="type" :size="size" :loading="loading" :plain="plain" :disabled="disabled">
      <div v-if="loading">
        <span>
          <slot name="loading"></slot>
        </span>
      </div>
      <div v-else>
        <ElIcon>
          <slot name="icon"></slot>
        </ElIcon>
        <span>
          <slot name="default"></slot>
        </span>
      </div>
    </ElButton>
  </div>
</template>
