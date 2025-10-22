<script setup>

const props = defineProps({
  show: {
    required: true,
  },
  playing: {
    required: true,
  },
  progress: {
    required: true,
  },
  duration: {
    required: true,
  },
});

const durationText = computed(() => {
  const minutes = getMinutes(props.duration);
  const seconds = getSeconds(props.duration);

  return `${minutes}:${seconds}`;
});

const progressText = computed(() => {
  const minutes = getMinutes(props.progress * props.duration);
  const seconds = getSeconds(props.progress * props.duration);

  return `${minutes}:${seconds}`;
});

const getMinutes = (totalSeconds) => {
  return Math.floor(totalSeconds / 60).toString().padStart(2, '0');
};

const getSeconds = (totalSeconds) => {
  return Math.floor(totalSeconds % 60).toString().padStart(2, '0');
};

</script>

<template>
  <Transition name="control">
    <ElContainer v-if="show" class="control-container">
      <div class="play-button-wrapper">
        <ElIcon size="50" color="#e5eaf3">
          <IconEpVideoPause v-if="playing"></IconEpVideoPause>
          <IconEpVideoPlay v-else></IconEpVideoPlay>
        </ElIcon>
      </div>
      <div class="duration-wrapper">
        <ElText class="duration-text">{{progressText}} / {{durationText}}</ElText>
      </div>
    </ElContainer>
  </Transition>
</template>

<style scoped>

.control-container {
  display: block;
  position: absolute;
  width: 856px;
  height: 480px;
  margin: 0;
  border: 0;
  padding: 0;
  background-color: var(--color-dark-transparent-00);
  user-select: none;
}

.control-enter-active {
  transition: all 300ms;
}

.control-leave-active {
  transition: all 300ms;
}

.control-enter-from {
  display: block;
  opacity: 0;
}

.control-leave-from {
  display: block;
  opacity: 1;
}

.control-enter-to {
  display: block;
  opacity: 1;
}

.control-leave-to {
  display: block;
  opacity: 0;
}

.play-button-wrapper {
  position: absolute;
  bottom: 30px;
  left: 30px;
  width: 50px;
  height: 50px;
  padding: 5px;
  margin: 0;
  border: 0;
  border-radius: 30px;
  background-color: var(--color-dark-transparent-40);
}

.duration-wrapper {
  position: absolute;
  bottom: 30px;
  right: 30px;
  height: 24px;
  margin: 0;
  border: 0;
  border-radius: 4px;
  padding: 0 10px 0 10px;
  background-color: var(--color-dark-transparent-40);
}

.duration-text {
  position: relative;
  margin: 0;
  border: 0;
  padding: 0;
  color: var(--color-text-dark-2);
  line-height: 24px;
}

</style>
