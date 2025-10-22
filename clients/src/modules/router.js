import {createRouter, createWebHistory} from 'vue-router';

import VideoView from '../views/VideoView.vue';
import SessionView from '../views/SessionView.vue';

export default createRouter({
  routes: [
    {
      name: 'videos',
      path: '/',
      component: VideoView,
    },
    {
      name: 'sessions',
      path: '/session',
      component: SessionView,
    },
  ],
  history: createWebHistory(),
});
