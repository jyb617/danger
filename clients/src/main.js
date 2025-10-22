import {createApp} from 'vue';

import MainApplication from './MainApplication.vue';
import Router from './modules/router';

import './assets/color.css';
import './assets/main.css';
import 'element-plus/dist/index.css';

createApp(MainApplication).use(Router).mount('#app');
