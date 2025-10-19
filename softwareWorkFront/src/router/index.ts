import { createRouter, createWebHistory } from 'vue-router'
import MainPage from '@/views/MainPage.vue'
import ExceptionAnalysis from '@/views/ExceptionAnalysis.vue'
import ClusterAnalysis from '@/views/ClusterAnalysis.vue'
import Predict from '@/views/Predict.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      component: MainPage,
    },
    {
      path: '/MainPage',
      component:MainPage,
    },
    {
      path: "/ExceptionAnalysis",
      component:ExceptionAnalysis
    },
    {
      path: "/ClusterAnalysis",
      component:ClusterAnalysis
    },
    {
      path: "/Predict",
      component:Predict
    }
  ],
})

export default router
