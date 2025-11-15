<template>
  <div class="page-container">
    <!-- 标题栏 -->
    <header class="title-bar">
      <h1 class="title">米兰地区流量分析</h1>
    </header>

    <!-- 内容区域 -->
    <main class="content">

      <nav class="tab-nav">
        <button
          v-for="tab in tabs"
          :key="tab.key"
          :class="{ active: currentTab === tab.key }"
          @click="currentTab = tab.key"
        >
          {{ tab.label }}
        </button>
      </nav>

      <div class="tab-content">
        <component :is="currentComponent" :regions="regions" />
      </div>
    </main>
  </div>

  <!-- 雨滴 Canvas 放在最后，确保渲染在最上层 -->
  <canvas ref="rainCanvas" class="rain-overlay"></canvas>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted,onBeforeMount } from 'vue'
import AbnormalDataView from '@/components/AbnormalDataView.vue'
import AbnormalCompareView from '@/components/AbnormalCompareView.vue'
import ClusteringView from '@/components/ClusteringView.vue'
import ForecastView from '@/components/ForecastView.vue'
import ForecastCompareView from '@/components/ForecastCompareView.vue'
import api from '@/api'

const tabs = [
  { key: 'abnormal-data', label: '异常数据分析', component: AbnormalDataView },
  { key: 'abnormal-compare', label: '异常分析比较', component: AbnormalCompareView },
  { key: 'clustering', label: '聚类分析', component: ClusteringView },
  { key: 'forecast', label: '未来预测', component: ForecastView },
  { key: 'forecast-compare', label: '未来预测比较', component: ForecastCompareView }
]

const currentTab = ref(tabs[0].key)

const regions = ref([])

const currentComponent = computed(() => {
  const tab = tabs.find(t => t.key === currentTab.value)
  return tab ? tab.component : null
})

const rainCanvas = ref(null)
let animationId = null

onBeforeMount(async() => {
  const response = await api.get("/api/get_milan_columns/")
  const data = response.data
  if (data.code === 200) {
    regions.value = data.columns
  }
  console.log(regions.value)
})

onMounted(() => {
  const canvas = rainCanvas.value
  if (!canvas) return

  const ctx = canvas.getContext('2d')
  canvas.width = window.innerWidth
  canvas.height = window.innerHeight

  const handleResize = () => {
    canvas.width = window.innerWidth
    canvas.height = window.innerHeight
  }
  window.addEventListener('resize', handleResize)

  class Raindrop {
    constructor() {
      this.reset()
    }

    reset() {
      this.x = Math.random() * canvas.width
      this.y = Math.random() * -canvas.height
      this.speed = 2 + Math.random() * 5
      this.length = 10 + Math.random() * 15
      this.thickness = 1 + Math.random() * 2
      this.opacity = 0.2 + Math.random() * 0.4 // 稍微调低一点，避免太遮挡
    }

    fall() {
      this.y += this.speed
      if (this.y > canvas.height) {
        this.reset()
      }
    }

    draw() {
      ctx.beginPath()
      ctx.moveTo(this.x, this.y)
      ctx.lineTo(this.x, this.y + this.length)
      ctx.strokeStyle = `rgba(173, 216, 230, ${this.opacity})`
      ctx.lineWidth = this.thickness
      ctx.stroke()
    }
  }

  const drops = Array.from({ length: 250 }, () => new Raindrop())

  const animate = () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    drops.forEach(drop => {
      drop.fall()
      drop.draw()
    })
    animationId = requestAnimationFrame(animate)
  }

  animate()

  onUnmounted(() => {
    window.removeEventListener('resize', handleResize)
    if (animationId) cancelAnimationFrame(animationId)
  })
})
</script>

<style scoped>
.page-container {
  display: flex;
  flex-direction: column;
  height: 100vh;
  font-family: Arial, sans-serif;
  position: relative;
  /* 不再需要 z-index，因为 canvas 在外部且层级更高 */
}

/* 标题栏：滚动渐变 */
.title-bar {
  padding: 20px;
  text-align: center;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
  background: linear-gradient(90deg, 
    #6495ED, /* 淡蓝色 */
    #7B68EE, /* 中等蓝紫色 */
    #BA55D3, /* 中紫罗兰色 */
    #8A2BE2, /* 较深一点的紫色，用于加深层次感 */
    #6495ED  /* 回到淡蓝色，完成循环 */
);
  background-size: 300% 100%;
  animation: gradientScroll 8s linear infinite;
}

@keyframes gradientScroll {
  0% { background-position: 0% 50%; }
  100% { background-position: 100% 50%; }
}

.title {
  color: white;
  font-weight: bold;
  margin: 0;
  font-size: 30px;
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
}

.content {
  flex: 1;
  padding: 20px 100px;
  display: flex;
  flex-direction: column;
  gap: 20px;
  /* 移除了半透明背景，让雨滴覆盖更真实 */
}

/* 其他样式保持不变 */
.region-selector {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.region-selector select {
  padding: 6px 10px;
  border: 1px solid #ccc;
  border-radius: 4px;
}
.tab-nav {
  display: flex;
  gap: 10px;
  border-bottom: 1px solid #ddd;
  padding-bottom: 10px;
}
.tab-nav button {
  padding: 8px 16px;
  background: none;
  border: 1px solid transparent;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
  color: #555;
  transition: all 0.2s;
}
.tab-nav button:hover {
  background-color: #f0f0f0;
}
.tab-nav button.active {
  color: #1e90ff;
  border-color: #1e90ff;
  background-color: #e6f2ff;
  font-weight: bold;
}
.tab-content h2 {
  margin: 0;
  color: #333;
}

/* ✅ 关键：雨滴 overlay 样式 */
.rain-overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  z-index: 9999; /* 最顶层 */
  pointer-events: none; /* ⚠️ 允许点击穿透到下方 UI */
  /* 可选：混合模式让雨滴更融合 */
  /* mix-blend-mode: screen; */
}
</style>