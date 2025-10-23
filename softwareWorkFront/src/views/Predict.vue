<script setup>
import { onMounted, ref } from 'vue'
import topNavigation from '@/components/topNavigation.vue'
import MapComponent from '@/components/mapComponent.vue'
import Selector from '@/components/selector.vue'

const rainCanvas = ref(null) 
const selectedMethod = ref("")
const predictMethods = [
  { label: '统计学算法', value: 'statistics' },
  { label: '机器学习', value: 'ML' },
  { label: '深度学习', value: 'DL' },
  { label: '大模型算法', value: 'transformer' },
];
const handleMethodChange = (option) => {
  console.log('选择了检测方法：', option);
};

onMounted(() => {
  const canvas = rainCanvas.value
  const ctx = canvas.getContext('2d')
  canvas.width = window.innerWidth
  canvas.height = window.innerHeight

  class RainDrop {
    constructor() {
      this.x = Math.random() * canvas.width
      this.y = Math.random() * -500
      this.speed = 2 + Math.random() * 5
      this.length = 10 + Math.random() * 20
    }
    draw() {
      ctx.strokeStyle = 'rgba(170, 204, 255, 0.7)'
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.moveTo(this.x, this.y)
      ctx.lineTo(this.x, this.y + this.length)
      ctx.stroke()
    }
    update() {
      this.y += this.speed
      if (this.y > canvas.height) {
        this.y = Math.random() * -100
        this.x = Math.random() * canvas.width
      }
    }
  }

  const drops = Array.from({ length: 200 }, () => new RainDrop())

  function animate() {
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    drops.forEach(drop => {
      drop.update()
      drop.draw()
    })
    requestAnimationFrame(animate)
  }
  animate()
})

</script>

<template>
  <!-- 背景 -->
    <canvas ref="rainCanvas" class="rain-canvas"></canvas>
    <!-- 上方导航栏 -->
    <div class="gradient-box">
        <topNavigation />
    </div>
  <MapComponent
  title="未来预测"
  />
  <Selector 
      title="未来预测算法："
      v-model="selectedMethod"
      :options="predictMethods"
      placeholder="请选择检测方法"
      @change="handleMethodChange"/>
    <div class="button-container">
        <button class="generate-chart-btn">
        生成图表
      </button>
    </div>
</template>

<style scoped>
.rain-canvas {
  position: fixed;
  background-color: #f5f5f5;
  top: 0;
  left: 0;
  z-index: -1;
}
.gradient-box {
    width: 100%;
    height: 7vh;
    
    /* 蓝色为主的渐变背景 */
    background: linear-gradient(to right, 
        rgba(30, 136, 229, 0.5),      /* #1e88e5 不透明 */
        rgba(66, 165, 245, 0.45),   /* #42a5f5 95%不透明 */
        rgba(100, 181, 246, 0.4),   /* #64b5f6 90%不透明 */
        rgba(144, 202, 249, 0.35),  /* #90caf9 85%不透明 */
        rgba(66, 165, 245, 0.45),   /* #42a5f5 95%不透明 */
        rgba(30, 136, 229, 0.5)       /* #1e88e5 不透明 */
    );
    
    /* 流动动画效果 */
    background-size: 200% 100%;
    animation: gradientFlow 8s ease infinite;
    
    display: flex;
    justify-content: center;
    align-items: center;
    color: white;
    font-family: 'Arial', sans-serif;
    font-weight: bold;
    text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    box-shadow: 0 8px 24px rgba(30, 136, 229, 0.3);
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
    z-index: 5;
}
/* 添加光泽效果 */
.gradient-box::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 40%;
    background: linear-gradient(to bottom, 
        rgba(255,255,255,0.25), 
        transparent);
}
/* 渐变流动动画 */
@keyframes gradientFlow {
    0% {
        background-position: 0% 50%;
    }
    50% {
        background-position: 100% 50%;
    }
    100% {
        background-position: 0% 50%;
    }
}
.button-container{
  width: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
}
.generate-chart-btn {
  /* 文字样式 */
  font-weight: bold;
  color: white;
  font-size: 24px;
  width: 150px;
  
  /* 按钮尺寸与边距 */
  padding: 12px 24px;
  border-radius: 8px;
  border: none;
  
  /* 正常状态：更深的天蓝色到浅蓝渐变（比之前深一点） */
  background: linear-gradient(135deg, #4DA6FF 0%, #E6F2FF 100%);
  
  /* 基础样式 */
  cursor: pointer;
  box-shadow: 0 3px 5px rgba(77, 166, 255, 0.25);
  
  /* 过渡动画 */
  transition: all 0.3s ease;
}

.generate-chart-btn:hover {
  /* 悬浮状态：稍浅的蓝色渐变（与正常状态差异更小） */
  background: linear-gradient(135deg, #6BA5D7 0%, #A6D1FF 100%);
  
  /* 轻微上浮效果（比之前幅度小） */
  transform: translateY(-2px);
  
  /* 柔和阴影变化 */
  box-shadow: 0 5px 8px rgba(107, 165, 215, 0.3);
}

.generate-chart-btn:active {
  /* 点击状态下沉 */
  transform: translateY(0);
  box-shadow: 0 2px 3px rgba(77, 166, 255, 0.2);
}

</style>