<script setup>
import { onMounted, ref } from 'vue'
import topNavigation from '@/components/topNavigation.vue'
import navBox from '@/components/navBox.vue'

const rainCanvas = ref(null)

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

</style>