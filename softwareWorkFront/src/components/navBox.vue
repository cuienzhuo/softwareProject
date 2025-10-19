<script setup>
import navIcon from './navIcon.vue';
import { ref, defineProps } from 'vue'

const props = defineProps({
  text: {
    type: String,
    default: ""
  },
  path: {
    type: String,
    default: ""
  }
})

const hoverText = (e) => {
  e.target.classList.add('active');
}

const resetText = (e) => {
  e.target.classList.remove('active');
}
</script>

<template>
  <div class="container-box">
    <!-- 左侧信息栏 -->
    <div class="info-section">
      <div>
        <div class="title" @mouseover="hoverText" @mouseout="resetText">{{ props.text }}</div>
        <navIcon text="算法学习" path="/static/folder.png" />
      </div>
      <div>
        <navIcon text="2025-7-10" path="/static/calendar.png" />
      </div>
    </div>
    
    <!-- 右侧图片 -->
    <div class="image-section">
      <div class="image-wrapper">
        <img :src="path" alt="示例图片" class="hover-zoom">
      </div>
    </div>
  </div>
</template>

<style scoped>
.container-box {
  display: flex;
  width: 860px;
  height: 230px;
  border-radius: 8px;
  overflow: hidden;
  justify-content: space-between;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
  border: 1px solid #e5e7eb;
}

/* 左侧信息栏 */
.info-section {
  width: 60%;
  padding: 20px;
  background: white;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
}

.info-section .title {
  margin-bottom: 10px;
  font-size: 28px;
}

/* 右侧图片 */
.image-section {
  width: 40%;
  overflow: hidden; /* 确保放大时不会溢出容器 */
}

.image-wrapper {
  width: 100%;
  height: 100%;
  display: flex;
  justify-content: center;
  align-items: center;
}

.hover-zoom {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
  transition: transform 0.3s ease; /* 添加过渡效果 */
}

/* 图片悬停放大效果 */
.hover-zoom:hover {
  transform: scale(1.05); /* 放大5% */
  cursor: pointer;
}

/* 文字悬停效果 */
.title:hover,
.type:hover,
.time:hover {
  color: #a855f7 !important; /* 亮紫色 */
  text-decoration: underline;
  cursor: pointer;
}

/* 或使用active类名控制（二选一） */
.active {
  color: #a855f7 !important;
  text-decoration: underline;
}
</style>