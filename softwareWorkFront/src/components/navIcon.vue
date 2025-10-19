<template>
  <div class="icon-text-item" @mouseover="isHovered = true" @mouseout="isHovered = false">
    <!-- 左侧图标 -->
    <img 
      class="icon" 
      :class="{ 'active': isHovered }"
      :src="path" 
      alt="图标"
    >
    <!-- 右侧文字 -->
    <span class="text" :class="{ 'active': isHovered }">
      {{ text }}
    </span>
  </div>
</template>

<script setup>
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
const isHovered = ref(false)
</script>

<style scoped>
.icon-text-item {
  /* 布局样式 */
  display: inline-flex;
  align-items: center;
  position: relative; /* 为下划线伪元素定位 */
  gap: 8px;
  padding-bottom: 2px; /* 为下划线留空间 */
  cursor: pointer;
  
  /* 基础样式 */
  transition: all 0.3s ease;
}

/* 图标样式 */
.icon {
  width: 20px;
  height: 20px;
  object-fit: contain;
  filter: grayscale(100%) opacity(0.7); /* 默认灰色半透明 */
  transition: inherit;
}

/* 文字默认样式 */
.text {
  color: #999;
  font-size: 14px;
  transition: inherit;
}

.icon-text-item:hover .text,
.text.active{
color: #a855f7;
}
/* 悬停时整体效果 */
.icon-text-item:hover .icon,
.icon.active {
  color: #a855f7;
  /* 使用滤镜将图标变成亮紫色 */
  filter: grayscale(0%) brightness(0.9) sepia(1) hue-rotate(250deg) saturate(4);
}

/* 贯穿图标和文字的下划线 */
.icon-text-item:hover::after {
  content: '';
  position: absolute;
  left: 0;
  right: 0;
  bottom: 0;
  height: 2px;
  background: #a855f7;
}
</style>