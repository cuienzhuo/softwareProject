<template>
  <div class="magic-box" @mouseover="isHovered = true" @mouseleave="isHovered = false" @click="navigateTo">
    {{ text }}
  </div>
</template>

<script setup>
import { ref, defineProps } from 'vue'
import { useRouter } from 'vue-router'

const router = useRouter()
const props = defineProps({
    text: {
        type: String,
        default:""
  },
  url: {
    type: String,
      default:""
    }
})

const navigateTo = () => {
  router.push(props.url)
}
const isHovered = ref(false)
</script>

<style scoped>
.magic-box {
  /* 基础样式 */
  width: 160px;
  height: 80px;
  border: 2px solid rgba(255, 255, 255, 0.3);
  border-radius: 10px;
  display: flex;
  justify-content: center;
  align-items: center;
  font-size: 18px;
  color: white;
  text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
  cursor: pointer;
  transition: all 0.4s ease;
  position: relative;
  overflow: hidden;
  z-index: 1;

  /* 模糊背景效果 */
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(8px);
  -webkit-backdrop-filter: blur(8px);
}

/* 悬浮效果 */
.magic-box:hover {
  color: black;
  text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.5);
  border-color: #c792ea;
  
  /* 亮紫色背景覆盖层 */
  &::before {
    opacity: 1;
  }
}

/* 背景色过渡层 */
.magic-box::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: #c792ea;
  opacity: 0;
  z-index: -1;
  transition: opacity 0.4s ease;
}

/* 悬浮时的小动画 */
.magic-box:hover {
  transform: translateY(-3px);
  box-shadow: 0 10px 20px rgba(199, 146, 234, 0.3);
}
</style>