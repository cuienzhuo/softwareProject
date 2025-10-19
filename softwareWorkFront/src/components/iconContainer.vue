<template>
  <div class="icon-text-container">
    <div 
      class="icon-text-item"
      @mouseenter="isHovered = true"
      @mouseleave="isHovered = false"
      
      @click="navigateTo"
    >
      <!-- 将图标和文本包裹在一个统一的下划线容器中 -->
      <div 
        class="content-wrapper"
        :style="{
          textDecoration: isHovered ? 'underline' : 'none',
          textUnderlineOffset: '4px',
          textDecorationColor: activeColor
        }"
      >
        <img
          class="icon"
          :src="path"
          :style="{ filter: isHovered ? activeFilter : defaultFilter }"
          alt="图标"
        />
        <div class="text" :style="{ color: isHovered ? activeColor : defaultColor }">
          {{ text }}
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, defineProps } from 'vue'
import { useRouter } from 'vue-router'
const props = defineProps({
  text: {
    type: String,
    default: "首页"
  },
  path: {
    type: String,
    default:""
  },
  url: {
    type: String,
    default:""
  }
})
const router = useRouter()
const isHovered = ref(false)
const defaultColor = "#ffffff"
const activeColor = "#c792ea"
const defaultFilter = "brightness(0) invert(1)"
const activeFilter = "brightness(0) invert(0.7) sepia(1) hue-rotate(250deg)"

const navigateTo = () => {
  router.push(props.url)
}
</script>

<style scoped>
::selection{
  background: rgba(199, 146, 234, 0.3);
  color: white; 
}
.icon-text-container {
  padding: 12px;
}

.icon-text-item {
  cursor: pointer;
}

.content-wrapper {
  display: flex;
  align-items: center;
  gap: 8px;
  transition: all 0.3s ease;
}

.icon {
  width: 24px;
  height: 24px;
  transition: filter 0.3s ease;
}

.text {
  font-size: 20px;
  transition: color 0.3s ease;
}
</style>