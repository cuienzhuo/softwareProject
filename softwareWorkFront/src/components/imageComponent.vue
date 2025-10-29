<script setup>
import { defineProps,watch } from 'vue';
const props = defineProps({
  frameUrl: {
    type: String,
    default: ''
    },
    isLoading: {
        type: Boolean,
        default:false
  }
});

// 监听 props 变化（可选，如需在变化时执行额外逻辑）
watch(
  () => props.message,
  (newVal, oldVal) => {
    console.log('子组件值变化：', newVal);
    // 执行其他逻辑...
  }
);
</script>

<template>
    <div class="frameContainer">
        <div>
          <div class="textStyle">结果图表</div>
          <div v-loading="isLoading">
            <img :src="frameUrl" alt="图片错误" class="imageStyle"></img>
          </div>
          
        </div>
      </div>
</template>

<style scoped>
.imageStyle{
  width: 1000px;
}
.frameContainer{
  width: 100%;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
}
.textStyle{
  font-size: 25px;
  font-weight: bold;
  padding: 20px 0;
}
</style>