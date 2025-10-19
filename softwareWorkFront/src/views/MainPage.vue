<script setup>
import { onMounted, ref } from 'vue';
import { useRouter } from 'vue-router';

import topNavigation from '@/components/topNavigation.vue';
import MainPageButton from '@/components/MainPageButton.vue';

const router = useRouter()
const text = ref("软件工程课设展示")
const displayText = ref("")
const speed = ref(100)

const isTyping = ref(true)

onMounted(() => {
  let i = 0
  const timer = setInterval(() => {
    displayText.value += text.value.charAt(i)
    i++
    if (i >= text.value.length) {
      clearInterval(timer)
    }
  }, speed.value)
})
</script>

<template>
    <div class="back">
        <div class="gradient-box">
            <topNavigation />
        </div>

        <div class="contentBox">
            <div class="titleBox">Software Engineering Curriculum</div>
            <div class="typewriter-container">
                <div class="text-content">{{ displayText }}</div>
                <div class="cursor" :class="{ 'blinking': isTyping }"></div>
            </div>
            <div class="selectBox">
                <MainPageButton text="异常数据分析" url="/ExceptionAnalysis" />
                <MainPageButton text="聚类分析" url="/ClusterAnalysis" />
                <MainPageButton text="未来预测" url="/Predict" />
            </div>
        </div>
    </div>

</template>

<style scoped>
.back {
    width: 100%;
    height: 100vh;
    margin: 0;
    background-image: url('/static/backImg.jpg');
    background-color: #f0f4f8;
    background-size: cover; /* 确保图片覆盖整个区域 */
    background-position: center; /* 图片居中 */
    background-repeat: no-repeat; /* 不重复 */
    background-attachment: fixed; /* 可选：固定背景 */
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


::selection{
    background: rgba(199, 146, 234, 0.3); /* 淡紫色+30%透明度 */
    color: white; 
}
.contentBox{
    display: flex;
    flex-direction: column;
    width: 100%;
    height: 93vh;
    align-items: center;
    justify-content: center;
}
.titleBox{
    font-size: 55px;          /* 大号字体 */
  font-weight: 700;         /* 加粗 (等同于bold) */
  color: white;             /* 白色字体 */
  text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5); /* 右下方阴影 */
  
  /* 可选增强效果 */
  letter-spacing: 1px;      /* 字间距微调 */
}
.typewriter-container {
  display: flex;
  align-items: center;
  position: relative;
  font-size: 24px; /* 字体大小 */
  font-weight: bold;
  color: white;
}

.text-content {
  margin-right: 8px; /* 文字与光标的间距 */
}

.cursor {
  display: inline-block;
  position: relative;
  top: 3px;
  width: 3px;
  height: 24px; /* 与字体大小相同 */
  background-color: white;
  vertical-align: middle;
}

.blinking {
  animation: blink 0.8s step-end infinite;
}

@keyframes blink {
  from, to { opacity: 1 }
  50% { opacity: 0 }
}
.selectBox{
    margin-top: 200px;
    width: 36%;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
</style>
