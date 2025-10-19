<script setup>
import iconContainer from './iconContainer.vue';
import { useRouter } from 'vue-router';

const router = useRouter()
const toMainPage = () => {
  router.push("/")
}
</script>

<template>
    <div class="linkContainer">
        <div class="leftContainer" @click="toMainPage">Software Engineering Curriculum</div>
        <div class="rightContainer">
            <iconContainer path="/static/home.png" text="首页" url="/" />
            <iconContainer path="/static/exception.png" text="异常数据分析" url="/ExceptionAnalysis" />
            <iconContainer path="/static/cluster.png" text="聚类分析" url="/ClusterAnalysis" />
            <iconContainer path="/static/predict.png" text="未来预测" url="/Predict" />
        </div>
    </div>
</template>

<style>
.linkContainer{
    display: flex;
    justify-content: space-between;
    align-items: center;
    width: 100%;
    padding: 20px 20px;
}

.rightContainer{
    display: flex;
    align-items: center;
    gap: 10px;
}

.leftContainer {
  color: white;
  cursor: pointer;
  font-size: 20px;
  transition: color 0.3s ease, text-decoration 0.3s ease;
}

.leftContainer:hover {
  color: #c792ea;
  text-decoration: underline;
  text-underline-offset: 4px;
}
</style>