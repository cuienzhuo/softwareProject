<template>
  <div class="container">
    <!-- 左侧参数设置 -->
    <div class="parameter-setting">
      <div class="form-group">
        <label>地区选择：</label>
        <select v-model="selectedRegion" >
            <option value="">请选择地区</option>
          <option v-for="region in props.regions" :key="region.value" :value="region.value">
            {{ region.label }}
          </option>
        </select>
      </div>

      <button @click="runPrediction"         
        class="generate-btn"
        :disabled="isGenerateDisabled"
        >生成图表</button>
    </div>

    <!-- 右侧图表显示 -->
    <div class="chart-display">
      <div ref="chart" style="width: 100%; height: 400px;"></div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, watch } from 'vue';
import * as echarts from 'echarts';
import api from '@/api';

// 父组件传入的地区数据
const props = defineProps({
  regions: {
    type: Array,
    required: true,
  },
});

let selectedRegion = ref('');

// ECharts 实例
let chartInstance = null;

// 存储从后端获取的数据
let predictionData = ref({});

// 获取图表配置项
function getChartOption() {
  if (!predictionData.value || Object.keys(predictionData.value).length === 0) {
    return {
      title: {
        text: '请选择地区并点击生成图表',
        left: 'center',
      },
      xAxis: {},
      yAxis: {},
      series: [],
    };
  }

  // 提取算法名和MAPE值
  const algorithms = Object.keys(predictionData.value);
  const mapeValues = Object.values(predictionData.value).map(val => parseFloat(val.toFixed(4)));

  return {
    title: {
      text: `${selectedRegion.value} 地区不同算法MAPE对比`,
      left: 'center',
    },
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        type: 'shadow'
      }
    },
    xAxis: {
      type: 'category',
      data: algorithms,
      axisLabel: {
        rotate: 45, // 防止标签过长重叠
      }
    },
    yAxis: {
      type: 'value',
      name: 'MAPE',
      min: 0,
      max: Math.max(...mapeValues) * 1.1, // 留出一些空间
      splitLine: {
        show: true
      }
    },
    series: [
      {
        name: 'MAPE',
        type: 'bar',
        barWidth: '40%',
        data: mapeValues,
        itemStyle: {
          color: '#87CEEB' // 天蓝色
        }
      }
    ]
  };
}

// 更新图表
function updateChart() {
  if (!chartInstance) return;
  chartInstance.setOption(getChartOption());
}

// 生成预测并更新图表
async function runPrediction() {
  if (!selectedRegion.value) return;

  try {
    const response = await api.post("/api/future_compare/", {
      address: selectedRegion.value
    });
    predictionData.value = response.data.result; // 假设返回的就是 { ML: 0.076..., ARIMA: ... }
    updateChart();
  } catch (error) {
    console.error('请求失败:', error);
    alert('请求失败，请检查网络或参数');
  }
}

// 初始化ECharts
onMounted(() => {
  chartInstance = echarts.init(document.querySelector('.chart-display > div'));
  updateChart(); // 初始为空图
});
</script>

<style scoped>
.container {
  display: flex;
}
.parameter-setting {
  width: 20%;
}
.chart-display {
  width: 80%;
}
.form-group {
  margin-bottom: 16px;
}

.form-group select{
    width: 100%;
    height: 30px;
}

.form-group label {
  display: block;
  margin-bottom: 6px;
  font-weight: bold;
}
.generate-btn {
  /* 天蓝色渐变 */
  background: linear-gradient(90deg, #87CEEB, #00BFFF);
  color: white;
  border: none;
  padding: 10px 24px;
  font-size: 16px;
  width: 100%;
  border-radius: 6px;
  cursor: pointer;
  transition: transform 0.1s, box-shadow 0.2s; /* 添加变换和阴影过渡效果 */
}

/* 禁用状态：灰色且不可点击 */
.generate-btn:disabled {
  background: #ccc;
  cursor: not-allowed;
  opacity: 1; /* 防止浏览器默认降低透明度 */
}

/* 按下时的效果 */
.generate-btn:active {
  transform: scale(0.95); /* 缩放按钮至95%大小 */
  box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.3); /* 添加阴影，制造按压效果 */
}

</style>