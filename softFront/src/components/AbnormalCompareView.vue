<template>
  <div class="analysis-panel">
    <!-- 左侧：参数定义 -->
    <div class="params-section">
      <div class="param-item">
        <label class="label">分析方法</label>
        <select v-model="selectedMethod" class="select">
          <option value="method1">
            同一地区在不同分析方法下的异常数据个数比较
          </option>
          <option value="method2">
            同一方法在不同地区下的异常数据个数比较
          </option>
        </select>
      </div>
      <div class="form-group" v-if="selectedMethod === 'method1'">
        <label class="label-bold">地区</label>
        <select v-model="selectedRegion">
          <option value="">请选择地区</option>
          <option
            v-for="region in props.regions"
            :key="region.value"
            :value="region.value"
          >
            {{ region.label }}
          </option>
        </select>
      </div>

      <div class="form-group" v-if="selectedMethod === 'method2'">
        <label class="label-bold">方法选择</label>
        <select v-model="selectedAnalysis">
          <option value="">请选择方法</option>
          <option value="iqr">IQR</option>
          <option value="zscore">Z-Score</option>
          <option value="isolation_forest">Isolation Forest</option>
          <option value="dbscan">DBSCAN</option>
        </select>
      </div>

      <button
        class="generate-btn"
        :disabled="isGenerateDisabled || isLoading"
        @click="generateChart"
      >
        生成图表
      </button>
    </div>

    <!-- 右侧：图像展示 -->
    <div class="chart-section">
      <div ref="chartRef" class="chart-container"></div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, defineProps,onMounted, onUnmounted } from 'vue'
import * as echarts from 'echarts'
import api from '@/api'

const props = defineProps({
  regions: {
    type: Array,
    required:true
  }
})

const isLoading = ref(false)
const selectedRegion = ref('')
const selectedAnalysis = ref('')

// 响应式数据：当前选中的分析方法
const selectedMethod = ref('method1')

// 计算属性：根据值显示对应的中文描述
const selectedMethodLabel = computed(() => {
  return selectedMethod.value === 'method1'
    ? '同一地区在不同分析方法下的异常数据个数比较'
    : '同一方法在不同地区下的异常数据个数比较'
})

const chartRef = ref(null)
let chartInstance = null

const generateChart = async () => {
  let request = null
  let title = ''

  isLoading.value = true
  if (selectedMethod.value === 'method1') {
    // 同一地区，不同方法
    request = {
      'address':selectedRegion.value
    }
    title = `${selectedRegion.value} - 四种异常检测方法异常点数量对比`
  } else {
    // 同一方法，不同地区
    request = {
      'method':selectedAnalysis.value
    }
    const methodMap = {
      iqr: 'IQR',
      zscore: 'Z-Score',
      isolation_forest: '孤立森林',
      dbscan: 'DBSCAN'
    }
    const methodName = methodMap[selectedAnalysis.value] || selectedAnalysis.value
    title = `各地区异常点数量对比（${methodName}）`
  }

  try {
    const res = await api.post("/api/anomaly_compare/",request)
    const data = res.data.data

    // 假设后端返回 { xAxis: [...], series: [...] }
    renderChart(data.method_names, data.anomaly_counts, title)
  } catch (err) {
    console.error('图表生成失败:', err)
    // alert('图表加载失败，请检查参数或稍后重试')
  }
  isLoading.value = false
}

// 渲染 ECharts
const renderChart = (xAxisData, seriesData, titleText) => {
  if (!chartRef.value) return

  if (chartInstance) {
    chartInstance.dispose()
  }

  chartInstance = echarts.init(chartRef.value)

  // 根据 method 动态设置颜色
  const isMethod1 = selectedMethod.value === 'method1'
  const color = isMethod1 ? 'skyblue' : 'lightcoral'
  const borderColor = isMethod1 ? 'navy' : 'darkred'

  const option = {
    title: {
      text: titleText,
      left: 'center',
      textStyle: { fontSize: 14 }
    },
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
    grid: {
      left: '3%',
      right: '4%',
      bottom: isMethod1 ? '15%' : '20%', // 地区名可能更长
      containLabel: true
    },
    xAxis: {
      type: 'category',
      data: xAxisData,
      axisLabel: {
        rotate: isMethod1 ? 30 : 45,
        fontSize: 12
      },
      axisTick: { alignWithLabel: true }
    },
    yAxis: {
      type: 'value',
      name: '异常点数量',
      nameLocation: 'middle',
      nameGap: 40,
      axisLabel: { fontSize: 12 }
    },
    series: [{
      name: '异常数量',
      type: 'bar',
      data: seriesData.map(count => ({
        value: count,
        itemStyle: { color, borderColor, borderWidth: 1 }
      })),
      label: {
        show: true,
        position: 'top',
        formatter: (params) => (params.value > 0 ? String(params.value) : ''),
        fontSize: 12
      }
    }]
  }

  chartInstance.setOption(option)
}

const handleResize = () => {
  chartInstance?.resize()
}

onMounted(() => {
  window.addEventListener('resize', handleResize)
})

onUnmounted(() => {
  window.removeEventListener('resize', handleResize)
  chartInstance?.dispose()
})

const isGenerateDisabled = computed(() => {
  return selectedMethod.value === '' ||
    (selectedMethod.value === 'method1' && selectedRegion.value === '') || (selectedMethod.value === 'method2' && selectedAnalysis.value === '')
})
</script>

<style scoped>
.analysis-panel {
  display: flex;
  height: 100%;
  width: 100%;
  gap: 24px;
  padding: 20px;
  box-sizing: border-box;
}

.params-section {
  flex: 0 0 300px; /* 固定宽度左侧 */
}

.param-item {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.label {
  font-weight: bold;
  font-size: 14px;
  color: #333;
}

.select {
  padding: 8px 12px;
  border: 1px solid #ccc;
  border-radius: 4px;
  font-size: 14px;
  background-color: white;
}

.chart-section {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: #f9f9f9;
  border: 1px dashed #ddd;
  border-radius: 6px;
}

.chart-placeholder p {
  color: #666;
  font-size: 16px;
}
.form-group {
  margin-bottom: 20px;
  margin-top: 20px;
}

.label-bold {
  display: block;
  font-weight: bold;
  margin-bottom: 6px;
}

select,
input {
  width: 100%;
  padding: 8px 12px;
  border: 1px solid #ccc;
  border-radius: 4px;
  font-size: 14px;
  box-sizing: border-box;
}

.generate-btn {
  /* 天蓝色渐变 */
  background: linear-gradient(90deg, #87CEEB, #00BFFF);
  color: white;
  border: none;
  padding: 10px 24px;
  font-size: 16px;
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
.chart-container {
  width: 100%;
  height: 400px;
  min-height: 300px;
}
</style>