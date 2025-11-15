<template>
  <div class="traffic-chart">
    <div ref="chartContainer" class="chart-container"></div>
  </div>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount, watch, computed,defineProps } from 'vue'
import * as echarts from 'echarts'

const props = defineProps({
  chartData: {
    type: Object,
    required: true,
    default: () => ({})
  }
})

const chartContainer = ref(null)
let chartInstance = null

// 判断是否在加载（数据未传入或字段缺失）
const loading = computed(() => {
  return !props.chartData ||
         !Array.isArray(props.chartData.timestamps) ||
         props.chartData.timestamps.length === 0
})

// 判断是否为空（有结构但无有效数据）
const isEmpty = computed(() => {
  return !loading.value &&
         (props.chartData.actuals?.length === 0 || 
          props.chartData.predictions?.length === 0 ||
          props.chartData.timestamps?.length !== props.chartData.actuals?.length ||
          props.chartData.timestamps?.length !== props.chartData.predictions?.length)
})

// 初始化或更新图表
const initChart = () => {
  if (!chartContainer.value || isEmpty.value || loading.value) return

  if (chartInstance) {
    chartInstance.dispose()
  }

  chartInstance = echarts.init(chartContainer.value)

  const option = {
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        type: 'cross'
      }
    },
    legend: {
      data: ['真实值', '预测值'],
      bottom: 10
    },
    grid: {
      left: '3%',
      right: '4%',
      bottom: '15%',
      top: '10%',
      containLabel: true
    },
    xAxis: {
      type: 'category',
      boundaryGap: false,
      data: props.chartData.timestamps,
      axisLabel: {
        rotate: 45,
        fontSize: 12
      }
    },
    yAxis: {
      type: 'value',
      name: '交通流量',
      nameLocation: 'middle',
      nameGap: 40
    },
    series: [
      {
        name: '真实值',
        type: 'line',
        data: props.chartData.actuals,
        smooth: true,
        symbol: 'none',
        lineStyle: {
          width: 2,
          color: '#38A169' // green-600
        },
        emphasis: {
          focus: 'series'
        }
      },
      {
        name: '预测值',
        type: 'line',
        data: props.chartData.predictions,
        smooth: true,
        symbol: 'none',
        lineStyle: {
          width: 2,
          color: '#DD6B20', // orange-600
          type: 'dashed'
        },
        emphasis: {
          focus: 'series'
        }
      }
    ]
  }

  chartInstance.setOption(option)
}

// 监听数据变化
watch(
  () => props.chartData,
  () => {
    if (!loading.value && !isEmpty.value) {
      initChart()
    }
  },
  { deep: true }
)

// 窗口缩放自适应
const handleResize = () => {
  if (chartInstance) {
    chartInstance.resize()
  }
}

onMounted(() => {
  window.addEventListener('resize', handleResize)
  // 首次挂载后初始化（确保 DOM 存在）
  if (!loading.value && !isEmpty.value) {
    initChart()
  }
})

onBeforeUnmount(() => {
  window.removeEventListener('resize', handleResize)
  if (chartInstance) {
    chartInstance.dispose()
    chartInstance = null
  }
})
</script>

<style scoped>
.traffic-chart {
  width: 100%;
  height: 100%;
  position: relative;
}

.chart-container {
  width: 100%;
  height: 400px;
}

.chart-loading,
.chart-empty {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 400px;
  color: #999;
  font-size: 16px;
  width: 100%;
}
</style>