<template>
  <div class="traffic-chart">
    <div ref="chartContainer" class="chart-container"></div>
  </div>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount, watch, computed,defineProps } from 'vue'
import * as echarts from 'echarts'

// Props
const props = defineProps({
  chartData: {
    type: Object,
    required: true,
    default: () => ({})
  }
})

// Refs
const chartContainer = ref(null)
let chartInstance = null

// Computed
const loading = computed(() => !props.chartData?.timestamps)
const isEmpty = computed(() => {
  return !loading.value && 
         (props.chartData.timestamps?.length === 0 || 
          props.chartData.predictions?.length === 0 ||
          props.chartData.actuals?.length === 0)
})

// 初始化图表
const initChart = () => {
  console.log('initChart called:', chartContainer.value)
  if (!chartContainer.value) return

  // 销毁已有实例
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
      nameGap: 40,
      axisLabel: {
        formatter: '{value}'
      }
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

watch(
  () => props.chartData,
  () => {
    if (!isEmpty.value && !loading.value) {
      initChart()
    }
  },
  { deep: true }
  // 注意：不再 immediate: true
)

onMounted(() => {
  window.addEventListener('resize', handleResize)
  
  // 首次挂载后初始化图表（此时 DOM 已存在）
  if (!isEmpty.value && !loading.value) {
    initChart()
  }
})

const handleResize = () => {
  if (chartInstance) {
    chartInstance.resize()
  }
}

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
}
</style>