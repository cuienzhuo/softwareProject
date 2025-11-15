<template>
  <div ref="chartRef" class="chart-container"></div>
</template>

<script setup>
import * as echarts from 'echarts'
import { onMounted, ref, watch } from 'vue'

// 接收父组件传递的数据
const props = defineProps({
  chartData: {
    type: Object,
    required: true,
    validator: (val) => {
      return val && Array.isArray(val.timestamps) && Array.isArray(val.true_values) && Array.isArray(val.predicted_values)
    }
  }
})

// 挂载图表容器
const chartRef = ref(null)
let chartInstance = null

// 初始化图表
const initChart = (data) => {
    const {timestamps,predicted_values,true_values} = data
  if (!chartRef.value) return

  const option = {
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        type: 'shadow'
      },
      formatter: (params) => {
        const date = params[0].axisValue
        let content = `<div style="padding: 5px; font-size: 12px;">${date}</div>`
        params.forEach(item => {
          content += `<div style="padding: 5px; color: ${item.color};">${item.seriesName}: ${item.value}</div>`
        })
        return content
      }
    },
    legend: {
      data: ['True Values', 'Predicted Values'],
      textStyle: {
        fontSize: 12
      }
    },
    grid: {
      left: '10%',
      right: '10%',
      bottom: '15%',
      top: '15%'
    },
    xAxis: {
      type: 'category',
      data: timestamps,
      axisLabel: {
        rotate: 45,
        interval: 50,
        formatter: (value) => {
          // 简化显示，如 "2013-12-21 06:20"
          return value.split(' ')[1] // 只显示时间部分
        }
      },
      splitLine: {
        show: false
      }
    },
    yAxis: {
      type: 'value',
      name: 'Value',
      axisLabel: {
        formatter: '{value}'
      },
      splitLine: {
        lineStyle: {
          opacity: 0.2
        }
      }
    },
    series: [
      {
        name: 'True Values',
        type: 'line',
        data: true_values,
        smooth: true,
        lineStyle: {
          width: 2,
          color: '#0066cc'
        },
        areaStyle: {
          opacity: 0.2,
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: '#0066cc' },
            { offset: 1, color: '#ffffff' }
          ])
        }
      },
      {
        name: 'Predicted Values',
        type: 'line',
        data: predicted_values,
        smooth: true,
        lineStyle: {
          width: 2,
          color: '#ff6b6b'
        },
        areaStyle: {
          opacity: 0.2,
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: '#ff6b6b' },
            { offset: 1, color: '#ffffff' }
          ])
        }
      }
    ]
  }

  if (chartInstance) {
    chartInstance.setOption(option)
  } else {
    chartInstance = echarts.init(chartRef.value)
    chartInstance.setOption(option)
  }

  // 响应式调整大小
  window.addEventListener('resize', () => {
    if (chartInstance) chartInstance.resize()
  })
}

// 监听数据变化并更新图表
watch(
  () => props.chartData,
  (newData) => {
    if (!newData) return;
    initChart(newData);
  },
  { deep: true }
);

onMounted(() => {
  if (props.chartData) {
    initChart(props.chartData);
  }
});
</script>

<style scoped>
.chart-container {
  width: 100%;
  height: 400px;
  background-color: #f9f9f9;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}
</style>