<template>
  <div class="prediction-chart">
    <div ref="chartRef" style="width: 100%; height: 500px;"></div>
  </div>
</template>

<script setup>
import { ref, onMounted, watch,onUnmounted,defineProps } from 'vue'
import * as echarts from 'echarts'

// 接收父组件传入的 chartData（来自后端）
const props = defineProps({
  chartData: {
    type: Object,
    required: true
  }
})

const chartRef = ref(null)
let myChart = null

// 初始化或更新图表
const initChart = (data) => {
  if (!chartRef.value) return

  // 销毁旧实例
  if (myChart) {
    myChart.dispose()
  }

  myChart = echarts.init(chartRef.value)

  const {
    focus_train_actual,
    focus_train_fitted,
    val_actual,
    val_predicted,
    split_time
  } = data

  // 提取 x 轴时间（去重并排序）
  const allTimes = new Set()
  ;[focus_train_actual, focus_train_fitted, val_actual, val_predicted].forEach(series => {
    series.forEach(item => allTimes.add(item.time))
  })
  const xAxisData = Array.from(allTimes).sort()

  const option = {
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        type: 'cross'
      }
    },
    legend: {
      data: [
        'Actual (Training, Focus)',
        'Fitted (Training, Focus)',
        'Actual (Validation)',
        'Rolling Forecast'
      ],
      top: 10
    },
    xAxis: {
      type: 'time',
      name: 'Time',
      boundaryGap: false
    },
    yAxis: {
      type: 'value',
      name: 'Value'
    },
    series: [
      {
        name: 'Actual (Training, Focus)',
        type: 'line',
        data: focus_train_actual.map(item => [item.time, item.value]),
        color: '#1f77b4',
        smooth: true,
        symbol: 'none'
      },
      {
        name: 'Fitted (Training, Focus)',
        type: 'line',
        data: focus_train_fitted.map(item => [item.time, item.value]),
        color: 'black',
        lineStyle: { type: 'solid' },
        smooth: true,
        symbol: 'none'
      },
      {
        name: 'Actual (Validation)',
        type: 'line',
        data: val_actual.map(item => [item.time, item.value]),
        color: '#2ca02c',
        smooth: true,
        symbol: 'none'
      },
      {
        name: 'Rolling Forecast',
        type: 'line',
        data: val_predicted.map(item => [item.time, item.value]),
        color: 'red',
        lineStyle: { type: 'dashed' },
        smooth: true,
        symbol: 'none'
      }
    ],
    graphic: split_time
      ? {
          elements: [
            {
              type: 'line',
              shape: {
                x1: split_time,
                y1: 0,
                x2: split_time,
                y2: 1
              },
              style: {
                stroke: 'gray',
                lineDash: [5, 5]
              },
              z: 100
            },
            {
              type: 'text',
              left: split_time,
              top: '10%',
              style: {
                text: 'Train/Validation Split',
                fill: 'gray',
                fontSize: 12
              }
            }
          ]
        }
      : {}
  }

  myChart.setOption(option)
}

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

// 组件卸载时销毁图表
onUnmounted(() => {
  if (myChart) {
    myChart.dispose()
  }
})
</script>

<style scoped>
.prediction-chart {
  width: 100%;
  height: 100%;
}
</style>