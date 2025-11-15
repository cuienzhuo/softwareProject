<template>
  <div ref="chartRef" class="chart"></div>
</template>

<script setup>
import { ref, onMounted, watch } from 'vue';
import * as echarts from 'echarts';

const props = defineProps({
  chartData: {
    type: Object,
    required: true,
    validator(data) {
      return (
        Array.isArray(data.input_history) &&
        Array.isArray(data.predictions) &&
        Array.isArray(data.truths) &&
        Array.isArray(data.timestamps_input) &&
        Array.isArray(data.timestamps_pred)
      );
    }
  }
});

const chartRef = ref(null);
let chartInstance = null;

// 合并时间轴和数据序列
const mergeData = (chartData) => {
  const { input_history, predictions, truths, timestamps_input, timestamps_pred } = chartData;
  const allTimestamps = [...timestamps_input, ...timestamps_pred];
  const actualValues = [...input_history, ...truths];
  const predictedValues = [...input_history, ...predictions];

  return {
    timestamps: allTimestamps,
    actual: actualValues,
    predicted: predictedValues
  };
};

// 初始化图表
const initChart = (chartData) => {
  if (!chartRef.value || !chartData) return;

  if (chartInstance) {
    chartInstance.dispose();
  }

  chartInstance = echarts.init(chartRef.value);

  const data = mergeData(chartData);

  const option = {
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        type: 'shadow'
      },
      formatter: (params) => {
        const time = params[0].axisValue;
        const actual = params[0].data[1];
        const predicted = params[1]?.data[1] || 'N/A';
        return `
          <div style="padding: 5px; font-size: 12px;">
            <span style="color: #333;">时间:</span> ${time}<br/>
            <span style="color: #00bfa5;">实际值:</span> ${actual.toFixed(2)}<br/>
            <span style="color: #ff9900;">预测值:</span> ${predicted.toFixed(2)}
          </div>
        `;
      }
    },
    legend: {
      data: ['实际值', '预测值'],
      textStyle: {
        color: '#fff'
      },
      top: '5%'
    },
    grid: {
      left: '5%',
      right: '5%',
      bottom: '15%',
      containLabel: true
    },
    xAxis: {
      type: 'category',
      data: data.timestamps,
      axisLabel: {
        rotate: 45,
        fontSize: 10,
        color: '#fff'
      },
      axisLine: {
        lineStyle: {
          color: '#fff'
        }
      },
      axisTick: {
        inside: false
      }
    },
    yAxis: {
      type: 'value',
      name: '交通流量',
      axisLabel: {
        color: '#fff'
      },
      splitLine: {
        lineStyle: {
          color: '#333'
        }
      },
      axisLine: {
        lineStyle: {
          color: '#fff'
        }
      }
    },
    series: [
      {
        name: '实际值',
        type: 'line',
        smooth: true,
        lineStyle: {
          color: '#00bfa5'
        },
        areaStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: '#00bfa5' },
            { offset: 1, color: '#00bfa5', opacity: 0.2 }
          ])
        },
        data: data.actual.map((val, idx) => [data.timestamps[idx], val])
      },
      {
        name: '预测值',
        type: 'line',
        smooth: true,
        lineStyle: {
          color: '#ff9900'
        },
        areaStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: '#ff9900' },
            { offset: 1, color: '#ff9900', opacity: 0.2 }
          ])
        },
        data: data.predicted.map((val, idx) => [data.timestamps[idx], val])
      }
    ]
  };

  chartInstance.setOption(option);

  // 响应式调整
  window.addEventListener('resize', () => {
    chartInstance.resize();
  });
};

// 监听数据变化
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
.chart {
  width: 100%;
  height: 500px;
  background-color: grey;
  border-radius: 8px;
  overflow: hidden;
}
</style>