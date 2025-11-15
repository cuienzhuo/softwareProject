<template>
  <div ref="chartRef" class="chart-container"></div>
</template>

<script setup>
import { ref, onMounted, watch } from 'vue';
import * as echarts from 'echarts';

const props = defineProps({
  chartData: {
    type: Object,
    required: true,
    validator(data) {
      return data && Array.isArray(data.timestamps) && Array.isArray(data.actual) && Array.isArray(data.predicted);
    }
  }
});

const chartRef = ref(null);
let chartInstance = null;

// 初始化图表
const initChart = (data) => {
    if (!chartRef.value) return;
    const { timestamps,predicted,actual } = data
  chartInstance = echarts.init(chartRef.value);

  const option = {
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        type: 'shadow'
      },
      formatter: (params) => {
        const time = params[0].axisValue;
        const actualNum = params[0].data[1];
        const predict = params[1]?.data[1] || 'N/A';
        return `
          <div style="padding: 5px; font-size: 12px;">
            <span style="color: #333;">时间:</span> ${time}<br/>
            <span style="color: #00bfa5;">实际值:</span> ${actualNum.toFixed(2)}<br/>
            <span style="color: #ff9900;">预测值:</span> ${predict.toFixed(2)}
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
      data: timestamps,
      axisLabel: {
        rotate: 45,
        interval: 100,
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
      name: '流量',
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
        emphasis: {
          focus: 'series'
        },
        data: actual.map((val, idx) => [timestamps[idx], val])
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
        emphasis: {
          focus: 'series'
        },
        data: predicted.map((val, idx) => [timestamps[idx], val])
      }
    ]
  };

  chartInstance.setOption(option);
};

// 当数据变化时重新渲染
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
  background-color: grey;
  border-radius: 8px;
  overflow: hidden;
}
</style>