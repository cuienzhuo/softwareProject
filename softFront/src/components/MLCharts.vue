<template>
  <div ref="chartRef" class="chart"></div>
</template>

<script setup>
import { ref, onMounted, watch,onUnmounted,defineProps } from 'vue';
import * as echarts from 'echarts';

const props = defineProps({
  chartData: {
    type: Object,
    required: true
  }
});

const chartRef = ref(null);
let chartInstance = null;

function initChart(data) {
  const { predictions, test_values, timestamps } = data;

  // 如果没有数据，返回
  if (!predictions || !test_values || !timestamps || predictions.length === 0) return;

  // 初始化 ECharts 实例
  if (chartInstance) {
    chartInstance.dispose();
  }

  chartInstance = echarts.init(chartRef.value);

  const option = {
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        type: 'shadow'
      },
      formatter: function(params) {
        const time = params[0].axisValue;
        let html = `<div style="padding: 5px; font-size: 12px;">${time}</div>`;
        params.forEach(item => {
          html += `<div style="margin-top: 5px;">${item.seriesName}: ${item.value}</div>`;
        });
        return html;
      }
    },
    legend: {
      data: ['预测值', '实际值'],
      top: '5%',
      right: '5%'
    },
    grid: {
      left: '10%',
      right: '15%',
      bottom: '15%',
      containLabel: true
    },
    xAxis: {
      type: 'category',
      data: timestamps,
      axisLabel: {
        rotate: 45,
        fontSize: 10
      },
      splitLine: {
        show: false
      }
    },
    yAxis: {
      type: 'value',
      name: '数值',
      splitLine: {
        show: true
      }
    },
    series: [
      {
        name: '预测值',
        type: 'line',
        smooth: true,
        data: predictions,
        lineStyle: {
          color: '#409eff'
        },
        areaStyle: {
          opacity: 0.3,
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: '#409eff' },
            { offset: 1, color: '#fff' }
          ])
        }
      },
      {
        name: '实际值',
        type: 'line',
        smooth: true,
        data: test_values,
        lineStyle: {
          color: '#67c234'
        },
        areaStyle: {
          opacity: 0.3,
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: '#67c234' },
            { offset: 1, color: '#fff' }
          ])
        }
      }
    ]
  };

  chartInstance.setOption(option);

  // 响应式调整
  window.addEventListener('resize', () => {
    chartInstance.resize();
  });
}

// 监听 chartData 变化并更新图表
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



// 清理函数
onUnmounted(() => {
  if (chartInstance) {
    chartInstance.dispose();
  }
});
</script>

<style scoped>
.chart {
  width: 100%;
  height: 500px;
  background-color: #f9f9f9;
}
</style>