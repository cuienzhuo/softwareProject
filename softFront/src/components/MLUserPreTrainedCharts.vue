<template>
  <div ref="chart" style="width: 100%; height: 400px;"></div>
</template>

<script setup>
import { ref, onMounted, watch,defineProps } from 'vue';
import * as echarts from 'echarts';

const props = defineProps({
  chartData: {
    type: Object,
    required: true
  }
});

const chart = ref(null);
let chartInstance = null;

onMounted(() => {
  chartInstance = echarts.init(chart.value);
  updateChart();
});

watch(() => props.chartData, () => {
  if (chartInstance) {
    updateChart();
  }
});

function updateChart() {
  const option = {
    title: {
      text: 'Prediction vs True Values'
    },
    tooltip: {
      trigger: 'axis'
    },
    legend: {
      data: ['True Values', 'Predicted Values']
    },
    xAxis: {
      type: 'category',
      data: props.chartData.timestamps
    },
    yAxis: {
      type: 'value'
    },
    series: [
      {
        name: 'True Values',
        type: 'line',
        data: props.chartData.true_values
      },
      {
        name: 'Predicted Values',
        type: 'line',
        data: props.chartData.pred_values
      }
    ]
  };
  chartInstance.setOption(option);
}
</script>

<style scoped>
/* 根据需要添加样式 */
</style>