<template>
  <div ref="chartRef" class="chart-container"></div>
</template>

<script setup>
import { ref, onMounted, watch } from 'vue';
import * as echarts from 'echarts';

// 接收父组件传入的数据
const props = defineProps({
  chartData: {
    type: Object,
    required: true
  }
});

// 获取 DOM 元素
const chartRef = ref(null);
let chartInstance = null;

// 初始化 ECharts 图表
const initChart = (data) => {
    const main = data;
    const { predictions, test } = main;

    // 确保数据存在
    if (!predictions || !test || !predictions.values || !test.values) return;

    const timestamps = predictions.timestamps; // 时间轴
    const predValues = predictions.values;
    const trueValues = test.values;

    // 创建图表实例
    chartInstance = echarts.init(chartRef.value);

    const option = {
        title: {
            text: 'ARIMA 测试集和预测值对比',
            textStyle: {
                fontSize: 16,
                fontWeight: 'bold',
                color: '#333'
            },
            left: 'center',
            top: '10%'
        },
        tooltip: {
            trigger: 'axis',
            formatter: function (params) {
                const time = params[0].axisValue;
                const pred = params[0].data[1];
                const trueVal = params[1]?.data[1];
                return `
        <div style="padding: 5px;">
          <strong>${time}</strong><br/>
          预测值: ${pred.toFixed(2)}<br/>
          真实值: ${trueVal?.toFixed(2)}
        </div>
      `;
            }
        },
        legend: {
            data: ['预测值', '真实值'],
            bottom: 10
        },
        grid: {
            left: '10%',
            right: '10%',
            top: '15%',
            bottom: '15%'
        },
        xAxis: {
            type: 'category',
            data: timestamps,
            axisLabel: {
                rotate: 45,
                interval: 40, // 👈 每隔 4 个显示一个（即第 0, 5, 10... 个）
                formatter: (value) => {
                return value.split(':').slice(0, 2).join(':');
                }
            },
            axisLine: {
                show: false
            },
            axisTick: {
                show: false
            },
            splitLine: {
                show: false
            },
            splitArea: {
                show: false
            }
        },
        yAxis: {
            type: 'value',
            name: '数值',
            splitLine: {
                lineStyle: {
                    type: 'dashed'
                }
            }
        },
        series: [
            {
                name: '预测值',
                type: 'line',
                smooth: true,
                symbol: 'circle',
                symbolSize: 4,
                lineStyle: {
                    color: '#FF6B6B'
                },
                data: predValues.map((val, idx) => [timestamps[idx], val])
            },
            {
                name: '真实值',
                type: 'line',
                smooth: true,
                symbol: 'circle',
                symbolSize: 4,
                lineStyle: {
                    color: '#4ECDC4'
                },
                data: trueValues.map((val, idx) => [timestamps[idx], val])
            }
        ]
    };
    chartInstance.setOption(option);
}

// 监听数据变化，重新渲染图表
watch(
  () => props.chartData,
  (newData) => {
    if (chartInstance) {
      chartInstance.dispose();
    }
    initChart(newData);
  },
  { deep: true } // 首次加载时也执行
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
  background-color: #fff;
}
</style>