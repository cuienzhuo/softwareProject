<template>
  <div class="anomaly-detection-container">
    <!-- 左侧：数据填写区域 -->
    <div class="left-panel">
      <div class="form-group">
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

      <div class="form-group">
        <label class="label-bold">方法选择</label>
        <select v-model="selectedMethod">
          <option value="">请选择方法</option>
          <option value="iqr">IQR</option>
          <option value="zscore">Z-Score</option>
          <option value="isolation_forest">Isolation Forest</option>
          <option value="dbscan">DBSCAN</option>
        </select>
      </div>

      <!-- IQR 阈值输入 -->
      <div v-if="selectedMethod === 'iqr'" class="form-group">
        <label class="label-bold">阈值(默认为1.5)</label>
        <input type="number" step="0.1" v-model.number="thresholdIQR" />
      </div>

      <!-- Z-Score 阈值输入 -->
      <div v-if="selectedMethod === 'zscore'" class="form-group">
        <label class="label-bold">阈值(默认为3)</label>
        <input type="number" step="0.1" v-model.number="thresholdZScore" />
      </div>

      <div v-if="selectedMethod === 'isolation_forest'" class="form-group">
        <label class="label-bold">孤立树数量(默认为100)</label>
        <input type="number" step="0.1" v-model.number="n_estimators" />

        <label class="label-bold" style="margin-top: 20px;">异常值比例预估(默认为0.01)</label>
        <input type="number" step="1" v-model.number="contamination" />
      </div>

      <!-- DBSCAN 参数输入 -->
      <div v-if="selectedMethod === 'dbscan'" class="form-group">
        <label class="label-bold">邻域半径(默认为0.5)</label>
        <input type="number" step="0.1" v-model.number="dbscanEps" />

        <label class="label-bold" style="margin-top: 20px;">最小邻域点数(默认为5)</label>
        <input type="number" step="1" v-model.number="dbscanMinPts" />
      </div>

      <button
        class="generate-btn"
        :disabled="isGenerateDisabled"
        @click="generateChart"
      >
        生成图表
      </button>
    </div>

    <!-- 右侧：图表与详情 -->
    <div class="right-panel">
      <!-- 异常数据分析图表（占位） -->
      <div class="chart-area">
        <h3>异常数据分析图表</h3>
        <!-- 此处可集成 ECharts、Chart.js 等 -->
        <!-- <div v-if="!hasChartData" class="chart-placeholder">图表区域（待实现）</div> -->
        <div ref="chartRef" class="chart-container"></div>
      </div>

      <!-- 异常分析数据详细信息 -->
      <div class="details-area">
        <h3>异常分析数据详情</h3>
        <div class="detail-item">
          <span>数据总数：</span>
          <strong>{{ totalDataCount }}</strong>
        </div>
        <div class="detail-item">
          <span>异常数据个数：</span>
          <strong>{{ anomalyCount }}</strong>
        </div>
        <div class="detail-item">
          <span>异常数据比例：</span>
          <strong>{{ anomalyRatio }}</strong>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, defineProps,onMounted,watch } from 'vue'
import * as echarts from 'echarts'
import api from '@/api'

// Props（根据你的实际传参方式调整）
const exceptionAnalysis = ref({
  location: { type: String, required: true },
  method: { type: String, required: true },
  methodCnName: { type: String, required: true },
  results: { type: Array, required: true } // 每项含 timestamp, value, ${method}_anomaly
})

const hasChartData = computed(() => exceptionAnalysis.value.results?.length > 0)

const chartRef = ref(null)
let myChart = null

const generateChart = async () => {
  const config = {
    address: selectedRegion.value,
    method: selectedMethod.value,
  };

  // 根据 selectedMethod 添加额外字段
  switch (selectedMethod.value) {
    case 'iqr':
      config.threshold = thresholdIQR.value === '' ? 1.5: thresholdIQR.value;
      break
    case 'zscore':
      config.threshold = thresholdZScore.value === '' ? 3 : thresholdZScore.value;
      break;

    case 'isolation_forest':
      config.contamination = contamination.value === '' ? 0.01 : contamination.value; // 默认值示例
      config.n_estimators = n_estimators.value === '' ? 100 : n_estimators.value;
      break;

    case 'dbscan':
      config.eps = dbscanEps.value === '' ? 0.5 : dbscanEps.value;
      config.min_samples = dbscanMinPts.value === '' ? 5 : dbscanMinPts.value;
      break;

    default:
      // 可选：处理未知方法，或保持只有 address + method
      break;
  }
  const response = await api.post('/api/anomaly-analysis/', config)
  const data = response.data
  console.log(data)
  if (data.code === 200) {
    exceptionAnalysis.value = data.data
    totalDataCount.value = data.analysis_overview.total_count
    anomalyCount.value = data.analysis_overview.anomaly_count
    anomalyRatio.value = data.analysis_overview.anomaly_ratio
  }
  console.log(exceptionAnalysis.value)
}

const initChart = () => {
  console.log(!chartRef.value)
  if (!chartRef.value) return
  console.log("开始")
  myChart?.dispose()
  myChart = echarts.init(chartRef.value)

  const anomalyCol = `${exceptionAnalysis.value.method}_anomaly`

  // 分离正常数据和异常点
  const normalData = []
  const anomalyData = []
  console.log("处理")
  exceptionAnalysis.value.results.forEach(item => {
    const time = item.timestamp // 假设是 'YYYY-MM-DD' 字符串
    const value = item.value
    normalData.push([time, value])
    if (item[anomalyCol] === 1) {
      anomalyData.push([time, value])
    }
  })

  console.log("选项")
  const option = {
    title: {
      text: `${exceptionAnalysis.value.location} - ${exceptionAnalysis.value.methodCnName} 异常检测结果`,
      subtext: `异常值判定规则：${exceptionAnalysis.value.methodCnName}`,
      left: 'center',
      top: '10px',
      textStyle: { fontSize: 16, fontWeight: 'bold' },
      subtextStyle: { fontSize: 12, color: '#666' },
      itemGap: 12 // 主副标题间距
    },
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        type: 'cross',
        label: { backgroundColor: '#6a7985' }
      }
    },
    legend: {
      data: ['正常数据', `异常点（共${anomalyData.length}个）`],
      top: '60px', // 下移图例，避开标题
      left: 'center',
      textStyle: { fontSize: 12 }
    },
    xAxis: {
      type: 'category',
      name: '时间',
      nameLocation: 'middle',
      nameGap: 35, // 增大轴名与刻度距离
      axisLabel: {
        rotate: 45,
        interval: Math.floor(normalData.length / 20), // 自动稀疏显示（每20个点显示1个）
        formatter: (value) => {
          const date = new Date(value)
          return `${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')}`
        },
        margin: 15
      },
      data: normalData.map(d => d[0]),
      axisTick: { alignWithLabel: true }
    },
    yAxis: {
      type: 'value',
      name: '数值',
      nameLocation: 'middle',
      nameGap: 40, // 增大 Y 轴名称距离
      axisLabel: { formatter: '{value}' }
    },
    series: [
      {
        name: '正常数据',
        type: 'line',
        data: normalData.map(d => d[1]),
        symbol: 'none',
        lineStyle: {
          color: 'blue',
          opacity: 0.6,
          width: 1.2
        },
        emphasis: { focus: 'series' }
      },
      {
        name: `异常点（共${anomalyData.length}个）`,
        type: 'scatter',
        data: anomalyData,
        symbolSize: 8,
        itemStyle: { color: 'red' },
        emphasis: { focus: 'series' }
      }
    ],
    grid: {
      left: '15%',
      right: '10%',
      bottom: '25%', // 为旋转的 X 轴标签留足空间
      top: '25%'   // 为标题和图例留出顶部空间
    }
  }

  myChart.setOption(option)
}

// 如果 results 动态变化，需监听更新
watch(() => exceptionAnalysis.value, () => {
  console.log("生成图表")
  initChart()
}, { deep: true })

const props = defineProps({
  regions: {
    type: Array,
    required:true
  }
})
// 地区和方法选择
const selectedRegion = ref('')
const selectedMethod = ref('')

// 方法参数
const thresholdIQR = ref(1.5)
const thresholdZScore = ref(3)
const n_estimators = ref(100)
const contamination = ref(0.01)
const dbscanEps = ref(0.5)
const dbscanMinPts = ref(5)

// 模拟分析结果数据（实际应来自后端或计算逻辑）
const totalDataCount = ref(0)
const anomalyCount = ref(0)
const anomalyRatio = ref("0.00%")
const isGenerateDisabled = computed(() => {
  return selectedRegion.value === '' || selectedMethod.value === ''
})
</script>

<style scoped>
.anomaly-detection-container {
  display: flex;
  height: 100%;
  gap: 24px;
  padding: 20px;
  box-sizing: border-box;
}

.left-panel {
  flex: 1;
  max-width: 400px;
}

.right-panel {
  flex: 2;
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.form-group {
  margin-bottom: 20px;
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

.chart-area,
.details-area {
  background-color: #f9f9f9;
  padding: 16px;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}

.chart-placeholder {
  height: 200px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #888;
  background-color: #fff;
  border: 1px dashed #ccc;
  border-radius: 4px;
  margin-top: 12px;
}

.detail-item {
  display: flex;
  justify-content: space-between;
  margin-top: 8px;
  font-size: 14px;
}

.detail-item span {
  color: #555;
}

.detail-item strong {
  color: #333;
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
  height: 500px; /* 可根据需求调整 */
}
</style>