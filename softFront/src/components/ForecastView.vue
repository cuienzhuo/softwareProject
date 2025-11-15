<template>
  <div class="container">
    <!-- 左侧参数设置 -->
    <div class="sidebar">

        <div class="form-group">
        <label>地区选择：</label>
        <select v-model="selectedRegion" :disabled="algorithmOptions.length === 0">
            <option value="">请选择地区</option>
          <option v-for="region in props.regions" :key="region.value" :value="region.value">
            {{ region.label }}
          </option>
        </select>
      </div>
      <!-- 是否使用预训练模型 -->
      <div class="form-group">
        <label>是否使用预训练模型：</label>
        <div class="radio-group">
          <label>
            <input
              type="radio"
              v-model="usePretrained"
              :value="true"
            /> 是
          </label>
          <label>
            <input
              type="radio"
              v-model="usePretrained"
              :value="false"
            /> 否
          </label>
        </div>
      </div>

      <!-- 算法选择 -->
      <div class="form-group">
        <label>选择算法：</label>
        <select v-model="selectedAlgorithm" :disabled="algorithmOptions.length === 0">
          <option v-for="opt in algorithmOptions" :key="opt.value" :value="opt.value">
            {{ opt.label }}
          </option>
        </select>
      </div>

      <!-- forecast_steps -->
      <div v-if="showForecastSteps()" class="form-group">
        <label>forecast_steps：</label>
        <input type="number" v-model.number="params.forecast_steps" />
      </div>

      <!-- backtest_start_pct -->
      <div v-if="showBacktestStartPct()" class="form-group">
        <label>backtest_start_pct：</label>
        <input type="number" step="0.1" min="0" max="1" v-model.number="params.backtest_start_pct" />
      </div>

      <!-- roll_steps -->
      <div v-if="showRollSteps()" class="form-group">
        <label>roll_steps：</label>
        <input type="number" v-model.number="params.roll_steps" />
      </div>

      <div v-if="showTrainRatio()" class="form-group">
        <label>train_ratio：</label>
        <input type="number" v-model.number="params.train_ratio" />
      </div>

      <div v-if="showPQD()" class="form-group">
        <label>p：</label>
        <input type="number" v-model.number="params.p" />
      </div>

      
      <div v-if="showPQD()" class="form-group">
        <label>d：</label>
        <input type="number" v-model.number="params.d" />
      </div>

      <div v-if="showPQD()" class="form-group">
        <label>q：</label>
        <input type="number" v-model.number="params.q" />
      </div>

      <div v-if="showNLags()" class="form-group">
        <label>n_lags：</label>
        <input type="number" v-model.number="params.n_lags" />
      </div>

      <div v-if="showLookBack()" class="form-group">
        <label>look_back：</label>
        <input type="number" v-model.number="params.look_back" />
      </div>

      <div v-if="showEpochs()" class="form-group">
        <label>epochs：</label>
        <input type="number" v-model.number="params.epochs" />
      </div>

      <div v-if="showBatchSize()" class="form-group">
        <label>batch_size：</label>
        <input type="number" v-model.number="params.batch_size" />
      </div>

      <div v-if="showLstmUnits()" class="form-group">
        <label>lstm_units：</label>
        <input type="number" v-model.number="params.lstm_units" />
      </div>

      <div v-if="showForecastHorizon()" class="form-group">
        <label>forecast_horizon：</label>
        <input type="number" v-model.number="params.forecast_horizon" />
      </div>

      <div v-if="showSeqLen()" class="form-group">
        <label>seq_len：</label>
        <input type="number" v-model.number="params.seq_len" />
      </div>

      <div v-if="showPredLen()" class="form-group">
        <label>pred_len：</label>
        <input type="number" v-model.number="params.pred_len" />
      </div>

       <div v-if="showPatchLen()" class="form-group">
        <label>patch_len：</label>
        <input type="number" v-model.number="params.patch_len" />
      </div>

      <div v-if="showStride()" class="form-group">
        <label>stride：</label>
        <input type="number" v-model.number="params.stride" />
      </div>

      <button @click="runPrediction">运行预测</button>
    </div>

    <!-- 右侧结果展示 -->
    <div class="main-content">
      <div class="chart-placeholder">
        <component
        v-if="currentComponent"
        :is="currentComponent"
        :chartData="chartData"
        />
        <div v-else class="mock-chart">
        请选择算法并运行预测以显示图表
        </div>
        <component
        v-if="currentComponent && !usePretrained && selectedAlgorithm !== 'autoformer'"
        :is="currentComponent"
        :chartData="partChartData"
        />

      </div>

      <!-- 评估指标表格 -->
      <div class="metrics-table">
        <h3>评估指标</h3>
        <table>
          <thead>
            <tr>
              <th>MAE</th>
              <th>RMSE</th>
              <th>MAPE (%)</th>
            </tr>
          </thead>
          <tbody>
            <tr v-if="hasRun">
              <td>{{ metrics.mae.toFixed(4) }}</td>
              <td>{{ metrics.rmse.toFixed(4) }}</td>
              <td>{{ (metrics.mape * 100).toFixed(2) }}</td>
            </tr>
            <tr v-else>
              <td colspan="3">尚未运行预测</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, defineProps } from 'vue'
import api from '@/api'
import ARIMAUserPreTrainedCharts from './ARIMAUserPreTrainedCharts.vue'
import MLUserPreTrainedCharts from './MLUserPreTrainedCharts.vue'
import DLUserPreTrainedCharts from './DLUserPreTrainedCharts.vue'
import TransformerUserPreTrainedCharts from './TransformerUserPreTrainedCharts.vue'
import ARIMACharts from './ARIMACharts.vue'
import MLCharts from './MLCharts.vue'
import DLCharts from './DLCharts.vue'
import AutoformerCharts from './AutoformerCharts.vue'
import PatchTSTCharts from './PatchTSTCharts.vue'

const props = defineProps({
    regions: {
        type: Array,
        required:true
    }
})
// 响应式数据
const usePretrained = ref(false)
const selectedAlgorithm = ref('arima')

const selectedRegion = ref('')
const currentComponent = ref(null)

const chartData = ref()
const partChartData = ref()

// 所有算法选项
const allAlgorithms = [
  { value: 'arima', label: 'ARIMA' },
  { value: 'random_forest', label: '随机森林' },
  { value: 'lstm', label: 'LSTM' },
  { value: 'autoformer', label: 'Autoformer' },
  { value: 'patchtst', label: 'PatchTST' }
]

// 根据是否使用预训练模型过滤算法选项
const algorithmOptions = computed(() => {
  if (usePretrained.value) {
    return allAlgorithms.filter(opt => opt.value !== 'patchtst')
  }
  return allAlgorithms
})

const showForecastSteps = () => {
  return  selectedAlgorithm.value === 'arima' || (usePretrained.value &&selectedAlgorithm.value === 'random_forest')
}

const showBacktestStartPct = () => {
  return usePretrained.value&& selectedAlgorithm.value === 'random_forest'
}

const showRollSteps = () => {
  return (usePretrained.value && selectedAlgorithm.value === 'lstm') || selectedAlgorithm.value === 'autoformer'
}

const showTrainRatio = () => {
  return !usePretrained.value && selectedAlgorithm.value !== 'autoformer'
}

const showPQD = () => {
  return !usePretrained.value && selectedAlgorithm.value === 'arima'
}

const showNLags = () => {
  return !usePretrained.value && selectedAlgorithm.value === 'random_forest'
}

const showLookBack = () => {
  return !usePretrained.value && (selectedAlgorithm.value === 'lstm' || selectedAlgorithm.value === 'autoformer')
}

const showEpochs = () => {
  return !usePretrained.value && (selectedAlgorithm.value === 'lstm' || selectedAlgorithm.value === 'autoformer' || selectedAlgorithm.value === 'patchtst')
}
const showBatchSize = () => {
  return (!usePretrained.value && selectedAlgorithm.value === 'lstm') || selectedAlgorithm.value === 'patchtst'
}
const showLstmUnits = () => {
  return !usePretrained.value && selectedAlgorithm.value === 'lstm'
}
const showForecastHorizon = () => {
  return !usePretrained.value && selectedAlgorithm.value === 'autoformer'
}
const showSeqLen = () => {
  return selectedAlgorithm.value === 'patchtst'
}
const showPredLen = () => {
  return selectedAlgorithm.value === 'patchtst'
}
const showPatchLen = () => {
  return selectedAlgorithm.value === 'patchtst'
}
const showStride = () => {
  return selectedAlgorithm.value === 'patchtst'
}
// 监听算法变化，重置参数
watch(selectedAlgorithm, () => {
  resetParams()
})

watch(usePretrained, () => {
  resetParams()
})

// 参数对象
const params = ref({
  forecast_steps: 5,
  backtest_start_pct: 0.5,
  rollsteps: 100,
  roll_steps: 100,
  train_ratio: 0.8,
  p: 1,
  q: 1,
  d: 1,
  n_lags: 6,
  look_back: 60,
  epochs: 50,
  batch_size: 64,
  lstm_units: 50,
  forecast_horizon: 12,
  seq_len: 96,
  pred_len: 24,
  patch_len: 16,
  stride: 8
})

// 重置参数为默认值（根据当前算法）
function resetParams() {
  const alg = selectedAlgorithm.value
  if (alg === 'arima') {
    if (usePretrained.value) {
      params.value = { forecast_steps: 5 }
    } else {
      params.value = { forecast_steps:5,train_ratio:0.8,p:1,q:1,d:1 }
    }
  } else if (alg === 'random_forest') {
    if (usePretrained.value) {
      params.value = { backtest_start_pct: 0.5, forecast_steps: 48 }
    } else {
      params.value = {train_ratio:0.8,n_lags:6}
    }
  } else if (alg === 'lstm') {
    if (usePretrained.value) {
      params.value = { roll_steps: 100 }
    } else {
      params.value = {train_ratio:0.8,look_back:60,epochs:50,batch_size:64,lstm_units:50}
    }
  } else if(alg === 'autoformer') {
    if (usePretrained.value) {
      params.value = { roll_steps: 100 }
    } else {
      params.value = {look_back:60,epochs:50,forecast_horizon:12,roll_steps:100}
    }
  } else {
    params.value = {train_ratio:0.8,batch_size:32,epochs:10,seq_len: 96,pred_len: 24,patch_len: 16,stride: 8}
  }
}

// 模拟运行预测
const hasRun = ref(false)
const metrics = ref({
  mae: 0,
  rmse: 0,
  mape: 0
})

const generateConfig = () => {
    const config = {
        address: selectedRegion.value,
        method: selectedAlgorithm.value,
        usePretrained:usePretrained.value
    }
    if (selectedAlgorithm.value === 'arima' && usePretrained.value) {
        console.log("使用预处理过的arima")
        let forecast_steps = params.value.forecast_steps
        config.forecast_steps = forecast_steps === '' ? 5 : forecast_steps
    } else if (selectedAlgorithm.value === 'random_forest' && usePretrained.value) {
        let backtest_start_pct = params.value.backtest_start_pct
        let forecast_steps = params.value.forecast_steps
        config.backtest_start_pct = backtest_start_pct === '' ? 0.5 : backtest_start_pct
        config.forecast_steps = forecast_steps === '' ? 48 : forecast_steps
    } else if (selectedAlgorithm.value === 'lstm' && usePretrained.value) {
        let roll_steps = params.value.roll_steps
        config.roll_steps = roll_steps === '' ? 100 : roll_steps
    } else if (selectedAlgorithm.value === 'autoformer' && usePretrained.value) {
        let roll_steps = params.value.roll_steps
        config.roll_steps = roll_steps === '' ? 100 : roll_steps
    } else if (selectedAlgorithm.value === 'arima' && !usePretrained.value) {
      let forecast_steps = params.value.forecast_steps
      let train_ratio = params.value.train_ratio
      let p = params.value.p
      let d = params.value.d
      let q = params.value.q
      config.forecast_steps = forecast_steps === '' ? 5 : forecast_steps
      config.train_ratio = train_ratio === '' ? 0.8: train_ratio
      config.p = p === '' ? 1 : p
      config.d = d === '' ? 1 : d
      config.q = q === '' ? 1 : q
    } else if (selectedAlgorithm.value === 'random_forest' && !usePretrained.value) {
      let train_ratio = params.value.train_ratio
      let n_lags = params.value.n_lags
      config.train_ratio = train_ratio !== '' ? train_ratio : 0.8
      config.n_lags = n_lags !== '' ? n_lags : 6
    } else if (selectedAlgorithm.value === 'lstm' && !usePretrained.value) {
      let train_ratio = params.value.train_ratio
      let look_back = params.value.look_back
      let epochs = params.value.epochs
      let batch_size = params.value.batch_size
      let lstm_units = params.value.lstm_units
      config.train_ratio = train_ratio === '' ? 0.8 : train_ratio
      config.look_back = look_back === '' ? 60 : look_back
      config.epochs = epochs === '' ? 50 : epochs
      config.batch_size = batch_size === '' ? 64 : batch_size
      config.lstm_units = lstm_units === '' ? 50 : lstm_units
    } else if (selectedAlgorithm.value === 'autoformer' && !usePretrained.value) {
      let look_back = params.value.look_back
      let forecast_horizon = params.value.forecast_horizon
      let roll_steps = params.value.roll_steps
      let epochs = params.value.epochs
      config.look_back = look_back === '' ? 24 : look_back
      config.forecast_horizon = forecast_horizon === '' ? 12 : forecast_horizon
      config.roll_steps = roll_steps === '' ? 100 : roll_steps
      config.epochs = epochs === '' ? 50 : epochs
    } else if (selectedAlgorithm.value === 'patchtst') {
      let train_ratio = params.value.train_ratio
      let seq_len = params.value.seq_len
      let pred_len = params.value.pred_len
      let patch_len = params.value.patch_len
      let stride = params.value.stride
      let epochs = params.value.epochs
      let batch_size = params.value.batch_size
      config.train_ratio = train_ratio === '' ? 0.8 : train_ratio
      config.seq_len = seq_len === '' ? 96 : seq_len
      config.pred_len = pred_len === '' ? 24 : pred_len
      config.patch_len = patch_len === '' ? 16 : patch_len
      config.stride = stride === '' ? 8 : stride
      config.epochs = epochs === '' ? 10 : epochs
      config.batch_size = batch_size === '' ? 32 : batch_size
    }
    return config
}

const selectChart = () => {
    if (selectedAlgorithm.value === 'arima' && usePretrained.value) {
        currentComponent.value = ARIMAUserPreTrainedCharts
    } else if (selectedAlgorithm.value === 'random_forest' && usePretrained.value) {
        currentComponent.value = MLUserPreTrainedCharts
    } else if (selectedAlgorithm.value === 'lstm' && usePretrained.value) {
        currentComponent.value = DLUserPreTrainedCharts
    } else if (selectedAlgorithm.value === 'autoformer' && usePretrained.value) {
        currentComponent.value = TransformerUserPreTrainedCharts
    } else if (selectedAlgorithm.value === 'arima' && !usePretrained.value) {
      currentComponent.value = ARIMACharts
    } else if (selectedAlgorithm.value === 'random_forest' && !usePretrained.value) {
      currentComponent.value = MLCharts
    } else if (selectedAlgorithm.value === 'lstm' && !usePretrained.value) {
      currentComponent.value = DLCharts
    } else if (selectedAlgorithm.value === 'autoformer' && !usePretrained.value) {
      currentComponent.value = AutoformerCharts
    } else if (selectedAlgorithm.value === 'patchtst') {
      currentComponent.value = PatchTSTCharts
    }
}

const runPrediction = async () => {
  const config = await generateConfig()
    const response = await api.post("/api/future-prediction/", config)
    const data = response.data
    console.log(data)
    await selectChart()
  if (data.code === 200) {
    hasRun.value = true
    metrics.value = data.result.metrics
    if (usePretrained.value) {
      chartData.value = data.result.chartData
    } else {
      chartData.value = data.result.plots.main
      console.log(chartData.value)
      partChartData.value = data.result.plots.zoom
    }
  }
  console.log(chartData.value)
}

// 初始化：确保初始算法在可用选项中
watch(algorithmOptions, (newOptions) => {
  if (!newOptions.some(opt => opt.value === selectedAlgorithm.value)) {
    selectedAlgorithm.value = newOptions[0]?.value || ''
  }
}, { immediate: true })

</script>

<style scoped>
.container {
  display: flex;
  height: 100vh;
  font-family: Arial, sans-serif;
  padding: 16px;
  box-sizing: border-box;
}

.sidebar {
  width: 320px;
  background-color: #f9f9f9;
  padding: 20px;
  border-radius: 8px;
  margin-right: 20px;
  overflow-y: auto;
}

.form-group {
  margin-bottom: 16px;
}

.form-group label {
  display: block;
  margin-bottom: 6px;
  font-weight: bold;
}

.radio-group {
  display: flex;
  gap: 16px;
}

.radio-group label {
  font-weight: normal;
  cursor: pointer;
}

input[type="number"],
select {
  width: 100%;
  padding: 6px;
  border: 1px solid #ccc;
  border-radius: 4px;
  box-sizing: border-box;
}

button {
  width: 100%;
  padding: 10px;
  background-color: #1890ff;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 16px;
  margin-top: 10px;
}

button:hover {
  background-color: #40a9ff;
}

.main-content {
  flex: 1;
  display: flex;
  flex-direction: column;
}

.chart-placeholder {
  flex: 1;
  background-color: #fafafa;
  border: 1px dashed #ccc;
  border-radius: 8px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 20px;
  margin-bottom: 20px;
}

.mock-chart {
  text-align: center;
  color: #666;
}

.metrics-table table {
  width: 100%;
  border-collapse: collapse;
  background-color: white;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
  border-radius: 8px;
  overflow: hidden;
}

.metrics-table th,
.metrics-table td {
  padding: 12px;
  text-align: center;
  border-bottom: 1px solid #eee;
}

.metrics-table th {
  background-color: #f0f2f5;
  font-weight: bold;
}

.metrics-table tbody tr:last-child td {
  border-bottom: none;
}
</style>