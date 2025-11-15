<template>
  <div class="cluster-container">
    <!-- 左侧：参数设置 -->
    <div class="settings-panel">
      <label class="setting-label">聚簇个数</label>
      <input
        v-model.number="clusterCount"
        type="number"
        min="1"
        @change="triggerClustering"
        class="setting-input"
      />
      <button
        class="generate-btn"
        :disabled="isLoading"
        @click="generateChart"
      >
        {{ isLoading ? '运行中...' : '运行聚簇' }} <!-- 添加加载状态文本 -->
      </button>
    </div>

    <!-- 右侧：聚簇结果展示（图片） -->
    <div class="result-panel">
      <h3>聚簇结果</h3>
      <div class="image-container">
        <div class="map-container" ref="mapContainer"></div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, defineProps, computed, onMounted,watch } from 'vue'
import api from '@/api'
import "leaflet/dist/leaflet.css";
import L from 'leaflet';
import geojsonData from '@/assets/milano.json'; // 替换为你的实际路径

const props = defineProps({
  regions: {
    type: Array,
    required: true
  }
})

const clusterData = ref([])
const isLoading = ref(false)
const clusterCount = ref(3) // 默认聚簇个数

const mapContainer = ref(null);
let map = null;

// 构建 NIL -> category 的映射
const buildCategoryMap = (data) => {
  const map = {};
  data.forEach(item => {
    map[item.NIL] = item.cluster; // 假设字段名为 category
  });
  return map;
};

// 颜色映射表，根据你的聚类数量和偏好进行定义
const clusterColors = ['#E6194B', '#3CB44B','#4363D8','#F58231','#911EB4','#46F0F0','#FCC052','#F032E6','#BFEF45','#FABED4']

// 获取颜色
const getColor = (cluster) => {
  const idx = parseInt(cluster)
  if (idx >= 0 && idx <= 9) {
    return clusterColors[idx];
  }
  return '#cccccc'; // 默认灰色
};

// 初始化地图
const initMap = () => {
  if (!mapContainer.value) return;

  map = L.map(mapContainer.value).setView([45.435, 9.135], 13); // 米兰中心

  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; OpenStreetMap contributors'
  }).addTo(map);
};

// 渲染 GeoJSON
const renderGeoJson = () => {
  if (!map) return;

  // 清除已有图层（避免重复）
  map.eachLayer((layer) => {
    if (layer instanceof L.GeoJSON) layer.remove();
  });

  const categoryMap = buildCategoryMap(clusterData.value);

  const geoJsonLayer = L.geoJSON(geojsonData, {
    style: (feature) => {
      const nil = feature.properties?.NIL;
      const category = categoryMap[nil] ?? '-1'; // 未匹配则用 -1
      return {
        fillColor: getColor(category),
        weight: 1,
        opacity: 1,
        color: '#333',
        dashArray: '3',
        fillOpacity: 0.7
      };
    },
    onEachFeature: (feature, layer) => {
      const nil = feature.properties?.NIL || 'Unknown';
      const category = categoryMap[nil] ?? 'N/A';
      layer.bindPopup(`<b>${nil}</b><br>Category: ${category}`);
    }
  }).addTo(map);
};

onMounted(() => {
  initMap();
  renderGeoJson();
});

// 监听 backendData 变化（如异步加载完成）
watch(() => clusterData.value, () => {
  renderGeoJson();
}, { deep: true });

const generateChart = async () => {
  isLoading.value = true
  try {
    const response = await api.post("/api/cluster-analysis/", {
      clusters: clusterCount.value
    })
    const data = response.data
    if (data.code === 200) {
      clusterData.value = data.clusterData
    } else {
      console.error("API returned an error:", data.message);
      clusterData.value = [];
    }
  } catch (error) {
    console.error("Error fetching cluster data:", error);
    clusterData.value = [];
  } finally {
    isLoading.value = false
  }
}

</script>

<style scoped>
.cluster-container {
  display: flex;
  height: 100%;
  gap: 20px;
  padding: 20px;
  box-sizing: border-box;
}

.settings-panel {
  width: 250px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.setting-label {
  font-weight: bold;
  font-size: 14px;
  display: block;
}

.setting-input {
  padding: 8px 10px;
  border: 1px solid #ccc;
  border-radius: 4px;
  font-size: 14px;
}

.run-button {
  padding: 8px 16px;
  background-color: #1890ff;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
  transition: background-color 0.2s;
}

.run-button:hover {
  background-color: #40a9ff;
}

.result-panel {
  flex: 1;
  border: 1px solid #ddd;
  border-radius: 6px;
  padding: 16px;
  background-color: #fafafa;
  display: flex;
  flex-direction: column;
}

.result-panel h3 {
  margin-top: 0;
  margin-bottom: 12px;
  font-size: 16px;
  color: #333;
}

.image-container {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 300px;
  border: 1px dashed #ccc;
  border-radius: 4px;
  background-color: #fff;
}

.cluster-image {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
}

.placeholder-text {
  color: #999;
  font-style: italic;
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
.map-container {
  width: 100%;
  height: 600px;
  border: 1px solid #ccc;
}
</style>