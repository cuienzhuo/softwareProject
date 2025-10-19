<template>
  <div class="map-container">
    <h1>米兰地区交互式地图</h1>
    
    <!-- 加载状态 -->
    <div v-if="loading" class="loading">加载中...</div>
    
    <!-- 错误信息 -->
    <div v-if="error" class="error">{{ error }}</div>
    
    <!-- 地图容器 -->
    <div v-else class="svg-container">
      <svg 
        :width="svgWidth" 
        :height="svgHeight" 
        @mousemove="handleMouseMove"
        @click="handleSvgClick"
      >
        <!-- 绘制所有区域 -->
        <g>
          <path
            v-for="(feature, index) in features"
            :key="index"
            :d="getPathData(feature.geometry.coordinates)"
            :fill="selectedFeature === index ? '#4CAF50' : getAreaColor(index)"
            :stroke="'#333'"
            :stroke-width="1"
            :class="{ 'hover-area': hoveredFeature === index }"
            @click.stop="handleAreaClick(index)"
            @mouseover.stop="handleAreaMouseOver(index)"
            @mouseout.stop="handleAreaMouseOut"
          />
        </g>
        
        <!-- 显示选中区域信息 -->
        <div 
          v-if="selectedFeature !== null" 
          class="info-box"
          :style="{ left: infoBoxX + 'px', top: infoBoxY + 'px' }"
        >
          <h3>{{ features[selectedFeature].properties.name }}</h3>
        </div>
      </svg>
    </div>
    
    <!-- 区域信息面板 -->
    <div class="info-panel">
      <h2>区域信息</h2>
      <p v-if="selectedFeature === null">请点击地图上的区域查看信息</p>
      <div v-else>
        <h3>{{ features[selectedFeature].properties.name }}</h3>
        <pre>{{ JSON.stringify(features[selectedFeature].properties, null, 2) }}</pre>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue';

// 地图数据
const features = ref([]);
const loading = ref(true);
const error = ref(null);

// SVG尺寸
const svgWidth = 1000;
const svgHeight = 800;

// 坐标转换相关
const minLat = ref(Infinity);
const maxLat = ref(-Infinity);
const minLng = ref(Infinity);
const maxLng = ref(-Infinity);

// 交互状态
const selectedFeature = ref(null);
const hoveredFeature = ref(null);
const mousePosition = ref({ x: 0, y: 0 });
const infoBoxX = ref(0);
const infoBoxY = ref(0);

// 计算所有坐标的边界范围
const calculateBounds = () => {
  features.value.forEach(feature => {
    if (feature.geometry.type === 'Polygon') {
      processCoordinates(feature.geometry.coordinates);
    } else if (feature.geometry.type === 'MultiPolygon') {
      feature.geometry.coordinates.forEach(polygon => {
        processCoordinates(polygon);
      });
    }
  });
};

// 处理坐标数组，更新边界范围
const processCoordinates = (coordinates) => {
  coordinates.forEach(ring => {
    ring.forEach(([lng, lat]) => {
      minLng.value = Math.min(minLng.value, lng);
      maxLng.value = Math.max(maxLng.value, lng);
      minLat.value = Math.min(minLat.value, lat);
      maxLat.value = Math.max(maxLat.value, lat);
    });
  });
};

// 将经纬度转换为SVG坐标
const convertLngLatToSvg = (lng, lat) => {
  const xRatio = (lng - minLng.value) / (maxLng.value - minLng.value);
  const yRatio = 1 - (lat - minLat.value) / (maxLat.value - minLat.value);
  
  const margin = 40;
  const availableWidth = svgWidth - margin * 2;
  const availableHeight = svgHeight - margin * 2;
  
  return {
    x: margin + xRatio * availableWidth,
    y: margin + yRatio * availableHeight
  };
};

// 生成路径数据
const getPathData = (coordinates) => {
  if (Array.isArray(coordinates[0][0][0])) {
    return coordinates.map(polygon => getPolygonPath(polygon)).join(' ');
  } else if (Array.isArray(coordinates[0][0])) {
    return getPolygonPath(coordinates);
  }
  return '';
};

// 生成多边形路径
const getPolygonPath = (polygon) => {
  let path = '';
  polygon.forEach((ring) => {
    ring.forEach(([lng, lat], pointIndex) => {
      const { x, y } = convertLngLatToSvg(lng, lat);
      path += pointIndex === 0 ? `M ${x} ${y} ` : `L ${x} ${y} `;
    });
    path += 'Z ';
  });
  return path;
};

// 获取区域颜色
const getAreaColor = (index) => {
  const hue = (index * 37) % 360;
  return `hsl(${hue}, 60%, 80%)`;
};

// 处理区域点击 - 新增console.log打印区域名称
const handleAreaClick = (index) => {
  selectedFeature.value = index;
  // 打印区域名称到控制台
  const areaName = features.value[index].properties.NIL;
  console.log('选中的区域名称:', areaName);
  updateInfoBoxPosition();
};

// 处理鼠标移动
const handleMouseMove = (event) => {
  const rect = event.currentTarget.getBoundingClientRect();
  mousePosition.value = {
    x: event.clientX - rect.left,
    y: event.clientY - rect.top
  };
  
  if (selectedFeature.value !== null) {
    updateInfoBoxPosition();
  }
};

// 更新信息框位置
const updateInfoBoxPosition = () => {
  infoBoxX.value = Math.min(mousePosition.value.x + 10, svgWidth - 200);
  infoBoxY.value = Math.min(mousePosition.value.y + 10, svgHeight - 100);
};

// 处理SVG点击（取消选择）
const handleSvgClick = () => {
  selectedFeature.value = null;
};

// 处理区域鼠标悬停
const handleAreaMouseOver = (index) => {
  hoveredFeature.value = index;
};

// 处理区域鼠标离开
const handleAreaMouseOut = () => {
  hoveredFeature.value = null;
};

// 组件挂载时加载数据
onMounted(async () => {
  try {
    const response = await fetch('milano.geojson');
    if (!response.ok) throw new Error('无法加载GeoJSON文件');
    
    const geojson = await response.json();
    features.value = geojson.features;
    calculateBounds();
  } catch (err) {
    error.value = err.message;
  } finally {
    loading.value = false;
  }
});
</script>

<style scoped>
/* 样式与之前保持一致 */
.map-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 20px;
  font-family: Arial, sans-serif;
}

.svg-container {
  border: 1px solid #ccc;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
  background-color: #f9f9f9;
  position: relative;
}

path {
  transition: all 0.3s ease;
  cursor: pointer;
}

path.hover-area {
  fill-opacity: 0.7;
  stroke-width: 2;
  stroke: #000;
}

.loading, .error {
  font-size: 1.2em;
  padding: 20px;
  color: #666;
}

.error {
  color: #dc3545;
}

.info-box {
  position: absolute;
  background-color: white;
  border: 1px solid #ccc;
  border-radius: 5px;
  padding: 10px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
  font-size: 0.9em;
  pointer-events: none;
  z-index: 10;
}

.info-panel {
  margin-top: 20px;
  width: 1000px;
  border-top: 1px solid #ccc;
  padding-top: 20px;
}

.info-panel pre {
  background-color: #f5f5f5;
  padding: 10px;
  border-radius: 5px;
  overflow-x: auto;
  max-height: 300px;
}
</style>