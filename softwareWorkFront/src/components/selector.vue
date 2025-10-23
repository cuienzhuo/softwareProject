<template>
  <div class="selector-wrapper">
    <!-- 类型标题 - 调整后样式 -->
    <label class="selector-label" v-if="title">{{ title }}</label>
    
    <!-- 选择器主体 -->
    <div class="custom-select-container" :class="{ open: isOpen }">
      <!-- 选择器触发按钮 -->
      <button 
        ref="triggerRef"
        class="select-trigger"
        @click="toggleSelect"
        aria-expanded="isOpen"
      >
        <span class="selected-value">
          {{ selectedValue || placeholder }}
        </span>
        <svg class="select-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M7 10L12 15L17 10" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </button>
      
      <!-- 下拉选项列表 -->
      <div class="select-dropdown" v-if="isOpen">
        <ul class="options-list">
          <li 
            v-for="(option, index) in options" 
            :key="index"
            class="option-item"
            :class="{ 
              'option-selected': selectedValue === option.label,
              'option-disabled': option.disabled
            }"
            @click="handleOptionClick(option)"
            :tabindex="option.disabled ? -1 : 0"
            :aria-disabled="option.disabled"
          >
            {{ option.label }}
          </li>
        </ul>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, defineProps, defineEmits, watch } from 'vue';

// 定义 props
const props = defineProps({
  // 选择器标题
  title: {
    type: String,
    default: ''
  },
  // 选项列表（支持对象格式 { label: string, value: string, disabled?: boolean }）
  options: {
    type: Array,
    required: true,
    validator: (value) => {
      return value.every(item => item.label && item.value !== undefined);
    }
  },
  // 占位符（未选择时显示）
  placeholder: {
    type: String,
    default: '请选择'
  },
  // v-model 绑定值（现在绑定的是label）
  modelValue: {
    type: String,
    default: ''
  }
});

// 定义事件
const emit = defineEmits(['update:modelValue', 'change']);

// 状态管理 - selectedValue与label保持一致
const isOpen = ref(false);
const selectedValue = ref(props.modelValue);
const triggerRef = ref(null);

// 切换选择器展开/收起
const toggleSelect = () => {
  isOpen.value = !isOpen.value;
};

// 处理选项点击 - 确保selectedValue设置为label
const handleOptionClick = (option) => {
  if (option.disabled) return;
  
  // 更新选中值为label并通知父组件
  selectedValue.value = option.label;
  emit('update:modelValue', option.label); // 向父组件传递label
  emit('change', option); // 额外抛出包含完整选项的事件
  isOpen.value = false;
};

// 监听modelValue变化，确保与label同步
watch(() => props.modelValue, (newVal) => {
  // 验证新值是否是有效的label
  const isValidLabel = props.options.some(option => option.label === newVal);
  if (isValidLabel || newVal === '') {
    selectedValue.value = newVal;
  }
});

// 点击外部关闭选择器
const handleClickOutside = (event) => {
  if (triggerRef.value && !triggerRef.value.contains(event.target)) {
    isOpen.value = false;
  }
};

// 键盘导航支持
const handleKeydown = (event) => {
  if (!isOpen.value) return;
  
  // ESC键关闭
  if (event.key === 'Escape') {
    isOpen.value = false;
    triggerRef.value?.focus();
  }
};

// 生命周期钩子
onMounted(() => {
  document.addEventListener('click', handleClickOutside);
  document.addEventListener('keydown', handleKeydown);
});

onUnmounted(() => {
  document.removeEventListener('click', handleClickOutside);
  document.removeEventListener('keydown', handleKeydown);
});
</script>

<style scoped>
.selector-wrapper {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  width: 100%;
  height: 350px;
  font-family: 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
}

/* 调整标题样式 - 放大字体大小和粗细 */
.selector-label {
    width: 1000px;
  font-size: 18px; /* 增大字体大小 */
  font-weight: 600; /* 加粗 */
  color: #1e293b; /* 稍深颜色增强视觉效果 */
  line-height: 1.5;
  margin-bottom: 4px;
}

/* 选择器容器 */
.custom-select-container {
  position: relative;
  width: 1000px;
}

/* 选择器触发按钮 */
.select-trigger {
  width: 100%;
  padding: 12px 16px;
  background-color: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  color: #1e293b;
  font-size: 14px;
  text-align: left;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: space-between;
  transition: all 0.2s ease;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
}

.select-trigger:hover {
  border-color: #94a3b8;
}

.select-trigger:focus {
  outline: none;
  border-color: #3b82f6;
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}

.selected-value {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: #0f172a;
}

/* 未选择时的占位符样式 */
.selected-value:empty::before {
  content: attr(data-placeholder);
  color: #94a3b8;
}

.select-icon {
  transition: transform 0.2s ease;
  color: #64748b;
  flex-shrink: 0;
  margin-left: 8px;
}

.open .select-icon {
  transform: rotate(180deg);
}

/* 下拉选项列表 */
.select-dropdown {
  position: absolute;
  top: calc(100% + 4px);
  left: 0;
  right: 0;
  background-color: #ffffff;
  border-radius: 8px;
  border: 1px solid #e2e8f0;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
  z-index: 50;
  max-height: 0;
  overflow: hidden;
  transition: max-height 0.2s ease-out, opacity 0.2s ease-out;
  opacity: 0;
}

.open .select-dropdown {
  max-height: 250px;
  opacity: 1;
  transition: max-height 0.3s ease-in, opacity 0.2s ease-in;
}

.options-list {
  list-style: none;
  padding: 6px 0;
  margin: 0;
  max-height: 250px;
  overflow-y: auto;
}

/* 选项样式 */
.option-item {
  padding: 10px 16px;
  cursor: pointer;
  transition: background-color 0.15s ease;
  user-select: none;
  font-size: 14px;
  color: #1e293b;
}

.option-item:hover:not(.option-selected):not(.option-disabled) {
  background-color: #f1f5f9;
  color: #0f172a;
}

.option-item:focus {
  outline: none;
  background-color: #e2e8f0;
}

.option-selected {
  background-color: #eff6ff;
  color: #1e40af;
  font-weight: 500;
}

.option-selected::after {
  content: "✓";
  float: right;
  margin-left: 8px;
  color: #3b82f6;
}

.option-disabled {
  color: #94a3b8;
  cursor: not-allowed;
  background-color: transparent !important;
  opacity: 0.7;
}

/* 滚动条美化 */
.options-list::-webkit-scrollbar {
  width: 6px;
}

.options-list::-webkit-scrollbar-track {
  background: #f1f5f9;
  border-radius: 3px;
}

.options-list::-webkit-scrollbar-thumb {
  background: #cbd5e1;
  border-radius: 3px;
}

.options-list::-webkit-scrollbar-thumb:hover {
  background: #94a3b8;
}
</style>