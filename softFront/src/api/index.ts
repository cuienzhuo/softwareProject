import axios from 'axios';

// 创建axios实例
const api = axios.create({
  baseURL: 'http://127.0.0.1:8000',
  headers: {
    'Content-Type': 'application/json'
  }
});

// 请求拦截器 - 添加token
api.interceptors.request.use(config => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = token || '';
  }
  return config;
}, error => {
  return Promise.reject(error);
});

// 响应拦截器 - 处理通用错误
api.interceptors.response.use(response => {
  return response;
}, error => {
  console.log(error)
  if (error.response.status === 401) {
    // 处理未授权错误
    console.error('未授权，请重新登录');
    // 可以在这里跳转到登录页
  }
  return Promise.reject(error);
});

export default api;