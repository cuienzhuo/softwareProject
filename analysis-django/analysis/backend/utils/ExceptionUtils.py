# 1. 导入必要库
import matplotlib
matplotlib.use('TkAgg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
import matplotlib.dates as mdates
import warnings
import io
from .OssUtils import OssUtils

warnings.filterwarnings('ignore')  # 忽略无关警告

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


class AnomalyDetector:
    def __init__(self, df):
        """初始化异常检测器

        参数:
            df: 包含时间序列数据的DataFrame，第一列为timestamp
        """
        self.df = df.copy()
        # 将timestamp转换为datetime类型
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        # 获取所有地点列名
        self.locations = [col for col in self.df.columns if col != 'timestamp']
        # 核心：建立 method 与 检测函数、中文名称 的映射字典
        self.method_mapping = {
            "iqr": (self.detect_iqr, "IQR四分位距法"),
            "zscore": (self.detect_zscore, "Z-Score法"),
            "isolation_forest": (self.detect_isolation_forest, "孤立森林"),
            "dbscan": (self.detect_dbscan, "DBSCAN聚类")
        }

    # ---------------------- 原有检测方法（保持不变）----------------------
    def detect_iqr(self, column, threshold=1.5):
        data = self.df[column]
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        anomalies = (data < lower_bound) | (data > upper_bound)
        return anomalies.astype(int)

    def detect_zscore(self, column, threshold=3):
        data = self.df[column]
        mean = data.mean()
        std = data.std()
        z_scores = (data - mean) / std
        anomalies = np.abs(z_scores) > threshold
        return anomalies.astype(int)

    def detect_isolation_forest(self, column, contamination=0.01, random_state=42):
        data = self.df[column].values.reshape(-1, 1)
        iso_forest = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=random_state
        )
        predictions = iso_forest.fit_predict(data)
        anomalies = (predictions == -1).astype(int)
        return pd.Series(anomalies, index=self.df.index)

    def detect_dbscan(self, column, eps=0.5, min_samples=5):
        data = self.df[column].values.reshape(-1, 1)
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        predictions = dbscan.fit_predict(data_scaled)
        anomalies = (predictions == -1).astype(int)
        return pd.Series(anomalies, index=self.df.index)

    # ---------------------- 新增：单方法执行与可视化 ----------------------
    def run_single_method(self, location, method):
        """根据指定method执行单个异常检测方法

        参数:
            location: 地点名称（需在self.locations中）
            method: 检测方法标识，可选值：["iqr", "zscore", "isolation_forest", "dbscan"]
            visualize: 是否可视化结果

        返回:
            包含该方法检测结果的DataFrame
        """
        # 1. 验证参数合法性
        if location not in self.locations:
            raise ValueError(f"地点 {location} 不在数据中，可选地点：{self.locations}")
        if method not in self.method_mapping:
            raise ValueError(
                f"方法 {method} 不支持，可选方法：{list(self.method_mapping.keys())}"
            )

        # 2. 从映射字典中获取对应的检测函数和中文名称
        detect_func, method_cn_name = self.method_mapping[method]

        # 3. 执行检测，获取异常标记
        anomaly_series = detect_func(location)  # 调用对应的检测方法

        # 4. 构造结果DataFrame（仅包含当前方法的结果）
        results = pd.DataFrame({
            'timestamp': self.df['timestamp'],
            'value': self.df[location],
            f'{method}_anomaly': anomaly_series  # 列名包含方法标识，便于区分
        })

        return self.visualize_single_method(results, location, method, method_cn_name)


    def visualize_single_method(self, results, location, method, method_cn_name):
        """可视化单个方法的异常检测结果

        参数:
            results: 包含检测结果的DataFrame
            location: 地点名称
            method: 检测方法标识（如"iqr"）
            method_cn_name: 检测方法中文名称（如"IQR四分位距法"）
        """
        # 创建单个子图（替代原2x2布局）
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        fig.suptitle(f'{location} - {method_cn_name} 异常检测结果', fontsize=14)

        # 绘制正常数据（蓝色线）
        ax.plot(results['timestamp'], results['value'],
                color='blue', alpha=0.6, label='正常数据')

        # 提取并绘制异常点（红色散点）
        anomaly_col = f'{method}_anomaly'  # 结果列名
        anomalies = results[results[anomaly_col] == 1]
        ax.scatter(anomalies['timestamp'], anomalies['value'],
                   color='red', s=50, label=f'异常点（共{len(anomalies)}个）')

        # 图表美化
        ax.set_title(f'异常值判定规则：{method_cn_name}', fontsize=12)
        ax.set_xlabel('时间', fontsize=10)
        ax.set_ylabel('数值', fontsize=10)
        ax.legend()
        # 格式化x轴日期（避免重叠）
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.xticks(rotation=45)
        # 调整布局，避免标题被截断
        plt.tight_layout()
        buffer = io.BytesIO()  # 初始化内存缓冲区
        plt.savefig(buffer, dpi=300, bbox_inches='tight', format='png')  # 保存图表到缓冲区（指定格式为png）
        buffer.seek(0)  # 将缓冲区指针移到开头，否则读取不到数据

        save_path = f"exception_{method}/{location}.png"
        ossUtils = OssUtils()
        image_url = ossUtils.OssUpload(buffer, save_path)

        plt.close()

        return image_url


# ---------------------- 对外调用函数（核心入口，根据method动态选择）----------------------
def exceptionAnalysis(address: str, method: str) -> str:
    # 1. 加载数据（确保CSV路径正确，可根据实际情况调整）
    try:
        df = pd.read_csv('E:\github01\softwareProject/analysis-django/analysis/backend/utils/milano_traffic_nid.csv')
    except FileNotFoundError:
        raise FileNotFoundError("数据文件 'milano_traffic_nid.csv' 未找到，请检查路径")

    # 2. 创建检测器实例
    detector = AnomalyDetector(df)

    # 3. 执行单个方法的检测（核心：传入address和method）
    print(f"开始分析：地点={address}，方法={method}")
    return detector.run_single_method(location=address, method=method)