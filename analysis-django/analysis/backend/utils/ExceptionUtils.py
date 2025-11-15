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

    def detect_isolation_forest(self, column,n_estimators=100, contamination=0.01):
        data = self.df[column].values.reshape(-1, 1)
        iso_forest = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=42
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
    def run_single_method(self,data):
        print(data)
        # 1. 验证参数合法性
        location = data["address"]
        method = data["method"]
        if location not in self.locations:
            raise ValueError(f"地点 {location} 不在数据中，可选地点：{self.locations}")
        if method not in self.method_mapping:
            raise ValueError(
                f"方法 {method} 不支持，可选方法：{list(self.method_mapping.keys())}"
            )

        # 2. 从映射字典中获取对应的检测函数和中文名称
        detect_func, method_cn_name = self.method_mapping[method]
        print("寻找方法部分")

        if method == 'iqr' or method == 'zscore':
            anomaly_series = self.detect_iqr(location, threshold=data["threshold"])
        elif method == 'isolation_forest':
            anomaly_series = self.detect_isolation_forest(location, contamination=data["contamination"], n_estimators=data["n_estimators"])
        else:
            anomaly_series = self.detect_dbscan(location, eps=data["eps"],min_samples=data["min_samples"])

        print("分析结束")
        # 4. 构造结果DataFrame（仅包含当前方法的结果）
        results = pd.DataFrame({
            'timestamp': self.df['timestamp'],
            'value': self.df[location],
            f'{method}_anomaly': anomaly_series  # 列名包含方法标识，便于区分
        })

        if anomaly_series.dtype == 'bool':
            total_count = len(anomaly_series)
            anomaly_count = anomaly_series.sum()  # True 视为 1
        else:
            total_count = len(anomaly_series)
            anomaly_count = (anomaly_series == 1).sum()

        anomaly_ratio = anomaly_count / total_count

        print("返回值")
        return location,method,method_cn_name,results,total_count,anomaly_count,anomaly_ratio


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

    def compare_methods_for_location(self, location):
        if location not in self.locations:
            raise ValueError(f"地点 {location} 不在数据中，可选地点：{self.locations}")

        method_names = []
        anomaly_counts = []

        for method_key, (detect_func, method_cn_name) in self.method_mapping.items():
            try:
                anomaly_series = detect_func(location)
                count = (int)(anomaly_series.sum())
                method_names.append(method_cn_name)
                anomaly_counts.append(count)
            except Exception as e:
                print(f"方法 {method_key} 在地点 {location} 上执行失败: {e}")
                method_names.append(method_cn_name)
                anomaly_counts.append(0)
        return method_names,anomaly_counts

    def compare_locations_for_method(self, method):
        if method not in self.method_mapping:
            raise ValueError(
                f"方法 {method} 不支持，可选方法：{list(self.method_mapping.keys())}"
            )

        detect_func, method_cn_name = self.method_mapping[method]
        locations = []
        anomaly_counts = []

        for loc in self.locations:
            try:
                anomaly_series = detect_func(loc)
                count = (int)(anomaly_series.sum())
                locations.append(loc)
                anomaly_counts.append(count)
            except Exception as e:
                print(f"方法 {method} 在地点 {loc} 上执行失败: {e}")
                locations.append(loc)
                anomaly_counts.append(0)
        return locations,anomaly_counts

        # # 排序（可选）：按异常数量降序
        # sorted_pairs = sorted(zip(locations, anomaly_counts), key=lambda x: x[1], reverse=True)
        # locations_sorted, counts_sorted = zip(*sorted_pairs) if sorted_pairs else ([], [])
        #
        # # 绘制柱状图
        # fig, ax = plt.subplots(figsize=(12, 6))
        # bars = ax.bar(locations_sorted, counts_sorted, color='lightcoral', edgecolor='darkred')
        # ax.set_title(f'各地区异常点数量对比（{method_cn_name}）', fontsize=14)
        # ax.set_ylabel('异常点数量', fontsize=12)
        # ax.set_xlabel('地区', fontsize=12)
        # ax.tick_params(axis='x', rotation=45)
        #
        # # 在柱子上显示数值（仅当数量>0）
        # for bar, count in zip(bars, counts_sorted):
        #     if count > 0:
        #         ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts_sorted)*0.01,
        #                 str(count), ha='center', va='bottom', fontsize=9)
        #
        # plt.tight_layout()
        #
        # buffer = io.BytesIO()
        # plt.savefig(buffer, dpi=300, bbox_inches='tight', format='png')
        # buffer.seek(0)
        # plt.close()
        #
        # save_path = f"exception_compare/locations_{method}.png"
        # ossUtils = OssUtils()
        # image_url = ossUtils.OssUpload(buffer, save_path)
        # return image_url


def exceptionAnalysis(data):
    # 1. 加载数据（确保CSV路径正确，可根据实际情况调整）
    try:
        df = pd.read_csv('E:\github01\softwareProject/analysis-django/analysis/backend/utils/milano_traffic_nid.csv')
    except FileNotFoundError:
        raise FileNotFoundError("数据文件 'milano_traffic_nid.csv' 未找到，请检查路径")

    # 2. 创建检测器实例
    detector = AnomalyDetector(df)
    print("进入跑单个方法部分")
    return detector.run_single_method(data)

def compareMethodsForLocation(address: str):
    df = pd.read_csv('E:\github01\softwareProject/analysis-django/analysis/backend/utils/milano_traffic_nid.csv')
    detector = AnomalyDetector(df)
    return detector.compare_methods_for_location(address)


def compareLocationsForMethod(method: str):
    df = pd.read_csv('E:\github01\softwareProject/analysis-django/analysis/backend/utils/milano_traffic_nid.csv')
    detector = AnomalyDetector(df)
    return detector.compare_locations_for_method(method)