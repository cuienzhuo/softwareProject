import matplotlib
matplotlib.use('TkAgg')  # 切换到TkAgg后端（稳定且兼容性好）

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.dates as mdates
import warnings
import io
from .OssUtils import OssUtils

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


class KMeansClustering:
    def __init__(self, df, n_clusters=4):
        """初始化K均值聚类分析器"""
        self.df = df.copy()
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.features = [col for col in self.df.columns if col != 'timestamp']
        self.X = self.df[self.features].copy()
        self.n_clusters = n_clusters
        self.model = None
        self.clusters = None
        self.scaler = StandardScaler()

    def preprocess_data(self):
        """数据预处理：标准化特征"""
        self.X_scaled = self.scaler.fit_transform(self.X)
        return self.X_scaled

    def perform_clustering(self, n_clusters=None):
        """执行K均值聚类"""
        if n_clusters:
            self.n_clusters = n_clusters

        if not hasattr(self, 'X_scaled'):
            self.preprocess_data()

        self.model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.clusters = self.model.fit_predict(self.X_scaled)
        self.df['cluster'] = self.clusters

        return self.clusters

    def visualize_cluster_by_address(self, address):
        """根据指定地址可视化不同簇的时间序列特征"""
        if self.clusters is None:
            raise ValueError("请先执行聚类分析（调用perform_clustering方法）")

        # 验证地址是否存在于特征中
        if address not in self.features:
            raise ValueError(f"地址 '{address}' 不在数据特征中，请检查输入")

        # 创建可视化图表
        plt.figure(figsize=(15, 7))

        # 绘制每个簇的时间序列
        for cluster_id in range(self.n_clusters):
            cluster_data = self.df[self.df['cluster'] == cluster_id]
            plt.plot(
                cluster_data['timestamp'],
                cluster_data[address],
                '.',
                alpha=0.6,
                label=f'簇 {cluster_id} (样本数={len(cluster_data)})'
            )

        plt.title(f'{address} 在不同聚类中的时间序列分布')
        plt.xlabel('时间')
        plt.ylabel('数值')
        plt.legend()
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.xticks(rotation=45)
        plt.tight_layout()

        buffer = io.BytesIO()  # 初始化内存缓冲区
        plt.savefig(buffer, dpi=300, bbox_inches='tight', format='png')  # 保存图表到缓冲区（指定格式为png）
        buffer.seek(0)  # 将缓冲区指针移到开头，否则读取不到数据

        save_path = f"cluster/{address}.png"
        ossUtils = OssUtils()
        image_url = ossUtils.OssUpload(buffer, save_path)

        plt.close()
        return image_url

# 主函数
def clusterAnalysis(address: str) -> str:
    # 从CSV文件读取数据
    df = pd.read_csv('E:\github01\softwareProject/analysis-django/analysis/backend/utils/milano_traffic_nid.csv')

    # 创建K均值聚类分析器实例
    kmeans_analyzer = KMeansClustering(df)

    # 选择合适的k值（可根据实际数据调整）
    optimal_k = 4

    # 执行K均值聚类
    kmeans_analyzer.perform_clustering(n_clusters=optimal_k)

    # 根据输入的address可视化结果
    return kmeans_analyzer.visualize_cluster_by_address(address=address)