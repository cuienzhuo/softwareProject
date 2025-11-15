
import pandas as pd
from sklearn.cluster import KMeans

def clusterAnalysis(n_clusters):
    # 1. 加载并处理交通数据
    df = pd.read_csv('E:\github01\softwareProject/analysis-django/analysis/backend/utils/milano_traffic_nid.csv', encoding='utf-8')
    df = df.set_index('timestamp')
    df_T = df.T

    # 标准化：对每个地区的时间序列做 z-score（按行）
    df_T_scaled = df_T.apply(lambda x: (x - x.mean()) / x.std(), axis=1)

    kmeans_final = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans_final.fit_predict(df_T_scaled)
    region_clusters = pd.Series(clusters, index=df_T.index, name='cluster')
    return region_clusters