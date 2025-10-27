from django.urls import path
from . import views  # 导入当前应用的视图

# 应用内的路由列表
urlpatterns = [
    path('anomaly-analysis/', views.anomaly_analysis_view, name='anomaly_analysis'),
    path('cluster-analysis/', views.cluster_analysis_view, name='cluster_analysis'),
    path('future-prediction/', views.future_prediction_view, name='future_prediction'),
]