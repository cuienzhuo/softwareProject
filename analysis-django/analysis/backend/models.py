from django.db import models
from datetime import datetime

class AnomalyAnalysis(models.Model):
    """异常数据分析记录"""
    method = models.CharField(max_length=100, verbose_name="分析方法")  # 如"Z-score"、"IQR"
    address = models.CharField(max_length=200, verbose_name="地址信息")
    image_url = models.URLField(max_length=500, verbose_name="分析结果图表URL")
    created_at = models.DateTimeField(default=datetime.utcnow, verbose_name="创建时间")

    class Meta:
        verbose_name = "异常分析记录"
        verbose_name_plural = verbose_name
        indexes = [
            models.Index(fields=['method', 'address']),  # 联合索引加速查询
        ]

class ClusterAnalysis(models.Model):
    """聚类分析记录"""
    address = models.CharField(max_length=200, verbose_name="地址信息")
    image_url = models.URLField(max_length=500, verbose_name="聚类结果图表URL")
    created_at = models.DateTimeField(default=datetime.utcnow, verbose_name="创建时间")

    class Meta:
        verbose_name = "聚类分析记录"
        verbose_name_plural = verbose_name
        indexes = [
            models.Index(fields=['address']),  # 索引加速查询
        ]

class FuturePrediction(models.Model):
    """未来预测记录"""
    method = models.CharField(max_length=100, verbose_name="预测方法")  # 如"ARIMA"、"LSTM"
    address = models.CharField(max_length=200, verbose_name="地址信息")
    image_url = models.URLField(max_length=500, verbose_name="预测结果图表URL")
    created_at = models.DateTimeField(default=datetime.utcnow, verbose_name="创建时间")

    class Meta:
        verbose_name = "未来预测记录"
        verbose_name_plural = verbose_name
        indexes = [
            models.Index(fields=['method', 'address']),  # 联合索引加速查询
        ]