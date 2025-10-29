from django.db import transaction
from .models import AnomalyAnalysis, ClusterAnalysis, FuturePrediction
from .utils import ARIMAUtils,DLUtils,MLUtils,TransformerUtils,ExceptionUtils,ClusterUtils

class ImageService:
    """图表存储服务（模拟图床功能，实际可对接阿里云OSS等）"""
    @staticmethod
    def upload_image(image_data=None):
        """上传图表到图床，返回URL（此处为模拟）"""
        # 实际开发中需替换为真实图床API调用
        return "https://example.com/charts/sample.png"

class AnalysisService:
    """分析业务服务"""
    @staticmethod
    @transaction.atomic  # 事务确保数据一致性
    def anomaly_analysis(address: str, method: str) -> str:
        """执行异常数据分析，返回结果图表URL"""
        # 1. 检查是否已有相同记录（避免重复计算）
        print("检查数据库")
        existing = AnomalyAnalysis.objects.filter(method=method, address=address).first()
        if existing:
            return existing.image_url
        print("进入分析部分")

        # 生成图标并上传
        image_url = ExceptionUtils.exceptionAnalysis(address,method)
        
        # 4. 保存记录到数据库
        AnomalyAnalysis.objects.create(
            method=method,
            address=address,
            image_url=image_url
        )
        return image_url

    @staticmethod
    @transaction.atomic
    def cluster_analysis(address: str) -> str:
        """执行聚类分析，返回结果图表URL"""
        existing = ClusterAnalysis.objects.filter(address=address).first()
        if existing:
            return existing.image_url

        image_url = ClusterUtils.clusterAnalysis(address)

        # 保存记录
        ClusterAnalysis.objects.create(
            address=address,
            image_url=image_url
        )
        return image_url

    @staticmethod
    @transaction.atomic
    def future_prediction(address: str, method: str) -> str:
        """执行未来预测，返回结果图表URL"""
        existing = FuturePrediction.objects.filter(method=method, address=address).first()
        if existing:
            return existing.image_url

        if method == "arima":
            image_url = ARIMAUtils.arimaAnalysis(address)
        elif method == "ML":
            image_url = MLUtils.MLAnalysis(address)
        elif method == "DL":
            image_url = DLUtils.DLAnalysis(address)
        elif method == "transformer":
            image_url = TransformerUtils.transformerAnalysis(address)
        else:
            raise ValueError("输入方法有误")

        # 保存记录
        FuturePrediction.objects.create(
            method=method,
            address=address,
            image_url=image_url
        )
        return image_url