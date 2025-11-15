from django.db import transaction
from .models import AnomalyAnalysis, ClusterAnalysis, FuturePrediction
from .utils import ARIMAUtils, DLUtils, MLUtils, TransformerUtils, ExceptionUtils, ClusterUtils, NotPreTrainTransformerr,PatchTSTUtils


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
    def anomaly_analysis(data):
        """执行异常数据分析，返回结果图表URL"""
        print("进入分析部分")

        return ExceptionUtils.exceptionAnalysis(data)

    @staticmethod
    @transaction.atomic
    def cluster_analysis(clusters):
        return ClusterUtils.clusterAnalysis(clusters)

    @staticmethod
    @transaction.atomic
    def future_prediction(data):
        address = data['address']
        method = data['method']
        usePreTrained = data['usePretrained']
        if method == "arima" and usePreTrained:
            return ARIMAUtils.arimaAnalysis(address, data['forecast_steps'])
        elif method == "random_forest" and usePreTrained:
            return MLUtils.MLAnalysis(address, data["backtest_start_pct"], data["forecast_steps"])
        elif method == "lstm" and usePreTrained:
            return DLUtils.DLAnalysis(address, data['roll_steps'])
        elif method == "autoformer" and usePreTrained:
            return TransformerUtils.transformerAnalysis(address, data['roll_steps'])
        elif method == "arima" and not usePreTrained:
            return ARIMAUtils.arimaAnalysisWithoutPreTrained(address, data['train_ratio'], data['p'],
                                                             data['d'], ['q'], data['forecast_steps'])
        elif method == "random_forest" and not usePreTrained:
            return MLUtils.MLAnalysisWithoutPreTrain(address, data['train_ratio'], data['n_lags'])
        elif method == "lstm" and not usePreTrained:
            return DLUtils.DLAnalysisWithoutPreTrain(address, data['train_ratio'], data['look_back'], data['epochs'],
                                                     data['batch_size'], data['lstm_units'])
        elif method == "autoformer" and not usePreTrained:
            return NotPreTrainTransformerr.main(address, data['look_back'], data["forecast_horizon"]
                                                , data["epochs"], data["roll_steps"])
        elif method == "patchtst":
            return PatchTSTUtils.patchtstAnalysis(address,data['train_ratio'],data['seq_len'],data['pred_len'],
                                                  data['patch_len'],data['stride'],data['epochs'],data['batch_size'])
        else:
            raise ValueError("输入方法有误")

    @staticmethod
    @transaction.atomic  # 事务确保数据一致性
    def anomaly_compare(data):
        if 'address' in data:
            return ExceptionUtils.compareMethodsForLocation(data['address'])
        if 'method' in data:
            return ExceptionUtils.compareLocationsForMethod(data['method'])

    @staticmethod
    @transaction.atomic
    def future_compare(address):
        methods = {
            "ML": MLUtils.MLAnalysisWithoutPreTrain,
            # "ARIMA": ARIMAUtils.arimaAnalysisWithoutPreTrained,
            "DL": DLUtils.DLAnalysisWithoutPreTrain,
            "Transformer": NotPreTrainTransformerr.main,
            "PatchTST": PatchTSTUtils.patchtstAnalysis
        }

        mape_results = {}

        for method_name, func in methods.items():
            try:
                result_obj = func(address)
                mape = result_obj['metrics']['mape']
                mape_results[method_name] = mape*100
                print("成功")
            except Exception as e:
                print(str(e))
                # 可选：记录错误或设置为 None / NaN
                mape_results[method_name] = None  # 或 float('nan')
                # 如果需要调试，可以打印错误信息：
                # print(f"Error in {method_name}: {e}")

        return mape_results

