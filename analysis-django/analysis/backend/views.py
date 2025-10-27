from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
import json
from .services import AnalysisService
from django.views.decorators.csrf import csrf_exempt

@require_http_methods(["POST"])  # 仅允许POST请求
@csrf_exempt  # 禁用 CSRF 保护
def anomaly_analysis_view(request):
    """异常数据分析接口"""
    try:
        data = json.loads(request.body)  # 解析请求体JSON
        # 验证参数
        if not all(key in data for key in ['address', 'method']):
            return JsonResponse({
                'code': 400,
                'error': '缺少参数：address或method'
            }, status=400)
        
        # 调用业务逻辑
        image_url = AnalysisService.anomaly_analysis(
            address=data['address'],
            method=data['method']
        )
        return JsonResponse({
            'code': 200,
            'image_url': image_url,
            'message': '异常数据分析完成'
        })
    except Exception as e:
        return JsonResponse({
            'code': 500,
            'error': str(e)
        }, status=500)

@require_http_methods(["POST"])
@csrf_exempt
def cluster_analysis_view(request):
    """聚类分析接口"""
    try:
        data = json.loads(request.body)
        if 'address' not in data:
            return JsonResponse({
                'code': 400,
                'error': '缺少参数：address'
            }, status=400)
        
        image_url = AnalysisService.cluster_analysis(address=data['address'])
        return JsonResponse({
            'code': 200,
            'image_url': image_url,
            'message': '聚类分析完成'
        })
    except Exception as e:
        return JsonResponse({
            'code': 500,
            'error': str(e)
        }, status=500)

@require_http_methods(["POST"])
@csrf_exempt
def future_prediction_view(request):
    """未来预测接口"""
    try:
        data = json.loads(request.body)
        if not all(key in data for key in ['address', 'method']):
            return JsonResponse({
                'code': 400,
                'error': '缺少参数：address或method'
            }, status=400)
        
        image_url = AnalysisService.future_prediction(
            address=data['address'],
            method=data['method']
        )
        return JsonResponse({
            'code': 200,
            'image_url': image_url,
            'message': '未来预测完成'
        })
    except Exception as e:
        return JsonResponse({
            'code': 500,
            'error': str(e)
        }, status=500)