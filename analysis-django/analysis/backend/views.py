from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
import json
from .services import AnalysisService
from django.views.decorators.csrf import csrf_exempt
import os
import pandas as pd

CSV_FILE_PATH = 'E:\github01\softwareProject/analysis-django/analysis/backend/utils/milano_traffic_nid.csv'

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
        location,method,method_cn_name,results,total_count,anomaly_count,anomaly_ratio = AnalysisService.anomaly_analysis(data)
        if 'timestamp' in results.columns:
            results = results.copy()  # 避免修改原始数据
            results['timestamp'] = results['timestamp'].astype(str)

        # 转为 JSON-兼容的列表
        results_list = results.to_dict(orient='records')

        data = {
            'location': location,
            'method': method,
            'methodCnName': method_cn_name,
            'results': results_list
        }

        analysis_overview = {
            'total_count': int(total_count),
            'anomaly_count': int(anomaly_count),
            'anomaly_ratio': f"{anomaly_ratio * 100:.2f}%"
        }

        return JsonResponse({
            'code': 200,
            'data':data,
            'analysis_overview':analysis_overview,
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
        if 'clusters' not in data:
            return JsonResponse({
                'code': 400,
                'error': '缺少参数：clusters'
            }, status=400)
        
        cluster_data = AnalysisService.cluster_analysis(data['clusters'])
        cluster_list_for_frontend = []
        for nil_name, cluster_id in cluster_data.items():
            cluster_list_for_frontend.append({
                'NIL': nil_name,
                'cluster': str(cluster_id)  # 建议将 cluster ID 转换为字符串，以确保前端颜色映射键的类型一致性
            })

        return JsonResponse({
            'code': 200,
            'clusterData':cluster_list_for_frontend,
            'message': '聚类分析完成'
        })
    except Exception as e:
        print(str(e))
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
        print(data)
        if not all(key in data for key in ['address', 'method']):
            return JsonResponse({
                'code': 400,
                'error': '缺少参数：address或method'
            }, status=400)
        
        result = AnalysisService.future_prediction(data)
        return JsonResponse({
            'code': 200,
            'result': result,
            'message': '未来预测完成'
        })
    except Exception as e:
        print(str(e))
        return JsonResponse({
            'code': 500,
            'error': str(e)
        }, status=500)

@require_http_methods(["GET"])
def get_milan_columns_view(request):
    """
    获取 milan.csv 中的所有列名（排除 'timestamp' 列）
    返回格式: { "columns": ["col1", "col2", ...] }
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(CSV_FILE_PATH):
            return JsonResponse({
                'code': 404,
                'error': f'CSV 文件未找到: {CSV_FILE_PATH}'
            }, status=404)

        # 只读取表头（第一行），避免加载整个大文件
        df = pd.read_csv(CSV_FILE_PATH, nrows=0)  # nrows=0 只读 header
        columns = df.columns.tolist()

        # 移除 'timestamp' 并写成label,value的形式传给前端
        filtered_columns = [
            {"label": col, "value": col}
            for col in columns
            if col.lower() != 'timestamp'
        ]

        return JsonResponse({
            'code': 200,
            'columns': filtered_columns
        })

    except Exception as e:
        return JsonResponse({
            'code': 500,
            'error': f'读取列名失败: {str(e)}'
        }, status=500)

@require_http_methods(["POST"])
@csrf_exempt
def anomaly_compare_view(request):
    try:
        data = json.loads(request.body)
        if 'address' not in data and 'method' not in data:
            return JsonResponse({
                'code': 400,
                'error': '必须要传入地址或方法其中之一'
            })
        if 'address' in data:
            print("同一地区不同方法")
            method_names,anomaly_counts = AnalysisService.anomaly_compare(data)
            data = {
                "method_names": method_names,
                "anomaly_counts": anomaly_counts
            }
        else:
            print("同一方法不同地区")
            locations,anomaly_counts = AnalysisService.anomaly_compare(data)
            data = {
                "locations": locations,
                "anomaly_counts": anomaly_counts
            }
        print("返回")
        return JsonResponse({
            'code':200,
            'data':data,
            'message':"返回成功"
        })
    except Exception as e:
        print(str(e))
        return JsonResponse({
            'code': 500,
            'error': str(e)
        }, status=500)

@require_http_methods(["POST"])
@csrf_exempt
def future_compare_view(request):
    try:
        data = json.loads(request.body)
        if 'address' not in data:
            return JsonResponse({
                'code': 400,
                'error': '必须要传入地址或方法其中之一'
            })
        print("预测比较")
        result = AnalysisService.future_compare(data['address'])
        return JsonResponse({
            'code':200,
            'result':result,
            'message':"返回成功"
        })
    except Exception as e:
        print(str(e))
        return JsonResponse({
            'code': 500,
            'error': str(e)
        }, status=500)