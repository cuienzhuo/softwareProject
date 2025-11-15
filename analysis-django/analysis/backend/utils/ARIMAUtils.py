import matplotlib
matplotlib.use('Agg')  # 非交互式环境保存图片

import pandas as pd
import joblib
from statsmodels.tsa.arima.model import ARIMA
import os
import numpy as np
from pmdarima.arima import ARIMA as TrainARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error


csv_path = "E:\\github01\\softwareProject/analysis-django/analysis/backend/utils/milano_traffic_nid.csv"  # 原始数据路径
model_dir = "E:\\github01\\softwareModel\\ARIMA_MODEL"  # 模型保存目录
TRAIN_RATIO = 0.8  # 训练/验证拆分比例

def arimaAnalysis(address, forecast_steps):
    print("数据处理")
    # 数据预处理
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df = df.asfreq('10min')  # 确保时间频率为10分钟
    ts = df[address].dropna()  # 过滤缺失值

    # 拆分训练/验证集（与训练时一致)
    train_size = int(len(ts) * TRAIN_RATIO)
    train_ts = ts[:train_size]
    val_ts = ts[train_size:]

    # 加载预训练模型
    model_path = f"{model_dir}/{address}_ARIMA_(1, 1, 1).pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件 {model_path} 不存在，请先运行训练代码。")
    arima_model = joblib.load(model_path)
    print(f"成功加载预训练模型：{model_path}")

    # 滚动预测：复用预训练模型参数
    current_history = train_ts.copy()  # 初始历史为训练集
    val_predictions = []

    for i in range(0, len(val_ts), forecast_steps):
        # 确保预测步数不越界
        valid_steps = min(forecast_steps, len(val_ts) - i)
        if valid_steps == 0:
            break

        # 用预训练模型的配置和参数，初始化新的ARIMA模型（基于当前历史）
        temp_model = ARIMA(
            current_history,
            order=arima_model.model.order,        # 复用训练好的(p,d,q)
            trend=arima_model.model.trend,        # 复用趋势设置
            enforce_stationarity=arima_model.model.enforce_stationarity,
            enforce_invertibility=arima_model.model.enforce_invertibility,
        )
        temp_fit = temp_model.fit(
            start_params=arima_model.params  # 直接使用预训练的参数
        )

        # 多步预测
        multi_pred = temp_fit.forecast(steps=valid_steps)
        for j in range(valid_steps):
            val_predictions.append(multi_pred.iloc[j])
            # 用真实值更新历史（模拟“滚动预测+真实数据反馈”）
            current_history = pd.concat([current_history, val_ts.iloc[[i + j]]])

    # 转换预测结果为Series（匹配验证集索引）
    val_pred_series = pd.Series(val_predictions, index=val_ts.index[:len(val_predictions)])

    # 对齐长度（确保真实值和预测值长度一致）
    aligned_val_ts = val_ts.loc[val_pred_series.index]

    # ========== 新增：计算评估指标 ==========
    def mean_absolute_error(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    def root_mean_squared_error(y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

    def mean_absolute_percentage_error(y_true, y_pred):
        # 避免除零：只在 y_true != 0 的位置计算
        mask = y_true != 0
        if not np.any(mask):
            return np.nan
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))  # 返回百分比

    def series_to_list(series):
        return [{"time": t.isoformat(), "value": float(v)} for t, v in series.items()]

    mae = mean_absolute_error(aligned_val_ts.values, val_pred_series.values)
    rmse = root_mean_squared_error(aligned_val_ts.values, val_pred_series.values)
    mape = mean_absolute_percentage_error(aligned_val_ts.values, val_pred_series.values)

    # ========== 可视化 ==========
    # 聚焦训练集末尾（避免前半段过长序列干扰，取最后500个点，若不足则取全部）
    focus_train_len = min(500, len(train_ts))
    focus_train_ts = train_ts[-focus_train_len:]

    # 合并“聚焦训练末尾 + 验证真实值”（用于绘图范围）
    combined_ts = pd.concat([focus_train_ts, val_ts])

    # 提取训练集拟合值的“聚焦部分”
    fitted_train = arima_model.fittedvalues
    focus_fitted_train = fitted_train[focus_train_ts.index[0]:focus_train_ts.index[-1]]

    chart_data = {
        "focus_train_actual": series_to_list(focus_train_ts),
        "focus_train_fitted": series_to_list(focus_fitted_train),
        "val_actual": series_to_list(val_ts),
        "val_predicted": series_to_list(val_pred_series),
        "split_time": train_ts.index[-1].isoformat()  # 分割线时间
    }

    return {
        "metrics": {
            "mae": float(mae),
            "rmse": float(rmse),
            "mape": float(mape) if not np.isnan(mape) else None
        },
        "chartData": chart_data
    }


def arimaAnalysisWithoutPreTrained(address, train_ratio=0.8, p=1, d=1, q=1, forecast_steps=5):
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    if address not in df.columns:
        raise ValueError(f"列 '{address}' 不存在于数据中。可用列: {list(df.columns)}")

    series = df[address].dropna()
    if len(series) == 0:
        raise ValueError(f"列 '{address}' 中无有效数据。")

    n = len(series)
    train_size = int(n * train_ratio)
    train, test = series[:train_size], series[train_size:]

    print(train_size)

    order = (p, d, q)
    base_model = TrainARIMA(order=order, seasonal_order=(0, 0, 0, 0),suppress_warnings=True)

    history = list(train)
    predictions = []

    for i in range(len(test)):
        try:
            model_fit = base_model.fit(history)
            forecast = model_fit.predict(n_periods=forecast_steps)
            yhat = forecast[0]
        except Exception as e:
            print(f"ARIMA 拟合失败（第 {i} 步）: {e}")
            yhat = history[-1] if history else np.mean(train)  # 回退策略

        predictions.append(yhat)
        history.append(test.iloc[i])  # 滚动加入真实值

    # 转换为 Series 以对齐索引
    pred_series = pd.Series(predictions, index=test.index)

    # ========== 计算评估指标 ==========
    y_true = test.values
    y_pred = pred_series.values

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    # 安全计算 MAPE（避免除零）
    epsilon = 1e-8
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))))

    # ========== 准备绘图数据（供前端使用）==========
    # 主图：完整测试集
    main_plot_data = {
        "test": {
            "timestamps": test.index.strftime("%Y-%m-%d %H:%M:%S").tolist(),
            "values": test.values.tolist()
        },
        "predictions": {
            "timestamps": pred_series.index.strftime("%Y-%m-%d %H:%M:%S").tolist(),
            "values": pred_series.values.tolist()
        }
    }

    # 放大图：前2小时（或前N点）
    if len(test) < 2:
        zoom_plot_data = None
    else:
        time_diff = (test.index[1] - test.index[0]).total_seconds()
        if time_diff <= 0:
            n_points = min(24, len(test))  # 默认按5分钟粒度估算2小时（24点）
        else:
            n_points = min(int(48 * 3600 / time_diff) + 1, len(test))

        zoom_test = test.iloc[:n_points]
        zoom_pred = pred_series.iloc[:n_points]

        zoom_plot_data = {
            "test": {
                "timestamps": zoom_test.index.strftime("%Y-%m-%d %H:%M:%S").tolist(),
                "values": zoom_test.values.tolist()
            },
            "predictions": {
                "timestamps": zoom_pred.index.strftime("%Y-%m-%d %H:%M:%S").tolist(),
                "values": zoom_pred.values.tolist()
            }
        }

    # ========== 返回结构化结果 ==========
    result = {
        "metrics": {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "mape": round(mape, 4)
        },
        "plots": {
            "main": main_plot_data,
            "zoom": zoom_plot_data  # 可能为 None
        }
    }

    return result