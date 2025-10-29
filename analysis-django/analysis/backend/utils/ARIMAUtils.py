import matplotlib
matplotlib.use('Agg')  # 非交互式环境保存图片

import io
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import os
from .OssUtils import OssUtils

csv_path = "E:\github01\softwareProject/analysis-django/analysis/backend/utils/milano_traffic_nid.csv"  # 原始数据路径
model_dir = "E:\github01\softwareModel\ARIMA_MODEL"  # 模型保存目录
TRAIN_RATIO = 0.8  # 训练/验证拆分比例
forecast_steps = 5  # 多步预测步数

def arimaAnalysis(address:str) -> str:
    # ---------------------- 数据预处理 ----------------------
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df = df.asfreq('10min')  # 确保时间频率为10分钟
    ts = df[address].dropna()  # 过滤缺失值


    # ---------------------- 拆分训练/验证集（与训练时一致） ----------------------
    train_size = int(len(ts) * TRAIN_RATIO)
    train_ts = ts[:train_size]
    val_ts = ts[train_size:]


    # ---------------------- 加载预训练模型 ----------------------
    model_path = f"{model_dir}/{address}_ARIMA_(1, 1, 1).pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件 {model_path} 不存在，请先运行训练代码。")
    arima_model = joblib.load(model_path)
    print(f"成功加载预训练模型：{model_path}")


    # ---------------------- 滚动预测：复用预训练模型参数 ----------------------
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


    # ---------------------- 结果可视化（聚焦预测区域） ----------------------
    # 转换预测结果为Series（匹配验证集索引）
    val_pred_series = pd.Series(val_predictions, index=val_ts.index[:len(val_predictions)])

    # 聚焦训练集末尾（避免前半段过长序列干扰，取最后500个点，若不足则取全部）
    focus_train_len = min(500, len(train_ts))
    focus_train_ts = train_ts[-focus_train_len:]

    # 合并“聚焦训练末尾 + 验证真实值”（用于绘图范围）
    combined_ts = pd.concat([focus_train_ts, val_ts])

    # 提取训练集拟合值的“聚焦部分”（仅保留与focus_train_ts重叠的时段）
    fitted_train = arima_model.fittedvalues
    focus_fitted_train = fitted_train[focus_train_ts.index[0]:focus_train_ts.index[-1]]


    plt.figure(figsize=(14, 7))

    # 1. 绘制「聚焦训练段」的真实值与拟合值
    plt.plot(focus_train_ts.index, focus_train_ts,
         label='Actual (Training, Focus)', color='blue', alpha=0.7)
    plt.plot(focus_fitted_train.index, focus_fitted_train,
         label='Fitted (Training, Focus)', color='black', linestyle='-', alpha=0.7)

    # 2. 绘制「验证段」的真实值与预测值
    plt.plot(val_ts.index, val_ts,
         label='Actual (Validation)', color='green', alpha=0.9)
    plt.plot(val_pred_series.index, val_pred_series,
         label=f'Rolling Forecast (Validation, {forecast_steps}-step)', color='red', linestyle='--')

    # 标注训练/验证拆分线
    plt.axvline(x=train_ts.index[-1], color='gray', linestyle=':', label='Train/Validation Split')

    # 图表格式优化
    plt.title(f'{address} - ARIMA(1,1,1) Rolling Forecast (Focus on Prediction)', fontsize=12)
    plt.xlabel('Time', fontsize=10)
    plt.ylabel('Value', fontsize=10)
    plt.legend(loc='upper right')
    plt.xticks(rotation=45)
    plt.tight_layout()  # 避免标签截断

    buffer = io.BytesIO()  # 初始化内存缓冲区
    plt.savefig(buffer, dpi=300, bbox_inches='tight', format='png')  # 保存图表到缓冲区（指定格式为png）
    buffer.seek(0)  # 将缓冲区指针移到开头，否则读取不到数据

    save_path = f"arima/{address}.png"
    ossUtils = OssUtils()
    image_url = ossUtils.OssUpload(buffer,save_path)

    plt.close()
    return image_url
