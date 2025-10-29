import matplotlib
matplotlib.use('Agg')  # 非交互式环境保存图片
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error
import io
from .OssUtils import OssUtils


# --------------------------
# 1. 复用特征工程函数（必须与训练时一致）
# --------------------------
def create_enhanced_features(series_data, lag_steps=12, rolling_window_size=6):
    """生成与训练时完全一致的特征（避免数据泄露）"""
    df_features = pd.DataFrame(index=series_data.index)
    df_features['target'] = series_data

    # 滞后特征（短期+长期）
    for i in range(1, lag_steps + 1):
        df_features[f'lag_{i}'] = series_data.shift(i)
    df_features['lag_day'] = series_data.shift(144)  # 1天前（假设10分钟/步，144步=24小时）
    df_features['lag_week'] = series_data.shift(1008)  # 1周前（144*7）

    # 滚动窗口特征
    df_features['rolling_mean'] = series_data.shift(1).rolling(window=rolling_window_size).mean()
    df_features['rolling_median'] = series_data.shift(1).rolling(window=rolling_window_size).median()
    df_features['rolling_std'] = series_data.shift(1).rolling(window=rolling_window_size).std()

    # 时间特征（周期编码）
    df_features['hour'] = df_features.index.hour
    df_features['day_of_week'] = df_features.index.dayofweek
    df_features['minute_of_day'] = df_features.index.hour * 60 + df_features.index.minute
    df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24)
    df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24)
    df_features['day_of_week_sin'] = np.sin(2 * np.pi * df_features['day_of_week'] / 7)
    df_features['day_of_week_cos'] = np.cos(2 * np.pi * df_features['day_of_week'] / 7)

    # 趋势特征
    df_features['trend'] = np.arange(len(df_features))

    df_features.dropna(inplace=True)  # 移除因移位产生的缺失值
    return df_features


# --------------------------
# 2. 核心函数：用已有模型进行历史预测并对比
# --------------------------
def predict_and_compare(
    csv_path,          # CSV数据路径
    model_path,        # joblib模型路径
    target_column,     # 要预测的列名
    feature_cols_path, # 训练时的特征列列表（需提前保存，见下方说明）
    backtest_start_pct=0.5,  # 回测起始点（数据的50%位置开始）
    forecast_steps=24        # 预测步数（如24步=4小时，10分钟/步）
):
    """
    用已有模型对历史数据进行预测，与真实值对比并绘图
    """
    # 1. 加载CSV数据并预处理
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])  # 转换为时间格式
    df.set_index('timestamp', inplace=True)            # 时间作为索引
    target_series = df[target_column].dropna()         # 提取目标列并去重
    print(f"加载数据完成，{target_column}列共{len(target_series)}条有效数据")

    # 2. 加载模型和训练时的特征列（关键：确保特征匹配）
    model = joblib.load(model_path)
    feature_columns = joblib.load(feature_cols_path)  # 需提前保存训练时的特征列
    print(f"加载模型完成，特征列数量：{len(feature_columns)}")

    # 3. 确定回测区间（用历史数据的一部分做预测）
    # 计算回测起始索引（从数据中间开始，确保有足够历史数据生成特征）
    backtest_start_idx = int(len(target_series) * backtest_start_pct)
    # 生成特征需要的历史数据长度（与训练时一致）
    required_history_len = max(12, 6, 144, 1008)  # lag_steps=12, rolling_window=6, 144/1008为日/周滞后
    input_end_idx = backtest_start_idx + required_history_len  # 输入历史的结束位置
    if input_end_idx >= len(target_series):
        raise ValueError("回测起始点过晚，没有足够数据生成特征")

    # 4. 滚动预测（逐步预测，避免用未来数据）
    input_hist = target_series.iloc[:input_end_idx]  # 用于预测的历史数据
    temp_hist = input_hist.copy()                    # 临时历史（逐步加入预测结果）
    pred_values = []                                 # 存储预测结果

    for i in range(forecast_steps):
        # 用当前临时历史生成特征
        features = create_enhanced_features(temp_hist)
        if len(features) == 0:
            break  # 特征不足时停止

        # 取最新一行特征，确保与模型特征列一致
        latest_feature = features.iloc[-1:].drop('target', axis=1)
        latest_feature = latest_feature.reindex(columns=feature_columns, fill_value=0)

        # 预测当前步
        pred = model.predict(latest_feature)[0]
        pred_values.append(pred)

        # 更新临时历史（加入预测值，模拟“未来”用于下一步预测）
        next_timestamp = target_series.index[input_end_idx + i]  # 真实时间戳
        temp_hist = pd.concat([temp_hist, pd.Series([pred], index=[next_timestamp])])

    # 5. 提取真实历史值（用于对比）
    true_start = input_end_idx
    true_end = input_end_idx + forecast_steps
    true_values = target_series.iloc[true_start:true_end]

    # 6. 对齐预测值与真实值（确保长度一致）
    pred_series = pd.Series(
        pred_values[:len(true_values)],
        index=true_values.index  # 用真实时间戳作为索引
    )

    # 7. 计算评估指标
    rmse = np.sqrt(mean_squared_error(true_values, pred_series))
    mae = mean_absolute_error(true_values, pred_series)
    print(f"\n预测评估指标：RMSE={rmse:.2f}, MAE={mae:.2f}")

    # 8. 绘图（对比历史输入、预测值、真实值）
    plt.figure(figsize=(15, 7))
    # 输入历史数据（用于预测的部分）
    plt.plot(input_hist.index[-50:], input_hist.values[-50:],
             label='Input History (Last 50 Steps)', color='blue', alpha=0.6)
    # 预测值
    plt.plot(pred_series.index, pred_series.values,
             label='Predicted Values', color='orange', linestyle='--', linewidth=2)
    # 真实值
    plt.plot(true_values.index, true_values.values,
             label='True Values', color='green', linewidth=2)
    # 标记预测起始点
    plt.axvline(x=input_hist.index[-1], color='red', linestyle=':', label='Prediction Start')

    plt.xlabel('Timestamp')
    plt.ylabel('Value')
    plt.title(f'Historical Prediction vs True Values: {target_column}')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    buffer = io.BytesIO()  # 初始化内存缓冲区
    plt.savefig(buffer, dpi=300, bbox_inches='tight', format='png')  # 保存图表到缓冲区（指定格式为png）
    buffer.seek(0)  # 将缓冲区指针移到开头，否则读取不到数据

    save_path = f"ML/{target_column}.png"
    ossUtils = OssUtils()
    image_url = ossUtils.OssUpload(buffer, save_path)

    plt.close()
    return image_url

def MLAnalysis(address:str) -> str:
    # --------------------------
    # 用户需要修改的参数
    # --------------------------
    CSV_PATH = "E:\github01\softwareProject/analysis-django/analysis/backend/utils/milano_traffic_nid.csv"  # 你的CSV数据路径
    MODEL_PATH = "E:\github01\softwareModel\ML_MODEL/traffic_historical_model.joblib"  # 已保存的模型路径
    # 关键：需提前保存训练时的特征列（见下方说明）
    FEATURE_COLS_PATH = "E:\github01\softwareModel\ML_MODEL/feature_columns.joblib"

    # 执行预测与对比
    image_url = predict_and_compare(
        csv_path=CSV_PATH,
        model_path=MODEL_PATH,
        target_column=address,
        feature_cols_path=FEATURE_COLS_PATH,
        backtest_start_pct=0.5,  # 从数据中间开始预测
        forecast_steps=48        # 预测48步（8小时，10分钟/步）
    )
    return image_url