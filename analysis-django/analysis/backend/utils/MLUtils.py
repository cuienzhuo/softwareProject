import matplotlib
matplotlib.use('Agg')  # 非交互式环境保存图片
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

CSV_PATH = "E:\github01\softwareProject/analysis-django/analysis/backend/utils/milano_traffic_nid.csv"

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))
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
    backtest_start_pct,  # 回测起始点（数据的50%位置开始）
    forecast_steps       # 预测步数（如24步=4小时，10分钟/步）
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
    rmse = np.sqrt(mean_squared_error(true_values, pred_series))
    mae = mean_absolute_error(true_values, pred_series)
    mape = mean_absolute_percentage_error(true_values.values, pred_series.values)

    print(f"\n预测评估指标：RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape:.2f}%")

    # 准备要传给前端的数据
    result_data = {
        "chartData":{
            'timestamps': true_values.index.tolist(),
            'true_values': true_values.values.tolist(),
            'pred_values': pred_series.values.tolist()
        },
        'metrics': {
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        }
    }

    return result_data

def MLAnalysis(address,backtest_start_pct,forecast_steps):
    MODEL_PATH = "E:\github01\softwareModel\ML_MODEL/traffic_historical_model.joblib"  # 已保存的模型路径
    # 关键：需提前保存训练时的特征列（见下方说明）
    FEATURE_COLS_PATH = "E:\github01\softwareModel\ML_MODEL/feature_columns.joblib"

    # 执行预测与对比
    return predict_and_compare(
        csv_path=CSV_PATH,
        model_path=MODEL_PATH,
        target_column=address,
        feature_cols_path=FEATURE_COLS_PATH,
        backtest_start_pct=backtest_start_pct,  # 从数据中间开始预测
        forecast_steps=forecast_steps        # 预测48步（8小时，10分钟/步）
    )

def MLAnalysisWithoutPreTrain(address, train_ratio=0.8, n_lags=6):
    df = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)
    series = df[address].dropna()
    if len(series) == 0:
        raise ValueError(f"列 '{address}' 中无有效数据。")

    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("DataFrame 索引必须是 DatetimeIndex。")

    # 划分训练集和测试集
    n = len(series)
    train_size = int(n * train_ratio)
    train, test = series[:train_size], series[train_size:]

    # 构建 lag 特征
    def create_lagged_dataset(ts, n_lags):
        data = pd.DataFrame(ts)
        for i in range(1, n_lags + 1):
            data[f'lag_{i}'] = ts.shift(i)
        data.dropna(inplace=True)
        X = data[[f'lag_{i}' for i in range(1, n_lags + 1)]].values
        y = data[ts.name].values
        return X, y

    X_train, y_train = create_lagged_dataset(train, n_lags)

    if len(X_train) == 0:
        raise ValueError(f"训练数据不足，无法构建 {n_lags} 阶滞后特征。")

    # 训练模型
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # 单步滚动预测（使用真实历史值）
    predictions = []
    history = list(train[-n_lags:])

    for i in range(len(test)):
        X_input = np.array(history[-n_lags:]).reshape(1, -1)
        y_pred = model.predict(X_input)[0]
        predictions.append(y_pred)
        history.append(test.iloc[i])  # 使用真实值

    pred_series = pd.Series(predictions, index=test.index)

    # 评估指标
    mae = mean_absolute_error(test, pred_series)
    rmse = np.sqrt(mean_squared_error(test, pred_series))
    mape = np.mean(np.abs((test - pred_series) / np.where(test == 0, 1e-8, test)))

    # 转换为 JSON 安全格式
    timestamps = test.index.strftime('%Y-%m-%d %H:%M:%S').tolist()
    test_values = test.tolist()
    pred_values = pred_series.tolist()

    # ========== 准备放大图数据（前2小时）==========
    zoom_test = []
    zoom_pred = []
    zoom_timestamps = []

    if len(test) >= 2:
        time_diff = (test.index[1] - test.index[0]).total_seconds()
        if time_diff <= 0 or np.isnan(time_diff):
            n_points = min(12, len(test))  # 默认 12 点（假设10分钟间隔）
        else:
            n_points = min(int((48 * 3600) / time_diff) + 1, len(test))  # 2小时 = 7200秒

        zoom_test = test.iloc[:n_points].tolist()
        zoom_pred = pred_series.iloc[:n_points].tolist()
        zoom_timestamps = test.index[:n_points].strftime('%Y-%m-%d %H:%M:%S').tolist()

    # 返回结构化数据
    result = {
        "plots":{
            "main": {
                "timestamps": timestamps,
                "test_values": test_values,
                "predictions": pred_values
            },
            "zoom": {
                "timestamps": zoom_timestamps,
                "test_values": zoom_test,
                "predictions": zoom_pred
            }
        },
        "metrics": {
            "mae": float(mae),
            "rmse": float(rmse),
            "mape": float(mape)
        }
    }

    return result