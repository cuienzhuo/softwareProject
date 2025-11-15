import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# --------------------------
# 1. 配置参数与路径
# --------------------------
MODEL_PATH = r"E:\github01\softwareModel\DL_MODEL\traffic_prediction_model.keras"
SCALER_PATH = r"E:\github01\softwareModel\DL_MODEL\traffic_scaler.pkl"
CSV_PATH = r"E:\github01\softwareProject\analysis-django\analysis\backend\utils\milano_traffic_nid.csv"
LOOK_BACK = 60
INITIAL_WINDOW_SIZE = 3000


# --------------------------
# 2. 加载依赖
# --------------------------
def load_dependencies():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"模型文件 {MODEL_PATH} 不存在")
    model = load_model(MODEL_PATH)
    print(f"成功加载模型：{MODEL_PATH}")

    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"归一化器 {SCALER_PATH} 不存在")
    scaler = joblib.load(SCALER_PATH)
    print(f"成功加载归一化器")

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"数据文件 {CSV_PATH} 不存在")
    df = pd.read_csv(CSV_PATH, index_col="timestamp", parse_dates=True)
    print(f"成功加载数据：共 {len(df)} 条记录，{df.shape[1]} 个特征列")

    return model, scaler, df


# --------------------------
# 3. 准备滚动数据
# --------------------------
def prepare_rolling_data(df, roll_steps):
    total_needed = INITIAL_WINDOW_SIZE + roll_steps
    if total_needed > len(df):
        total_needed = len(df)
        adjusted_roll_steps = total_needed - INITIAL_WINDOW_SIZE
        print(f"警告：原始数据长度不足，滚动步数自动调整为 {adjusted_roll_steps}（原始数据共 {len(df)} 条）")
        return df.iloc[:total_needed], adjusted_roll_steps
    return df.iloc[:total_needed], roll_steps


# --------------------------
# 4. 计算 MAPE（注意避免除零）
# --------------------------
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # 避免除以0：当真实值为0时，跳过或设为0误差（根据业务逻辑）
    mask = y_true != 0
    if not np.any(mask):
        return np.nan  # 或 0.0，视情况而定
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))
    return mape


# --------------------------
# 5. 执行滚动预测并收集所有必要数据
# --------------------------
def run_rolling_test(model, scaler, df_test, roll_steps, num_features):
    predictions = []
    actuals = []
    step_rmse_list = []
    step_mae_list = []
    step_mape_list = []

    rolling_data = df_test.iloc[:INITIAL_WINDOW_SIZE].values
    rolling_data_scaled = scaler.transform(rolling_data)

    for i in range(roll_steps):
        input_seq = rolling_data_scaled[-LOOK_BACK:].reshape(1, LOOK_BACK, num_features)
        pred_scaled = model.predict(input_seq, verbose=0)[0]
        pred_actual = scaler.inverse_transform(pred_scaled.reshape(1, -1))[0]

        actual_idx = INITIAL_WINDOW_SIZE + i
        actual_actual = df_test.iloc[actual_idx].values

        # 更新窗口（使用真实值）
        actual_scaled = scaler.transform(actual_actual.reshape(1, -1))[0]
        rolling_data_scaled = np.append(
            rolling_data_scaled[1:],
            actual_scaled.reshape(1, -1),
            axis=0
        )

        predictions.append(pred_actual)
        actuals.append(actual_actual)

        # 单步指标
        step_rmse = np.sqrt(mean_squared_error(actual_actual, pred_actual))
        step_mae = mean_absolute_error(actual_actual, pred_actual)
        step_mape = mean_absolute_percentage_error(actual_actual, pred_actual)

        step_rmse_list.append(float(step_rmse))
        step_mae_list.append(float(step_mae))
        step_mape_list.append(float(step_mape) if not np.isnan(step_mape) else None)

        if (i + 1) % 10 == 0:
            print(f"滚动进度：{i + 1}/{roll_steps} | RMSE: {step_rmse:.2f}, MAE: {step_mae:.2f}")

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # 整体指标（整个序列）
    overall_rmse = float(np.sqrt(mean_squared_error(actuals, predictions)))
    overall_mae = float(mean_absolute_error(actuals, predictions))
    overall_mape = float(mean_absolute_percentage_error(actuals, predictions))

    return {
        "predictions": predictions.tolist(),
        "actuals": actuals.tolist(),
        "timestamps": df_test.index[INITIAL_WINDOW_SIZE: INITIAL_WINDOW_SIZE + roll_steps].strftime(
            "%Y-%m-%d %H:%M:%S").tolist(),
    }


# --------------------------
# 6. 主函数：返回结构化数据给前端
# --------------------------
def DLAnalysis(address, roll_steps):
    """
    返回结构化预测结果，供前端绘图和展示指标
    :param address: 特征列名（如 'nid_123'）
    :param roll_steps: 滚动步数
    :return: dict 包含预测值、真实值、时间戳、指标
    """
    model, scaler, df = load_dependencies()

    if address not in df.columns:
        raise ValueError(f"指定的特征列 '{address}' 不存在于数据中。可用列：{list(df.columns)}")

    num_features = df.shape[1]
    df_test, roll_steps = prepare_rolling_data(df, roll_steps)

    full_result = run_rolling_test(model, scaler, df_test, roll_steps, num_features)

    # 提取目标特征索引
    feature_idx = df.columns.get_loc(address)

    # 提取单特征序列
    predictions_single = [p[feature_idx] for p in full_result["predictions"]]
    actuals_single = [a[feature_idx] for a in full_result["actuals"]]

    # 重新计算该特征的三大指标（更准确！）
    overall_rmse = float(np.sqrt(mean_squared_error(actuals_single, predictions_single)))
    overall_mae = float(mean_absolute_error(actuals_single, predictions_single))
    overall_mape = mean_absolute_percentage_error(actuals_single, predictions_single)
    overall_mape = float(overall_mape) if not np.isnan(overall_mape) else None

    # 构造前端所需结构
    return {
        "chartData": {
            "timestamps": full_result["timestamps"],  # 时间戳已在 run_rolling_test 中生成
            "predictions": predictions_single,
            "actuals": actuals_single
        },
        "metrics": {
            "rmse": overall_rmse,
            "mae": overall_mae,
            "mape": overall_mape
        }
    }


def DLAnalysisWithoutPreTrain(address, train_ratio=0.8, look_back=60, epochs=50, batch_size=64, lstm_units=50):
    np.random.seed(42)
    tf.random.set_seed(42)

    try:
        df = pd.read_csv(CSV_PATH, index_col='timestamp', parse_dates=True)
    except FileNotFoundError:
        return {"error": "CSV file 'milano_traffic_nid.csv' not found."}

    df = df.dropna()
    print(f"数据形状（处理缺失值后）: {df.shape}")
    print(f"可用列: {df.columns.tolist()}")

    # 获取目标列索引
    if address not in df.columns:
        return {"error": f"指定的地址列 '{address}' 不存在于数据中。可用列: {df.columns.tolist()}"}
    target_idx = df.columns.get_loc(address)
    num_features = df.shape[1]

    # 划分训练/测试集
    train_size = int(len(df) * train_ratio)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]

    # 归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(train_df.values)
    scaled_test = scaler.transform(test_df.values)

    # 数据集生成函数：多变量输入 → 单变量输出
    def create_dataset_multivar_to_single(data, look_back, target_col_index):
        X, Y = [], []
        for i in range(len(data) - look_back):
            X.append(data[i:(i + look_back), :])
            Y.append(data[i + look_back, target_col_index])
        return np.array(X), np.array(Y)

    X_train, y_train = create_dataset_multivar_to_single(scaled_train, look_back, target_idx)
    X_test, y_test = create_dataset_multivar_to_single(scaled_test, look_back, target_idx)

    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    print(f"\n训练集: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"测试集: X_test={X_test.shape}, y_test={y_test.shape}")

    # 构建LSTM模型（使用传入的 lstm_units）
    model = Sequential(name="Single_Target_LSTM")
    model.add(Input(shape=(look_back, num_features)))
    model.add(LSTM(lstm_units, return_sequences=True))
    model.add(LSTM(lstm_units))
    model.add(Dense(1))  # 单输出

    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    model.summary()

    # 回调函数
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath='best_single_model.keras', monitor='val_loss', save_best_only=True, verbose=1)
    ]

    # 训练
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=2
    )

    # 预测
    train_predict = model.predict(X_train, verbose=0)
    test_predict = model.predict(X_test, verbose=0)

    # 反归一化辅助函数
    def inverse_transform_single(scaler, data, target_col_index):
        dummy = np.zeros((data.shape[0], num_features))
        dummy[:, target_col_index] = data.flatten()
        inversed = scaler.inverse_transform(dummy)
        return inversed[:, target_col_index]

    # 反归一化
    train_predict_actual = inverse_transform_single(scaler, train_predict, target_idx)
    y_train_actual = inverse_transform_single(scaler, y_train, target_idx)
    test_predict_actual = inverse_transform_single(scaler, test_predict, target_idx)
    y_test_actual = inverse_transform_single(scaler, y_test, target_idx)

    # MAPE 计算（避免除零）
    def mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        non_zero = y_true != 0
        if not np.any(non_zero):
            return np.nan
        mape = np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero]))
        return mape

    test_mae = mean_absolute_error(y_test_actual, test_predict_actual)
    test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_predict_actual))
    test_mape = mean_absolute_percentage_error(y_test_actual, test_predict_actual)

    print("\n" + "=" * 50)
    print(f"📊 {address} 模型评估结果")
    print("=" * 50)
    print(f"测试集 - MAE: {test_mae:.2f}, RMSE: {test_rmse:.2f}, MAPE: {test_mape:.2f}%")
    print("=" * 50)

    # === 准备返回给前端的数据 ===
    test_time_full = test_df.index[look_back:]  # 对应 y_test_actual 的时间戳
    timestamps_full = test_time_full.strftime('%Y-%m-%d %H:%M:%S').tolist()
    actual_full = y_test_actual.tolist()
    predicted_full = test_predict_actual.tolist()

    # 最近48小时（假设每行是1小时；若非小时粒度，可调整 HOURS_TO_SHOW 含义）
    HOURS_TO_SHOW = 192
    if len(timestamps_full) >= HOURS_TO_SHOW:
        recent_slice = slice(-HOURS_TO_SHOW, None)
    else:
        recent_slice = slice(None)

    result = {
        "metrics": {
            "mae": float(test_mae),
            "rmse": float(test_rmse),
            "mape": float(test_mape) if not np.isnan(test_mape) else None
        },
        "plots":{
            "main": {
                "timestamps": timestamps_full,
                "actual": actual_full,
                "predicted": predicted_full
            },
            "zoom": {
                "timestamps": timestamps_full[recent_slice],
                "actual": actual_full[recent_slice],
                "predicted": predicted_full[recent_slice]
            }
        }
    }

    print("\n✅ 数据已准备完毕，可返回前端用于绘图。")
    return result