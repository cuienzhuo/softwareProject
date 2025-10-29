import matplotlib
matplotlib.use('Agg')  # 非交互式后端，适合保存图片

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import io
from .OssUtils import OssUtils


# --------------------------
# 1. 配置参数与路径（需确保文件在当前目录）
# --------------------------
MODEL_PATH = "E:\github01\softwareModel\DL_MODEL/traffic_prediction_model.keras"  # 模型文件（修正为.keras格式）
SCALER_PATH = "E:\github01\softwareModel\DL_MODEL/traffic_scaler.pkl"  # 归一化器文件
CSV_PATH = "E:\github01\softwareProject/analysis-django/analysis/backend/utils/milano_traffic_nid.csv"           # 待预测的CSV数据
LOOK_BACK = 60                                # 与训练时一致：用过去60步预测1步
INITIAL_WINDOW_SIZE = 3000                    # 初始历史窗口大小（需 > LOOK_BACK）
ROLL_STEPS = 100                              # 滚动预测的总步数


# --------------------------
# 2. 加载模型、归一化器、原始数据
# --------------------------
def load_dependencies():
    """加载模型、归一化器和原始数据"""
    # 加载模型
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"模型文件 {MODEL_PATH} 不存在，请检查路径")
    model = load_model(MODEL_PATH)  # 明确使用tf.keras的load_model
    print(f"成功加载模型：{MODEL_PATH}（TensorFlow {tf.__version__}）")

    # 加载归一化器
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"归一化器 {SCALER_PATH} 不存在，请检查路径")
    scaler = joblib.load(SCALER_PATH)
    print(f"成功加载归一化器：{SCALER_PATH}")

    # 加载CSV数据
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"数据文件 {CSV_PATH} 不存在，请检查路径")
    df = pd.read_csv(CSV_PATH, index_col="timestamp", parse_dates=True)
    print(f"成功加载数据：{CSV_PATH}，共 {len(df)} 条记录，{df.shape[1]} 个特征列")

    return model, scaler, df


# --------------------------
# 3. 准备滚动检验数据（避免数据长度不足）
# --------------------------
def prepare_rolling_data(df):
    """截取用于滚动检验的数据片段，自动调整步数以匹配原始数据长度"""
    total_needed = INITIAL_WINDOW_SIZE + ROLL_STEPS
    if total_needed > len(df):
        total_needed = len(df)
        adjusted_roll_steps = total_needed - INITIAL_WINDOW_SIZE
        print(f"警告：原始数据长度不足，滚动步数自动调整为 {adjusted_roll_steps}（原始数据共 {len(df)} 条）")
        return df.iloc[:total_needed], adjusted_roll_steps
    return df.iloc[:total_needed], ROLL_STEPS


# --------------------------
# 4. 执行历史滚动预测（核心逻辑）
# --------------------------
def run_rolling_test(model, scaler, df_test, roll_steps, num_features):
    """
    滚动预测逻辑：
    1. 用初始窗口数据预测下1步
    2. 用**真实值**更新窗口（避免预测值污染历史数据）
    3. 重复上述步骤完成所有滚动步数
    """
    predictions = []  # 存储原始尺度的预测值
    actuals = []      # 存储原始尺度的真实值
    rmse_list = []    # 存储每步的RMSE

    # 初始化滚动窗口（前INITIAL_WINDOW_SIZE条数据）
    rolling_data = df_test.iloc[:INITIAL_WINDOW_SIZE].values
    rolling_data_scaled = scaler.transform(rolling_data)  # 归一化（复用训练好的scaler）

    for i in range(roll_steps):
        # ① 提取输入序列：最近LOOK_BACK个时间步的所有特征
        input_seq = rolling_data_scaled[-LOOK_BACK:].reshape(1, LOOK_BACK, num_features)

        # ② 模型预测（归一化尺度）
        pred_scaled = model.predict(input_seq, verbose=0)[0]  # 输出形状：(num_features,)

        # ③ 反归一化到原始交通流量尺度
        pred_actual = scaler.inverse_transform(pred_scaled.reshape(1, -1))[0]

        # ④ 获取当前步的真实值（原始尺度）
        actual_idx = INITIAL_WINDOW_SIZE + i
        actual_actual = df_test.iloc[actual_idx].values

        # ⑤ 用真实值的归一化结果更新滚动窗口（关键：用真实值保证窗口真实性）
        actual_scaled = scaler.transform(actual_actual.reshape(1, -1))[0]
        rolling_data_scaled = np.append(
            rolling_data_scaled[1:],  # 移除最旧的1条数据
            actual_scaled.reshape(1, -1),  # 加入最新真实值的归一化结果
            axis=0
        )

        # ⑥ 记录结果
        predictions.append(pred_actual)
        actuals.append(actual_actual)

        # ⑦ 计算单步RMSE
        step_rmse = np.sqrt(mean_squared_error(actual_actual, pred_actual))
        rmse_list.append(step_rmse)

        # 打印进度（每10步提示一次）
        if (i + 1) % 10 == 0:
            print(f"滚动进度：{i+1}/{roll_steps} 步 | 当前步RMSE：{step_rmse:.2f}")

    return np.array(predictions), np.array(actuals), np.array(rmse_list)


# --------------------------
# 5. 结果可视化（预测值与真实值同图对比）
# --------------------------
def visualize_results(df_test, predictions, actuals, feature_name):
    """生成滚动检验的可视化图表（RMSE趋势 + 预测/真实值对比）"""
    # 创建结果保存目录
    if not os.path.exists("rolling_test_results"):
        os.makedirs("rolling_test_results")

    # ---- 5.3 预测值与真实值对比图 ----
    feature_idx = df_test.columns.get_loc(feature_name)  # 获取特征列索引
    pred_feature = predictions[:, feature_idx]
    actual_feature = actuals[:, feature_idx]
    # 时间轴：对应滚动预测的真实时间
    time_index = df_test.index[INITIAL_WINDOW_SIZE:INITIAL_WINDOW_SIZE + len(predictions)]

    plt.figure(figsize=(15, 8))
    plt.plot(time_index, actual_feature, color="forestgreen", linewidth=2, label=f"真实值（{feature_name}）")
    plt.plot(time_index, pred_feature, color="darkorange", linestyle="--", linewidth=2, label=f"预测值（{feature_name}）")
    plt.title(f"历史滚动检验 - {feature_name} 预测对比")
    plt.xlabel("时间")
    plt.ylabel("交通流量")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    buffer = io.BytesIO()  # 初始化内存缓冲区
    plt.savefig(buffer, dpi=300, bbox_inches='tight', format='png')  # 保存图表到缓冲区（指定格式为png）
    buffer.seek(0)  # 将缓冲区指针移到开头，否则读取不到数据

    save_path = f"DL/{feature_name}.png"
    ossUtils = OssUtils()
    image_url = ossUtils.OssUpload(buffer, save_path)

    plt.close()

    return image_url


# --------------------------
# 主函数：串联全流程
# --------------------------
def DLAnalysis(address:str) -> str:
    # 加载依赖
    model, scaler, df = load_dependencies()
    num_features = df.shape[1]  # 特征列数量（需与训练时一致）

    # 准备滚动检验数据
    df_test, roll_steps = prepare_rolling_data(df)

    # 执行滚动检验
    print(f"\n开始历史滚动检验（初始窗口大小：{INITIAL_WINDOW_SIZE}，滚动步数：{roll_steps}）...")
    predictions, actuals, rmse_list = run_rolling_test(model, scaler, df_test, roll_steps, num_features)

    # 可视化结果
    return visualize_results(df_test, predictions, actuals, address)