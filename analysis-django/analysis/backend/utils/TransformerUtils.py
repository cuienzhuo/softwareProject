import matplotlib
matplotlib.use('Agg')  # 非交互式环境保存图片
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch.nn as nn
import io
from .OssUtils import OssUtils

# 固定随机种子（保证预测过程可复现）
torch.manual_seed(42)
np.random.seed(42)

# --------------------- 1. 复用Autoformer模型结构（必须与训练时一致） ---------------------
class Autoformer(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=12, seq_len=24, num_layers=2, num_heads=4):
        super().__init__()
        self.seq_len = seq_len       # 输入历史窗口长度
        self.output_dim = output_dim # 单次预测步长

        # 输入投影（单变量→隐藏维度）
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # 编码器（学习历史时序特征）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 解码器（生成未来时序）
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # 输出投影（隐藏维度→预测值）
        self.output_proj = nn.Linear(hidden_dim, 1)  # 单步输出，多步通过重复拼接

    def forward(self, x):
        batch_size = x.shape[0]
        # 输入编码：单变量→隐藏维度
        x_emb = self.input_proj(x)  # shape: (batch_size, seq_len, hidden_dim)
        # 编码器提取历史特征
        memory = self.encoder(x_emb)  # shape: (batch_size, seq_len, hidden_dim)
        # 解码器输入：重复编码器最后一步，作为多步预测的初始输入
        decoder_input = memory[:, -1:, :].repeat(1, self.output_dim, 1)  # shape: (batch_size, output_dim, hidden_dim)
        # 解码器生成未来序列
        output = self.decoder(decoder_input, memory)  # shape: (batch_size, output_dim, hidden_dim)
        # 投影到最终预测值（展平最后一维）
        output = self.output_proj(output).squeeze(-1)  # shape: (batch_size, output_dim)
        return output


# --------------------- 2. 模型与标准化器加载函数 ---------------------
def load_model_and_scalers(col_name, save_dir, device="cpu"):
    """按列名加载对应的模型权重与标准化器"""
    # 检查模型文件是否存在
    model_path = os.path.join(save_dir, f"{col_name}_autoformer.pth")
    scaler_path = os.path.join(save_dir, f"{col_name}_scaler.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"列 {col_name} 的模型文件不存在: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"列 {col_name} 的标准化器文件不存在: {scaler_path}")

    # 初始化模型结构
    model = Autoformer(
        input_dim=1,
        hidden_dim=64,
        output_dim=12,
        seq_len=24,
        num_layers=2,
        num_heads=4
    ).to(device)
    # 加载模型权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 切换为评估模式
    # 加载标准化器
    scalers = joblib.load(scaler_path)
    return model, scalers


# --------------------- 3. 历史滚动预测函数 ---------------------
def rolling_predict_column(model, scalers, df, column_name, lookback=24, roll_steps=100, device="cpu"):
    """滚动预测逻辑（仅预测指定列）"""
    if column_name not in scalers:
        raise ValueError(f"列 {column_name} 无训练数据，请检查列名")

    scaler, _ = scalers[column_name]
    raw_data = df[[column_name]].dropna().values.flatten()  # 原始数据（一维数组）

    # 校验数据量是否足够
    if len(raw_data) < lookback + roll_steps:
        effective_steps = len(raw_data)-lookback
        print(f"警告：列 {column_name} 数据量不足，实际滚动 {effective_steps} 步（需至少 {lookback+roll_steps} 步）")
        roll_steps = effective_steps
        if roll_steps <= 0:
            return np.array([]), np.array([]), np.array([])

    input_history = raw_data[:lookback].copy()  # 初始历史窗口
    predictions, true_values = [], []          # 存储预测值、真实值

    model.eval()
    with torch.no_grad():  # 关闭梯度计算
        current_window = input_history.copy()
        for i in range(roll_steps):
            # 标准化当前窗口（适配模型输入形状）
            window_scaled = scaler.transform(current_window.reshape(-1, 1)).reshape(1, lookback, 1)
            window_tensor = torch.FloatTensor(window_scaled).to(device)

            # 模型预测（取多步预测的第1步）
            pred_scaled = model(window_tensor).cpu().numpy().flatten()[0]

            # 反标准化为原始尺度
            pred_original = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()[0]
            predictions.append(pred_original)

            # 获取真实值并更新窗口
            true_idx = lookback + i
            if true_idx < len(raw_data):
                true_val = raw_data[true_idx]
                true_values.append(true_val)
                current_window = np.append(current_window[1:], true_val)  # 滚动更新窗口
            else:
                break

    return input_history, np.array(predictions), np.array(true_values)


# --------------------- 4. 主函数：仅预测指定的单个列 ---------------------
def main_single_column_inference(
    csv_file_path,
    target_column,  # 目标列名（仅预测这一列）
    model_dir="E:\github01\softwareModel/transformerModels",
    plot_dir="TRANSFORMER_PLOTS"
):
    # 设备选择（优先GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载原始CSV数据
    print("加载原始数据...")
    df = pd.read_csv(csv_file_path, parse_dates=["timestamp"], index_col="timestamp")
    df = df.sort_index()

    # 检查目标列是否在数据中
    if target_column not in df.columns:
        raise ValueError(f"目标列 {target_column} 不在CSV数据中，请检查列名")

    print(f"\n===== 开始预测目标列: {target_column} =====")

    # 步骤1：加载该列的模型与标准化器
    try:
        model, scalers = load_model_and_scalers(target_column, save_dir=model_dir, device=device)
        print(f"成功加载列 {target_column} 的模型和标准化器")
    except Exception as e:
        print(f"加载列 {target_column} 失败: {e}")
        return

    # 步骤2：执行历史滚动预测
    lookback, roll_steps = 24, 100  # 可根据需要调整
    print(f"对列 {target_column} 执行滚动预测，步数: {roll_steps}")
    input_history, pred, true = rolling_predict_column(
        model, scalers, df, target_column, lookback=lookback, roll_steps=roll_steps, device=device
    )

    # 步骤3：计算预测指标（MSE、MAE）
    if len(true) > 0 and len(pred) == len(true):
        mse = mean_squared_error(true, pred)
        mae = mean_absolute_error(true, pred)
        print(f"列 {target_column} 预测指标 - MSE: {mse:.4f}, MAE: {mae:.4f}")
    else:
        print(f"列 {target_column} 数据长度不匹配，无法计算指标")

    # 步骤4：可视化预测结果
    plt.figure(figsize=(15, 6))
    # 绘制输入的历史窗口
    input_x = range(len(input_history))
    plt.plot(input_x, input_history, label="输入历史数据", color="blue", alpha=0.7)
    # 绘制预测序列
    pred_x = range(len(input_history), len(input_history) + len(pred))
    plt.plot(pred_x, pred, label="预测值 (predict)", color="red", linestyle="--", linewidth=2)
    # 绘制真实序列（若存在）
    if len(true) > 0:
        true_x = range(len(input_history), len(input_history) + len(true))
        plt.plot(true_x, true, label="真实值 (truth)", color="green", linewidth=2)

    plt.title(f"{target_column} - 历史滚动预测对比")
    plt.xlabel("时间步")
    plt.ylabel("交通流量")
    plt.legend()
    plt.grid(alpha=0.3, linestyle="--")

    buffer = io.BytesIO()  # 初始化内存缓冲区
    plt.savefig(buffer, dpi=300, bbox_inches='tight', format='png')  # 保存图表到缓冲区（指定格式为png）
    buffer.seek(0)  # 将缓冲区指针移到开头，否则读取不到数据

    save_path = f"transformer/{target_column}.png"
    ossUtils = OssUtils()
    image_url = ossUtils.OssUpload(buffer, save_path)

    plt.close()

    return image_url


def transformerAnalysis(address:str) -> str:
    # --------------------------
    # 在这里修改目标列名和CSV路径
    csv_file_path = "E:\github01\softwareProject/analysis-django/analysis/backend/utils/milano_traffic_nid.csv"  # 替换为你的CSV文件路径
    # --------------------------

    return main_single_column_inference(
        csv_file_path=csv_file_path,
        target_column=address
    )