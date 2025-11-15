import pandas as pd
import numpy as np
import torch
import os
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch.nn as nn
from .OssUtils import OssUtils  # 如果后续不需要可删除

# 固定随机种子
torch.manual_seed(42)
np.random.seed(42)

# --------------------- 1. Autoformer 模型定义（不变） ---------------------
class Autoformer(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=12, seq_len=24, num_layers=2, num_heads=4):
        super().__init__()
        self.seq_len = seq_len
        self.output_dim = output_dim

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        x_emb = self.input_proj(x)
        memory = self.encoder(x_emb)
        decoder_input = memory[:, -1:, :].repeat(1, self.output_dim, 1)
        output = self.decoder(decoder_input, memory)
        output = self.output_proj(output).squeeze(-1)
        return output


# --------------------- 2. 加载模型与标准化器（不变） ---------------------
def load_model_and_scalers(col_name, save_dir, device="cpu"):
    model_path = os.path.join(save_dir, f"{col_name}_autoformer.pth")
    scaler_path = os.path.join(save_dir, f"{col_name}_scaler.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"列 {col_name} 的模型文件不存在: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"列 {col_name} 的标准化器文件不存在: {scaler_path}")

    model = Autoformer(
        input_dim=1,
        hidden_dim=64,
        output_dim=12,
        seq_len=24,
        num_layers=2,
        num_heads=4
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    scalers = joblib.load(scaler_path)
    return model, scalers


# --------------------- 3. 滚动预测函数（不变） ---------------------
def rolling_predict_column(model, scalers, df, column_name, lookback=24, roll_steps=100, device="cpu"):
    if column_name not in scalers:
        raise ValueError(f"列 {column_name} 无训练数据，请检查列名")

    scaler, _ = scalers[column_name]
    raw_data = df[[column_name]].dropna().values.flatten()

    if len(raw_data) < lookback + roll_steps:
        effective_steps = len(raw_data) - lookback
        print(f"警告：列 {column_name} 数据量不足，实际滚动 {effective_steps} 步")
        roll_steps = effective_steps
        if roll_steps <= 0:
            return np.array([]), np.array([]), np.array([])

    input_history = raw_data[:lookback].copy()
    predictions, true_values = [], []

    model.eval()
    with torch.no_grad():
        current_window = input_history.copy()
        for i in range(roll_steps):
            window_scaled = scaler.transform(current_window.reshape(-1, 1)).reshape(1, lookback, 1)
            window_tensor = torch.FloatTensor(window_scaled).to(device)

            pred_scaled = model(window_tensor).cpu().numpy().flatten()[0]
            pred_original = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()[0]
            predictions.append(pred_original)

            true_idx = lookback + i
            if true_idx < len(raw_data):
                true_val = raw_data[true_idx]
                true_values.append(true_val)
                current_window = np.append(current_window[1:], true_val)
            else:
                break

    return input_history, np.array(predictions), np.array(true_values)


# --------------------- 4. 新增：计算评估指标（含 MAPE） ---------------------
def calculate_metrics(y_true, y_pred):
    if len(y_true) == 0 or len(y_pred) == 0 or len(y_true) != len(y_pred):
        return {"mae": None, "rmse": None, "mape": None}

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # MAPE: 避免除零（当真实值为0时跳过或设为0）
    non_zero_mask = y_true != 0
    if np.any(non_zero_mask):
        mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask]))
    else:
        mape = float('nan')  # 或设为 None

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape) if not np.isnan(mape) else None
    }


# --------------------- 5. 主推理函数：返回绘图数据 + 指标 ---------------------
def main_single_column_inference(
    csv_file_path,
    target_column,
    model_dir="E:/github01/softwareModel/transformerModels",
    roll_steps=100
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    df = pd.read_csv(csv_file_path, parse_dates=["timestamp"], index_col="timestamp")
    df = df.sort_index()

    if target_column not in df.columns:
        raise ValueError(f"目标列 {target_column} 不在CSV数据中")

    print(f"\n===== 开始预测目标列: {target_column} =====")

    try:
        model, scalers = load_model_and_scalers(target_column, save_dir=model_dir, device=device)
        print(f"成功加载列 {target_column} 的模型和标准化器")
    except Exception as e:
        raise RuntimeError(f"加载列 {target_column} 失败: {e}")

    lookback = 24
    input_history, pred, true = rolling_predict_column(
        model, scalers, df, target_column, lookback=lookback, roll_steps=roll_steps, device=device
    )

    # 计算指标
    metrics = calculate_metrics(true, pred)

    # 构造前端绘图所需数据
    # 时间步索引（从0开始）

    pred_x = list(range(len(input_history), len(input_history) + len(pred)))
    pred_y = pred.tolist()

    true_y = true.tolist()

    chart_data = {
        "timestamps": [str(i) for i in (pred_x)],  # 前端可直接用作x轴标签（简化为字符串序号）
        "actuals": true_y,
        "predictions": pred_y  # 注意：预测序列前段用历史值填充，便于连续显示
    }

    # 返回结构化结果
    result = {
        "column": target_column,
        "metrics": metrics,
        "chartData": chart_data
    }

    return result


# --------------------- 6. 对外接口函数 ---------------------
def transformerAnalysis(address, roll_steps=100):
    csv_file_path = "E:/github01/softwareProject/analysis-django/analysis/backend/utils/milano_traffic_nid.csv"
    return main_single_column_inference(
        csv_file_path=csv_file_path,
        target_column=address,
        roll_steps=roll_steps
    )