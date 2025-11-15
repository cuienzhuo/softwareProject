import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

# 固定随机种子
torch.manual_seed(42)
np.random.seed(42)

CSV_PATH = r"E:\github01\softwareProject\analysis-django\analysis\backend\utils\milano_traffic_nid.csv"

# --------------------- 1. 数据加载与预处理 ---------------------
def load_and_preprocess_single_column_data(file_path, target_column, lookback=24, forecast_horizon=12):
    df = pd.read_csv(file_path, parse_dates=["timestamp"], index_col="timestamp")
    df = df.sort_index()

    raw_series = df[target_column].dropna()
    if len(raw_series) < lookback + forecast_horizon:
        return None, None, None, df

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(raw_series.values.reshape(-1, 1))

    X, y = [], []
    for i in range(len(data_scaled) - lookback - forecast_horizon + 1):
        X.append(data_scaled[i:i + lookback])
        y.append(data_scaled[i + lookback:i + lookback + forecast_horizon])

    X_tensor = torch.FloatTensor(np.array(X, dtype=np.float32))
    y_tensor = torch.FloatTensor(np.array(y, dtype=np.float32))

    return X_tensor, y_tensor, scaler, df


# --------------------- 2. 数据集类 ---------------------
class TrafficDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# --------------------- 3. Autoformer 模型 ---------------------
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
        x_emb = self.input_proj(x)
        memory = self.encoder(x_emb)
        decoder_input = memory[:, -1:, :].repeat(1, self.output_dim, 1)
        output = self.decoder(decoder_input, memory)
        return self.output_proj(output).squeeze(-1)


# --------------------- 4. 训练函数 ---------------------
def train_model_for_single_column(X_tensor, y_tensor, col_name, lookback, forecast_horizon, epochs=50, device="cpu"):
    dataset = TrafficDataset(X_tensor, y_tensor)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = Autoformer(
        input_dim=1,
        hidden_dim=64,
        output_dim=forecast_horizon,
        seq_len=lookback,
        num_layers=2,
        num_heads=4
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    print(f"开始训练列 {col_name}...")

    for epoch in range(epochs):
        running_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.squeeze())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    return model


# --------------------- 5. 滚动预测函数 ---------------------
def rolling_predict_column(model, scaler, df, column_name, lookback=24, roll_steps=100, device="cpu"):
    raw_series = df[column_name].dropna()
    raw_data = raw_series.values
    timestamps = raw_series.index

    if len(raw_data) < lookback + roll_steps:
        roll_steps = max(0, len(raw_data) - lookback)

    if roll_steps <= 0:
        return {
            "input_history": [],
            "predictions": [],
            "truths": [],
            "timestamps_input": [],
            "timestamps_pred": [],
            "metrics": {"mae": None, "rmse": None, "mape": None}
        }

    input_history = raw_data[:lookback].tolist()
    input_timestamps = timestamps[:lookback].strftime('%Y-%m-%d %H:%M:%S').tolist()

    current_window = raw_data[:lookback].copy()
    predictions = []
    truths = []
    pred_timestamps = []

    model.eval()
    with torch.no_grad():
        for i in range(roll_steps):
            true_idx = lookback + i
            if true_idx >= len(raw_data):
                break

            # 标准化窗口
            window_scaled = scaler.transform(current_window.reshape(-1, 1)).reshape(1, lookback, 1)
            window_tensor = torch.FloatTensor(window_scaled).to(device)

            # 预测（取第1步）
            pred_scaled = model(window_tensor).cpu().numpy().flatten()[0]
            pred_original = scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]

            predictions.append(float(pred_original))
            truths.append(float(raw_data[true_idx]))
            pred_timestamps.append(timestamps[true_idx].strftime('%Y-%m-%d %H:%M:%S'))

            # 滚动更新：用真实值推进窗口
            current_window = np.append(current_window[1:], raw_data[true_idx])

    # 计算指标
    if len(truths) > 0:
        mae = float(mean_absolute_error(truths, predictions))
        rmse = float(np.sqrt(mean_squared_error(truths, predictions)))
        # MAPE (avoid division by zero)
        truths_arr = np.array(truths)
        preds_arr = np.array(predictions)
        non_zero = truths_arr != 0
        if np.any(non_zero):
            mape = float(np.mean(np.abs((truths_arr[non_zero] - preds_arr[non_zero]) / truths_arr[non_zero])))
        else:
            mape = None
    else:
        mae = rmse = mape = None

    return {
        "plots":{
            "main":{
                "input_history": input_history,
                "predictions": predictions,
                "truths": truths,
                "timestamps_input": input_timestamps,
                "timestamps_pred": pred_timestamps,
            }
        },
        "metrics": {
            "mae": mae,
            "rmse": rmse,
            "mape": mape
        }
    }


# --------------------- 6. 主函数：返回结构化结果 ---------------------
def main(address, lookback=24, forecast_horizon=12, epochs=50, roll_steps=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    X_tensor, y_tensor, scaler, df = load_and_preprocess_single_column_data(
        CSV_PATH, address, lookback=lookback, forecast_horizon=forecast_horizon
    )

    if X_tensor is None:
        return {"error": f"列 '{address}' 数据不足或不存在"}

    print(f"开始训练并预测列: {address}")
    model = train_model_for_single_column(
        X_tensor, y_tensor, address,
        lookback=lookback, forecast_horizon=forecast_horizon,
        epochs=epochs, device=device
    )

    result = rolling_predict_column(
        model, scaler, df, address,
        lookback=lookback, roll_steps=roll_steps, device=device
    )

    result["target_column"] = address
    print(f"✅ 列 {address} 处理完成，返回前端数据。")
    return result