import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from transformers import PatchTSTForPrediction, PatchTSTConfig
import warnings

warnings.filterwarnings("ignore")

CSV_PATH = r"E:\github01\softwareProject\analysis-django\analysis\backend\utils\milano_traffic_nid.csv"

def patchtstAnalysis(address, train_ratio=0.8, seq_len=96, pred_len=24, patch_len=16, stride=8, epochs=10, batch_size=32):
    df = pd.read_csv(CSV_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')

    data_values = df[address].values.reshape(-1, 1)

    train_size = int(len(data_values) * train_ratio)
    train_data = data_values[:train_size]
    test_data = data_values[train_size:]

    # 标准化
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)

    class TimeSeriesDataset(Dataset):
        def __init__(self, data, seq_len, pred_len):
            self.data = data
            self.seq_len = seq_len
            self.pred_len = pred_len

        def __len__(self):
            return len(self.data) - self.seq_len - self.pred_len + 1

        def __getitem__(self, idx):
            end_idx_x = idx + self.seq_len
            end_idx_y = end_idx_x + self.pred_len
            past_values = self.data[idx:end_idx_x]
            future_values = self.data[end_idx_x:end_idx_y]
            return torch.tensor(past_values, dtype=torch.float32), \
                   torch.tensor(future_values, dtype=torch.float32)

    train_dataset = TimeSeriesDataset(train_scaled, seq_len, pred_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    input_dim = 1
    config = PatchTSTConfig(
        context_length=seq_len,
        prediction_length=pred_len,
        patch_length=patch_len,
        stride=stride,
        input_channels=input_dim,
        num_input_channels=input_dim,
        head_type="prediction"
    )
    model = PatchTSTForPrediction(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_past_values, batch_future_values in train_loader:
            batch_past_values = batch_past_values.to(device)
            batch_future_values = batch_future_values.to(device)

            optimizer.zero_grad()
            outputs = model(past_values=batch_past_values, future_values=batch_future_values).prediction_outputs
            loss = criterion(outputs, batch_future_values)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    # 滚动预测
    model.eval()
    true_values = []
    predicted_values = []

    with torch.no_grad():
        for i in range(len(test_data) - seq_len - pred_len + 1):
            past_values_input = test_scaled[i: i + seq_len]
            true_future_values = test_scaled[i + seq_len: i + seq_len + pred_len]

            past_values_input = torch.tensor(past_values_input, dtype=torch.float32).unsqueeze(0).to(device)
            outputs = model(past_values=past_values_input).prediction_outputs
            prediction = outputs.squeeze(0).cpu().numpy()

            prediction_unscaled = scaler.inverse_transform(prediction.reshape(-1, input_dim)).flatten()
            true_unscaled = scaler.inverse_transform(true_future_values.reshape(-1, input_dim)).flatten()

            true_values.extend(true_unscaled.tolist())
            predicted_values.extend(prediction_unscaled.tolist())

    min_len = min(len(true_values), len(predicted_values))
    true_values = true_values[:min_len]
    predicted_values = predicted_values[:min_len]

    # 评估指标（全局）
    if len(true_values) > 0:
        true_values_np = np.array(true_values)
        predicted_values_np = np.array(predicted_values)
        mae = float(mean_absolute_error(true_values_np, predicted_values_np))
        rmse = float(np.sqrt(mean_squared_error(true_values_np, predicted_values_np)))
        mape = float(np.mean(np.abs((true_values_np - predicted_values_np) / (true_values_np + 1e-8))))
    else:
        mae, rmse, mape = None, None, None

    # 获取对应的时间戳（对齐预测结果）
    start_idx_plot = train_size + seq_len
    end_idx_plot = start_idx_plot + len(true_values)
    plot_timestamps = df.index[start_idx_plot:end_idx_plot].strftime('%Y-%m-%d %H:%M:%S').tolist()

    # 确保长度一致
    if len(plot_timestamps) != len(true_values):
        min_plot_len = min(len(plot_timestamps), len(true_values))
        plot_timestamps = plot_timestamps[:min_plot_len]
        true_values = true_values[:min_plot_len]
        predicted_values = predicted_values[:min_plot_len]

    # 构建返回数据
    main = {
        "timestamps": plot_timestamps,
        "true_values": true_values,
        "predicted_values": predicted_values,
    }
    metrics = {
        "mae": round(mae, 2) if mae is not None else None,
        "rmse": round(rmse, 2) if rmse is not None else None,
        "mape": round(mape, 2) if mape is not None else None
    }

    # 可选：添加48小时子集（假设10分钟间隔 → 288点）
    points_48_hours = 288
    if len(true_values) >= points_48_hours:
        true_48h = true_values[:points_48_hours]
        pred_48h = predicted_values[:points_48_hours]
        ts_48h = plot_timestamps[:points_48_hours]

        true_48h_np = np.array(true_48h)
        pred_48h_np = np.array(pred_48h)

        zoom = {
            "timestamps": ts_48h,
            "true_values": true_48h,
            "predicted_values": pred_48h,
        }
    else:
        zoom = None

    return {
        "plots":{
            "main": main,
            "zoom": zoom,
        },
        "metrics": metrics
    }