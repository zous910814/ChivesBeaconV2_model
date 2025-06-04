import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
import joblib

# === 1. 資料與模型路徑 ===
data_path = 'data/data.csv'
model_path = './model/LSTM/lstm_model_15.pth'
scaler_path = './model/LSTM/lstm_scaler_15.joblib'

# === 2. 特徵設定 ===
features = ['opening_price', 'closing_price', 'highest_price', 'lowest_price', 'mid_price', 'price_diff']

# === 3. 載入資料 ===
df = pd.read_csv(data_path)
df['date'] = pd.to_datetime(df['date'])
df.sort_values(['stock_code', 'date'], inplace=True)
df['mid_price'] = (df['highest_price'] - df['lowest_price']) / 2
df['price_diff'] = df['opening_price'] - df['closing_price']
df.dropna(subset=features, inplace=True)

# === 4. 定義模型結構 ===
class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out).squeeze()

# === 5. 載入模型與 scaler ===
scaler = joblib.load(scaler_path)
model = LSTMRegressor(input_size=len(features))
model.load_state_dict(torch.load(model_path))
model.eval()

# === 6. 預測與推薦 ===
latest_date = df['date'].max()
recommendations = []

for stock_id, group in df.groupby('stock_code'):
    group = group[group['date'] < latest_date].reset_index(drop=True)
    if len(group) < 15:
        continue
    seq = group.iloc[-15:][features].values
    seq_scaled = scaler.transform(seq)
    X_input = torch.tensor(seq_scaled[np.newaxis, :, :], dtype=torch.float32)
    with torch.no_grad():
        pred = model(X_input).item()
    recommendations.append({
        'stock_code': stock_id,
        'date': latest_date.strftime('%Y-%m-%d'),
        'kd_k_predicted': round(pred, 2)
    })

# === 7. 輸出推薦結果 ===
top10_df = pd.DataFrame(sorted(recommendations, key=lambda x: x['kd_k_predicted'], reverse=True)[:10])
print("\nLSTM 推薦 KD_K 最高的 10 支股票：")
print(top10_df)

