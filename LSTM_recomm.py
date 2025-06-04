import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
import joblib
import os
import gc
from tqdm import tqdm

# === 1. 載入資料 ===
df = pd.read_csv('data/data.csv')
df['date'] = pd.to_datetime(df['date'])
df.sort_values(['stock_code', 'date'], inplace=True)

# === 2. 特徵工程 ===
df['mid_price'] = (df['highest_price'] - df['lowest_price']) / 2
df['price_diff'] = df['opening_price'] - df['closing_price']

# 使用這些特徵
features = ['opening_price', 'closing_price', 'highest_price', 'lowest_price', 'mid_price', 'price_diff']
target = 'kd_k'
df.dropna(subset=features + [target], inplace=True)

# === 3. 標準化器建立（全域共用） ===
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# === 4. 定義 LSTM 模型 ===
class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out).squeeze()

model = LSTMRegressor(input_size=len(features))
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# === 5. 分批訓練每支股票 ===
sequence_length = 15
epochs = 3

for stock_id, group in tqdm(df.groupby('stock_code'), desc='股票訓練迴圈'):
    group = group.reset_index(drop=True)
    X_list, y_list = [], []
    for i in range(len(group) - sequence_length):
        X_seq = group.loc[i:i+sequence_length-1, features].values
        y_target = group.loc[i + sequence_length, target]
        X_list.append(X_seq)
        y_list.append(y_target)

    if len(X_list) == 0:
        continue

    X_stock = torch.tensor(np.array(X_list), dtype=torch.float32)
    y_stock = torch.tensor(np.array(y_list), dtype=torch.float32)

    dataset = TensorDataset(X_stock, y_stock)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for xb, yb in tqdm(loader, desc=f'{stock_id} Epoch {epoch+1}/{epochs}', leave=False):
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    del X_list, y_list, X_stock, y_stock, dataset, loader
    gc.collect()

# === 6. 評估：使用隨機抽樣部分資料進行快速測試 ===
model.eval()
y_true_all = []
y_pred_all = []

sample_stocks = df['stock_code'].unique()  # 只評估前 30 支股票

with torch.no_grad():
    for stock_id in tqdm(sample_stocks, desc='模型評估中'):
        group = df[df['stock_code'] == stock_id].reset_index(drop=True)
        X_list, y_list = [], []
        for i in range(len(group) - sequence_length):
            X_seq = group.loc[i:i+sequence_length-1, features].values
            y_target = group.loc[i + sequence_length, target]
            X_list.append(X_seq)
            y_list.append(y_target)

        if len(X_list) == 0:
            continue

        X_batch = torch.tensor(np.array(X_list), dtype=torch.float32)
        y_batch = np.array(y_list)
        pred_batch = model(X_batch).numpy()

        y_true_all.extend(y_batch.tolist())
        y_pred_all.extend(pred_batch.tolist())

        del X_list, y_list, X_batch, y_batch, pred_batch
        gc.collect()

    mse = mean_squared_error(y_true_all, y_pred_all)
    print(f"LSTM 分批訓練模型測試集 MSE: {mse:.4f}")

# === 7. 儲存模型與標準化器 ===
os.makedirs('./model/LSTM', exist_ok=True)
torch.save(model.state_dict(), './model/LSTM/lstm_model_15.pth')
joblib.dump(scaler, './model/LSTM/lstm_scaler_15.joblib')
