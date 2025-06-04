import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib

# === 1. 載入資料 ===
df = pd.read_csv('data/data.csv')
df['date'] = pd.to_datetime(df['date'])
df.sort_values(['stock_code', 'date'], inplace=True)

# === 2. 特徵工程 ===
df['mid_price'] = (df['highest_price'] - df['lowest_price']) / 2
df['price_diff'] = df['opening_price'] - df['closing_price']

# 去除缺值
df.dropna(subset=['opening_price', 'closing_price', 'highest_price', 'lowest_price', 'kd_k'], inplace=True)

# 特徵與目標欄位
features = ['opening_price', 'closing_price', 'highest_price', 'lowest_price', 'mid_price', 'price_diff']
target = 'kd_k'

# === 3. 載入模型與 scaler ===
model = joblib.load('./model/lightgbm/lightgbm_model.joblib')
scaler = joblib.load('./model/lightgbm/lightgbm_scaler.joblib')

# === 4. 準備測試資料（用最新一天以外資料）===
test_df = df[df['date'] < df['date'].max()].copy()
X_test = scaler.transform(test_df[features])
y_test = test_df[target].values

# === 5. 預測與評估 ===
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print(f"LightGBM 模型測試集 MSE: {mse:.4f}")
