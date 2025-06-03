import pandas as pd
import joblib
import torch
import os

# === 1. 載入資料 ===
df = pd.read_csv("./data/data.csv")
df['date'] = pd.to_datetime(df['date'])
df = df[['stock_code', 'date', 'opening_price', 'closing_price', 'highest_price', 'lowest_price']]
df = df.dropna()

# === 2. 取整體最新日期 ===
latest_date = df['date'].max()
latest_df = df[df['date'] == latest_date].copy()

# === 3. 建立特徵 ===
latest_df['price_range_half'] = (latest_df['highest_price'] - latest_df['lowest_price']) / 2
latest_df['open_close_diff'] = latest_df['opening_price'] - latest_df['closing_price']

features = ['opening_price', 'closing_price', 'highest_price', 'lowest_price',
            'price_range_half', 'open_close_diff']

# === 4. 載入模型與標準化器 ===
model = joblib.load("./model/xgboost/xgboost_model.joblib")
scaler = joblib.load("./model/xgboost/xgboost_scaler.joblib")

# === 5. 預測 ===
X_scaled = scaler.transform(latest_df[features])
preds = model.predict(X_scaled)
latest_df['predicted_k'] = preds

# === 6. 推薦 K 值最高的前 10 支股票 ===
top10 = latest_df.sort_values('predicted_k', ascending=False).head(10)
print("預測 K 值最高的前 10 支股票：")
print(top10[['stock_code', 'date', 'predicted_k']])
