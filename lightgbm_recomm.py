import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import joblib
import os

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

# === 3. 特徵標準化 ===
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# === 4. 訓練資料（排除最新一天）===
train_df = df[df['date'] < df['date'].max()]
X_train = train_df[features]
y_train = train_df[target]

# === 5. LightGBM 模型訓練 ===
lgb_train = lgb.Dataset(X_train, y_train)
params = {
    'objective': 'regression',
    'metric': 'mse',
    'verbosity': -1,
    'boosting_type': 'gbdt'
}
model = lgb.train(params, lgb_train, num_boost_round=100)

# === 6. 模型與標準化器儲存 ===
os.makedirs('./model/lightgbm', exist_ok=True)
joblib.dump(model, './model/lightgbm/lightgbm_model.joblib')
joblib.dump(scaler, './model/lightgbm/lightgbm_scaler.joblib')

# === 7. 預測最新一天的 K 值，推薦前 10 名股票 ===
latest_date = df['date'].max()
latest_df = df[df['date'] == latest_date].copy()
latest_X = latest_df[features]
latest_df['k_pred'] = model.predict(latest_X)

# 推薦前 10 支股票
top10 = latest_df.sort_values('k_pred', ascending=False).head(10)
print("推薦股票（預測 KD_K 最高）：")
print(top10[['stock_code', 'date', 'k_pred']])
