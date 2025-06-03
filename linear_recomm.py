import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib
import os

# === 1. 載入資料 ===
df = pd.read_csv('data/data.csv')
df['date'] = pd.to_datetime(df['date'])
df.sort_values(['stock_code', 'date'], inplace=True)

# === 2. 建立衍生特徵 ===
df['mid_price'] = (df['highest_price'] - df['lowest_price']) / 2
df['open_close_diff'] = df['opening_price'] - df['closing_price']

# === 3. 特徵欄位 ===
feature_cols = [
    'opening_price', 'closing_price', 'highest_price', 'lowest_price',
    'mid_price', 'open_close_diff'
]

# === 4. 建立目標（5日後的 KD_K）===
df['Target'] = df.groupby('stock_code')['kd_k'].shift(-5)

# === 5. 移除缺失值 ===
df_model = df.dropna(subset=feature_cols + ['Target'])

# === 6. 標準化與訓練 ===
scaler = StandardScaler()
X = scaler.fit_transform(df_model[feature_cols])
y = df_model['Target'].values

model = LinearRegression()
model.fit(X, y)

# === 7. 儲存模型與標準化器 ===
os.makedirs('./model', exist_ok=True)
joblib.dump(model, 'model/linear_regression.joblib')
joblib.dump(scaler, 'model/scaler_fresh.joblib')

y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
print(f"\nLinear Regression 訓練集 MSE: {mse:.2f}")

# === 8. 推薦最新一天 ===
latest_date = df['date'].max()
latest_data = df[df['date'] == latest_date].dropna(subset=feature_cols)
X_latest = scaler.transform(latest_data[feature_cols])
latest_data['Predicted_KD_K'] = model.predict(X_latest)

top10 = latest_data.sort_values(by='Predicted_KD_K', ascending=False).head(10)

# === 9. 顯示推薦結果 ===
print(f"\n模型與標準化器已儲存至 ./model/")
print(f"\n最新推薦（{latest_date.date()}）未來KD_K最高的前10支股票：\n")
print(top10[['stock_code', 'date', 'Predicted_KD_K']])
