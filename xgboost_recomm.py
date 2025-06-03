import pandas as pd
import xgboost as xgb
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# === 1. 讀取資料 ===
df = pd.read_csv("./data/data.csv")
df['date'] = pd.to_datetime(df['date'])

# === 2. 特徵工程與目標設計 ===
df = df[['stock_code', 'date', 'opening_price', 'closing_price', 'highest_price', 'lowest_price', 'kd_k', 'kd_d']]
df = df.dropna()

df = df.sort_values(['stock_code', 'date'])
df['price_range_half'] = (df['highest_price'] - df['lowest_price']) / 2
df['open_close_diff'] = df['opening_price'] - df['closing_price']
df['target_k'] = df.groupby('stock_code')['kd_k'].shift(-5)
df = df.dropna(subset=['target_k'])

features = ['opening_price', 'closing_price', 'highest_price', 'lowest_price',
            'price_range_half', 'open_close_diff']
X = df[features]
y = df['target_k']

# === 3. 特徵標準化 ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === 4. 資料分割 ===
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# === 5. 模型訓練 ===
model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# === 6. 模型評估 ===
y_pred = model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
print(f"XGBoost Validation MSE: {mse:.2f}")

# === 7. 儲存模型與標準化器 ===
os.makedirs("./model/xgboost", exist_ok=True)
joblib.dump(model, "./model/xgboost/xgboost_model.joblib")
joblib.dump(scaler, "./model/xgboost/xgboost_scaler.joblib")
print("模型與標準化器已儲存")
