import pandas as pd
import torch
import joblib
from sklearn.preprocessing import StandardScaler

# === 1. 載入資料，只取每支股票最新一筆 ===
df = pd.read_csv("./data/data.csv")

# 將 date 欄位轉為 datetime 型別並排序
df['date'] = pd.to_datetime(df['date'])
df = df[['stock_code', 'date', 'opening_price', 'closing_price', 'highest_price', 'lowest_price']].dropna()
df = df.sort_values('date')

# 取每支股票的最新一筆資料
latest_df = df.groupby('stock_code').tail(1).copy()

# 只保留等於最新日期的資料（強制推薦當天的）
latest_df = latest_df[latest_df['date'] == latest_df['date'].max()]
print("使用推薦資料的日期為：", latest_df['date'].unique()[0])

# === 2. 特徵處理 ===
latest_df['price_range_half'] = (latest_df['highest_price'] - latest_df['lowest_price']) / 2
latest_df['open_close_diff'] = latest_df['opening_price'] - latest_df['closing_price']

features = ['opening_price', 'closing_price', 'highest_price', 'lowest_price',
            'price_range_half', 'open_close_diff']

# === 3. 載入模型與標準化器 ===
k = 4
centers = torch.load(f"./model/kmeans/kmeans_centers {k}.pt")
scaler = joblib.load(f"./model/kmeans/kmeans_scaler {k}.joblib")

# === 4. 標準化並分群 ===
X_scaled = scaler.transform(latest_df[features])
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
distances = torch.cdist(X_tensor, centers)
latest_df['cluster'] = distances.argmin(dim=1).numpy()

# === 5. 每群的特徵平均值 ===
group_summary = latest_df.groupby("cluster")[features].mean().round(2)
print("\n每群特徵平均值：")
print(group_summary)

# === 6. 每群前 5 支股票（包含日期）===
for i in sorted(latest_df['cluster'].unique()):
    print(f"\n=== 群 {i} 的前 5 支股票 ===")
    print(latest_df[latest_df['cluster'] == i].head(5)[['stock_code', 'date'] + features])
