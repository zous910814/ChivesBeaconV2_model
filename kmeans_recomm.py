import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
import joblib
import os
import numpy as np
from sklearn.metrics import silhouette_score

# === 0. GPU 確認 ===
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用設備：{torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")

# === 1. 載入與轉換日期欄位 ===
df = pd.read_csv("./data/data.csv")
df['date'] = pd.to_datetime(df['date'])  # 確保正確排序
df = df[['stock_code', 'date', 'opening_price', 'closing_price', 'highest_price', 'lowest_price']].dropna()
df = df.sort_values('date')

# === 2. 保留每支股票最新一筆資料 ===
latest_df = df.groupby('stock_code').tail(1).copy()
# 保留距離最新日不超過 3 天內的資料（你可自行調整天數）
# 取所有股票中最新的日期
max_date = latest_df['date'].max()

# 保留距離最新日不超過 3 天內的資料（你可自行調整天數）
latest_df = latest_df[latest_df['date'] >= max_date - pd.Timedelta(days=3)]
print("最新資料日：", latest_df['date'].max())

# === 3. 特徵工程 ===
latest_df['price_range_half'] = (latest_df['highest_price'] - latest_df['lowest_price']) / 2
latest_df['open_close_diff'] = latest_df['opening_price'] - latest_df['closing_price']

features = ['opening_price', 'closing_price', 'highest_price', 'lowest_price',
            'price_range_half', 'open_close_diff']

# === 4. 標準化 ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(latest_df[features])
X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

# === 5. PyTorch 手寫 KMeans 分群 ===
def kmeans_torch(X, k=5, num_iter=100):
    n, d = X.shape
    indices = torch.randperm(n, device=X.device)[:k]
    centroids = X[indices]

    for _ in range(num_iter):
        distances = torch.cdist(X, centroids)
        labels = distances.argmin(dim=1)
        new_centroids = torch.stack([
            X[labels == j].mean(dim=0) if (labels == j).any() else centroids[j]
            for j in range(k)
        ])
        if torch.allclose(centroids, new_centroids, atol=1e-4):
            break
        centroids = new_centroids

    return labels.cpu(), centroids.cpu()

# === 6. 模型訓練與分群 ===
k = 4
labels, centers = kmeans_torch(X_tensor, k=k)
latest_df['cluster'] = labels.numpy()

# === 7. 評估 Inertia + Silhouette Score（抽樣） ===
assigned_centers = centers[labels]
inertia = ((X_tensor.cpu() - assigned_centers.cpu()) ** 2).sum().item()
print(f"Inertia（群內平方誤差平方和）: {inertia:.2f}")

sample_size = min(1000, len(X_scaled))
sample_idx = np.random.choice(len(X_scaled), size=sample_size, replace=False)
X_sample = X_scaled[sample_idx]
labels_sample = labels.numpy()[sample_idx]
sil_score = silhouette_score(X_sample, labels_sample)
print(f"Silhouette Score（抽樣 {sample_size} 筆）: {sil_score:.4f}")

# === 8. 每群推薦代表股票 ===
distances = torch.cdist(torch.tensor(X_scaled, dtype=torch.float32), centers)
latest_df['distance'] = distances.min(dim=1).values.numpy()
recommendations = latest_df.loc[latest_df.groupby('cluster')['distance'].idxmin()][['stock_code', 'date', 'cluster']]
recommendations = recommendations.sort_values('cluster')

print(f"\n每群代表推薦股票(使用最新資料) k={k}")
print(recommendations)

# === 9. 儲存模型與標準化器 ===
os.makedirs("./model/kmeans", exist_ok=True)
torch.save(centers, f"./model/kmeans/kmeans_centers {k}.pt")
joblib.dump(scaler, f"./model/kmeans/kmeans_scaler {k}.joblib")
