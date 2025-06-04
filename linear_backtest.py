import pandas as pd
import joblib

# 1. 載入模型與標準化器
model = joblib.load('./model/linear_regression.joblib')
scaler = joblib.load('./model/scaler_fresh.joblib')

# 2. 載入資料
df = pd.read_csv('./data/data.csv')
df['date'] = pd.to_datetime(df['date'])

# 3. 建立衍生特徵欄位
df['mid_price'] = (df['highest_price'] - df['lowest_price']) / 2
df['open_close_diff'] = df['opening_price'] - df['closing_price']

# 4. 定義特徵欄位
feature_cols = [
    'opening_price', 'closing_price', 'highest_price', 'lowest_price',
    'mid_price', 'open_close_diff'
]

# 5. 指定推薦日期
target_date = '2025-05-29'  # 可修改為你要的日期
target_date = pd.to_datetime(target_date)

# 6. 篩選資料並預測
df_target = df[df['date'] == target_date].dropna(subset=feature_cols)

if df_target.empty:
    print(f"沒有 {target_date.date()} 的可用資料")
else:
    X_target = scaler.transform(df_target[feature_cols])
    df_target['Predicted_KD_K'] = model.predict(X_target)

    # 7. 推薦前 10 名
    top10 = df_target.sort_values(by='Predicted_KD_K', ascending=False).head(10)

    # 8. 顯示結果
    print(f"\n推薦日期:{target_date.date()}，預測未來 5 日 KD_K 最高的前 10 檔股票：\n")
    print(top10[['stock_code', 'date', 'Predicted_KD_K']])
