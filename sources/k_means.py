import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
# Ma trận đặc trưng (đã StandardScaler + One-HotEncoder)
X_pre = joblib.load("X_preprocessed.pkl")   

#Áp dụng K-Means
k = 6          # K đã chọn ở trên
model = KMeans(n_clusters=k, random_state=42, n_init="auto")
labels = model.fit_predict(X_pre)

#Ghép các user gốc với các cụm 
user_df = pd.read_parquet("user_df.parquet")    # bảng đặc trưng gốc (chưa chuẩn hóa)
user_df["cluster"] = labels
'''
# 4) PCA 2 CHIỀU ĐỂ TRỰC QUAN HÓA
# ------------------------------------------------------------
pca = PCA(n_components=2, random_state=42)
X_reduced = pca.fit_transform(X_pre.toarray() if hasattr(X_pre, "toarray") else X_pre)

plt.figure(figsize=(6,5))
plt.scatter(X_reduced[:,0], X_reduced[:,1], c=labels, cmap="viridis", s=10, alpha=0.7)
plt.title("Phân cụm khách hàng (PCA 2D)")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.tight_layout()
plt.show()
'''
# ------------------------------------------------------------
# 5) BẢNG CHI TIẾT CỤM (TRUNG BÌNH & SỐ LƯỢNG)
# ------------------------------------------------------------
feature_cols = [
    "total_spent", "avg_spent", "total_orders", "total_items",
    "frequency", "recency", "age"
]

cluster_mean = user_df.groupby("cluster")[feature_cols].mean().round(2)
cluster_count = user_df["cluster"].value_counts().sort_index()

# Thêm cột số lượng vào bảng trung bình
cluster_summary = cluster_mean.copy()
cluster_summary["n_customers"] = cluster_count
print("\n BẢNG CHI TIẾT CÁC CỤM ")
print(cluster_summary)

# ------------------------------------------------------------
# 6) HEATMAP trực quan trung bình đặc trưng
# ------------------------------------------------------------
plt.figure(figsize=(12,6))
sns.heatmap(cluster_mean, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Trung bình các đặc trưng theo cụm")
plt.xlabel("Đặc trưng")
plt.ylabel("Cụm")
plt.tight_layout()
plt.show()

# Lưu ra CSV để kiểm tra kết quả
user_df.to_csv("user_clusters_k.csv".format(k), index=False)
