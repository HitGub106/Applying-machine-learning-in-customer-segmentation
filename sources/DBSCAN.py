import joblib, numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

# Load data đã lưu
X = joblib.load("X_preprocessed.pkl")          
user_df = pd.read_parquet("user_df.parquet")   
'''
# Vẽ K-DISTANCE PLOT để tìm eps
# ------------------------------------------------------------
k = 10
X_dense = X.toarray() if hasattr(X, "toarray") else X  # convert sparse nếu cần

neighbors = NearestNeighbors(n_neighbors=k)
neighbors_fit = neighbors.fit(X_dense)
distances, _ = neighbors_fit.kneighbors(X_dense)
k_distances = np.sort(distances[:, k-1])

plt.figure(figsize=(6,4))
plt.plot(k_distances)
plt.xlabel("Các điểm (đã sắp xếp)")
plt.ylabel(f"Khoảng cách đến hàng xóm thứ {k}")
plt.title("Biểu đồ K-distance ")
plt.grid(True)
plt.tight_layout()
plt.show()
'''
# Dùng DBSCAN  
eps_value   = 2.5
min_samples = 10

dbscan = DBSCAN(eps=eps_value, min_samples=min_samples, n_jobs=-1)
labels = dbscan.fit_predict(X)   

user_df["dbscan_cluster"] = labels

# Hiển thị kết quả
num_cols = ["total_spent", "avg_spent", "total_orders",
            "total_items", "frequency", "recency", "age"]

cluster_sizes = user_df["dbscan_cluster"].value_counts().sort_index()
cluster_summary = (
    user_df.groupby("dbscan_cluster")[num_cols]
           .mean().round(2)
           .assign(n_customers=cluster_sizes)
)
print("BẢNG CHI TIẾT CỤM (DBSCAN)")
print(cluster_summary)

# TRỰC QUAN HÓA – PCA 2D
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X.toarray() if hasattr(X, "toarray") else X)

plt.figure(figsize=(10,6))
# cho nhiễu màu đen
noise = labels == -1
plt.scatter(X_pca[noise,0], X_pca[noise,1], c='black', s=10, label='Noise')

# vẽ từng cụm
unique_clusters = sorted([c for c in np.unique(labels) if c != -1])
palette = sns.color_palette("tab10", len(unique_clusters))
for c, color in zip(unique_clusters, palette):
    mask = labels == c
    plt.scatter(X_pca[mask,0], X_pca[mask,1],
                s=10, color=color, label=f"Cluster {c}")

plt.title("Phân cụm khách hàng (DBSCAN + PCA 2D)")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.legend(markerscale=3, bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

# Vẽ heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(cluster_summary, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("Trung bình các đặc trưng theo từng cụm khách hàng")
plt.xlabel("Đặc trưng")
plt.ylabel("Cụm")
plt.tight_layout()
plt.show()

#lưu kết quả ra file
user_df.to_csv("user_clusters_dbscan.csv", index=False)
