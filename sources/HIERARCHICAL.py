import pandas as pd
import numpy as np
import joblib
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

# Load data đã lưu
X_full = joblib.load("X_preprocessed.pkl")
user_df = pd.read_parquet("user_df.parquet")

# Lấy mẫu 30000 khách để làm dendrogram
sample_size = 30000
rng = np.random.default_rng(42)
idx = rng.choice(X_full.shape[0], size=sample_size, replace=False)
X_sample = X_full[idx].toarray() if hasattr(X_full, "toarray") else X_full[idx]
user_sample = user_df.iloc[idx].copy()

# Tính liên kết bằng phương pháp ward
linked = linkage(X_sample, method='ward')

#Vẽ dendrogram
plt.figure(figsize=(12, 6))
dendrogram(linked, truncate_mode='lastp', p=30,
           leaf_rotation=90., leaf_font_size=10., show_contracted=True)
plt.title('Biểu đồ cây Hierarchical Clustering')
plt.xlabel('Chỉ số mẫu cụm')
plt.ylabel('Khoảng cách')
plt.tight_layout()
plt.show()

# Cắt cụm từ dendrogram, chọn 6 cụm
n_clusters = 6
user_sample['cluster_hier'] = fcluster(linked, n_clusters, criterion='maxclust')

# Hiển thị kết quả
summary = user_sample.groupby('cluster_hier')[
    ['total_spent', 'avg_spent', 'total_orders', 'total_items', 'frequency', 'recency', 'age']
].mean()
summary['n_customers'] = user_sample.groupby('cluster_hier').size()

print("BẢNG CHI TIẾT CỤM ")
print(summary)

labels_sample = user_sample['cluster_hier'].values   # cụm 1 đến 6
# Tính centroid (trung bình) của 6 cụm trong không gian đặc trưng
k = 6
centroids = []
for c in range(1, k+1):
    rows = X_sample[labels_sample == c]
    centroids.append(rows.mean(axis=0))
centroids = np.vstack(centroids)        

# Gán nhãn cho toàn bộ điểm bằng “centroid gần nhất”
dist = pairwise_distances(
    X_full, centroids, metric="euclidean", n_jobs=-1
) 
labels_full = dist.argmin(axis=1) + 1     

# 4) Lưu & phân tích
user_df["cluster_hier"] = labels_full
user_df.to_csv("user_clusters_hier_full.csv", index=False)

# Bảng chi tiết sau khi gán cụm tất cả khách 
num_cols = ["total_spent","avg_spent","total_orders",
            "total_items","frequency","recency","age"]
summary = (
    user_df.groupby("cluster_hier")[num_cols]
           .mean().round(2)
           .assign(n_customers=user_df.groupby("cluster_hier").size())
)
print("BẢNG CÁC CỤM HIERARCHICAL chi tiết")
print(summary)
from sklearn.decomposition import PCA
import seaborn as sns

# Giảm chiều dữ liệu toàn bộ xuống 2D để vẽ
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_full.toarray() if hasattr(X_full, "toarray") else X_full)

# Gán nhãn vào DataFrame
user_df['cluster_hier'] = labels_full  # cụm được gán từ centroid gần nhất

# Vẽ scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_reduced[:, 0], y=X_reduced[:, 1], hue=user_df['cluster_hier'], palette='viridis', s=10, alpha=0.6)
plt.title("Phân cụm khách hàng Hierarchical Clustering ")
plt.xlabel("Thành phần 1")
plt.ylabel("Thành phần 2")
plt.legend(title="Cụm")
plt.tight_layout()
plt.show()

# Vẽ heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(summary, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("Trung bình các đặc trưng theo từng cụm khách hàng")
plt.xlabel("Đặc trưng")
plt.ylabel("Cụm")
plt.tight_layout()
plt.show()