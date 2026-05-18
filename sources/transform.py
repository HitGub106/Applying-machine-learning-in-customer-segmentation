import joblib
import pandas as pd

# Load dữ liệu
X_preprocessed = joblib.load("X_preprocessed.pkl")
preprocessor = joblib.load("preprocessor.pkl")
user_df = pd.read_parquet("user_df.parquet")

# Lấy tất cả tên cột feature đã tiền xử lý (bao gồm numerical + cat)
all_feature_names = preprocessor.get_feature_names_out()
print("Tổng số tên cột feature:", len(all_feature_names))
print("X_preprocessed shape:", X_preprocessed[:10].shape)

# Chuyển 10 dòng đầu sang DataFrame (nếu là sparse matrix)
if hasattr(X_preprocessed, "toarray"):
    data_array = X_preprocessed[:10].toarray()
else:
    data_array = X_preprocessed[:10]

df_preprocessed = pd.DataFrame(data_array, columns=all_feature_names)

# Ghép với thông tin USERID gốc tương ứng
df_display = pd.concat([user_df[['USERID']].head(10).reset_index(drop=True), df_preprocessed], axis=1)
df_display.to_csv("user_preprocessed_sample.csv", index=False, sep=",", decimal=".")
print(df_display)
