from sqlalchemy import create_engine
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from datetime import datetime

engine = create_engine(
    "mssql+pyodbc://DESKTOP-S96H7IF/SALES50M"
    "?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
)

query = """
SELECT
    USERID,
    MAX(DATE_)                                       AS last_order_date,
    DATEDIFF(DAY, MAX(DATE_), GETDATE())             AS recency,
    COUNT(DISTINCT ORDERID)                          AS total_orders,
    COUNT(*)                                         AS frequency,
    SUM(CAST(REPLACE(TOTALPRICE, ',', '.') AS FLOAT)) AS total_spent,
    AVG(CAST(REPLACE(TOTALPRICE, ',', '.') AS FLOAT)) AS avg_spent,
    SUM(AMOUNT)                                      AS total_items,
    USERGENDER,
    REGION,
    CITY,
    MIN(USERBIRTHDATE)                               AS USERBIRTHDATE
FROM SALES
GROUP BY USERID, USERGENDER, REGION, CITY;
"""

# ① Đọc dữ liệu đã tổng hợp 
user_df = pd.read_sql(query, engine)

# ② Tính thêm AGE
latest_date = datetime.today()
user_df["USERBIRTHDATE"] = pd.to_datetime(user_df["USERBIRTHDATE"], errors="coerce")
user_df["age"] = (latest_date - user_df["USERBIRTHDATE"]).dt.days // 365

# ③ Chọn cột đưa vào mô hình
features = [
    "total_spent", "avg_spent", "total_orders", "total_items",
    "frequency", "recency", "age",
    "USERGENDER", "REGION", "CITY"
]
categorical = ["USERGENDER", "REGION", "CITY"]
numerical   = [c for c in features if c not in categorical]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
])

X_preprocessed = preprocessor.fit_transform(user_df[features])

# ④ Lưu kết quả & pipeline
joblib.dump(X_preprocessed, "X_preprocessed.pkl")
joblib.dump(preprocessor,    "preprocessor.pkl")
user_df.to_parquet("user_df.parquet", index=False)
