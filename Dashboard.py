import pandas as pd
from sklearn.cluster import KMeans
import joblib

# Đọc dữ liệu
sales_df = pd.read_csv('./vgsales.csv', encoding='unicode_escape')

# Tiền xử lý dữ liệu (ví dụ: chọn các cột số, chuẩn hóa, v.v.)
# Ở đây chỉ là ví dụ, bạn cần xử lý giống như trong notebook của mình
X = sales_df.select_dtypes(include='number').dropna()

# Huấn luyện mô hình
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X)

# Lưu mô hình
joblib.dump(kmeans, 'kmeans_model.pkl')