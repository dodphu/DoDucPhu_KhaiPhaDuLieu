import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_data(file_path="data/adult.csv"):
    # Đọc dữ liệu từ tệp CSV và chỉ định tên cột
    column_names = [
        "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
        "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
        "hours-per-week", "native-country", "income"
    ]
    data = pd.read_csv(file_path, header=None, names=column_names, na_values=" ?", skipinitialspace=True)

    # Xử lý dữ liệu còn thiếu
    data.fillna("Unknown", inplace=True)  # Thay thế dấu "?" bằng giá trị "Unknown" hoặc giá trị khác

    # Chuyển đổi biến phân loại thành dạng số
    label_encoder = LabelEncoder()
    categorical_columns = data.select_dtypes(include=["object"]).columns
    for col in categorical_columns:
        data[col] = label_encoder.fit_transform(data[col])

    return data
