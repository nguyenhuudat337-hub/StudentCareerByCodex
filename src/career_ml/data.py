"""Nạp, kiểm tra chất lượng, chống leakage và chuẩn bị dữ liệu."""
from __future__ import annotations
import io
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from .config import DATA_PATH, EXCLUDE_COLUMNS, ID_COL, TARGET, SEED, TEST_SIZE


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise ValueError(f"Không tìm thấy nhãn '{TARGET}' trong CSV")
    if df[TARGET].isna().any():
        raise ValueError("Nhãn có NaN; cần kiểm tra nguồn dữ liệu trước khi huấn luyện")
    return df


def audit_data(df):
    """Chỉ quan sát/ghi nhận, không tự động điền/xóa dữ liệu."""
    buff = io.StringIO()
    df.info(buf=buff)
    missing = df.isnull().sum()  # bắt buộc dù kết quả bằng 0
    duplicates_all = int(df.duplicated().sum())  # bắt buộc dù bằng 0
    without_id = df.drop(columns=[ID_COL], errors="ignore")
    duplicates_without_id = int(without_id.duplicated().sum())
    num = df.select_dtypes(include="number")
    numeric_checks = {}
    for column in num.columns:
        vals = num[column].to_numpy(dtype=float)
        valid = vals[np.isfinite(vals)]
        q1, q3 = np.percentile(valid, [25, 75]) if valid.size else (np.nan, np.nan)
        iqr = q3 - q1
        outliers = int(((valid < q1-1.5*iqr) | (valid > q3+1.5*iqr)).sum()) if valid.size else 0
        numeric_checks[column] = {
            "iqr_outliers_flagged": outliers,
            "min": float(valid.min()) if valid.size else None,
            "max": float(valid.max()) if valid.size else None,
            "infinite": int(np.isinf(vals).sum()),
        }
    known_bounds = {"age": (16, 100), "cgpa": (0, 10), "attendance_pct": (0, 100)}
    bounds_flags = {}
    for name, (lower, upper) in known_bounds.items():
        if name in df.columns:
            vals = pd.to_numeric(df[name], errors="coerce")
            bounds_flags[name] = {"suggested_range": [lower, upper],
                                  "outside_range": int(((vals < lower) | (vals > upper)).sum())}
    return {
        "bounds_heuristic_checks": bounds_flags,
        "rows": int(len(df)), "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "info": buff.getvalue(),
        "head": df.head(5).replace({np.nan: None}).to_dict(orient="records"),
        "missing_by_column": missing.astype(int).to_dict(),
        "missing_total": int(missing.sum()),
        "duplicate_rows": duplicates_all,
        "duplicate_rows_without_student_id": duplicates_without_id,
        "target_counts": df[TARGET].value_counts().astype(int).to_dict(),
        "numeric_ranges": numeric_checks,
    }


def prepare_xy(df, excluded=EXCLUDE_COLUMNS):
    if df[TARGET].isna().any():
        raise ValueError("Nhãn bị thiếu")
    X = df.drop(columns=[TARGET, *excluded], errors="ignore").copy()
    if X.empty or X.shape[1] == 0:
        raise ValueError("Không có đặc trưng đầu vào hợp lệ")
    y_encoder = LabelEncoder()
    y_encoded = y_encoder.fit_transform(df[TARGET].astype(str))
    return X, pd.Series(y_encoded, index=df.index, name=TARGET), y_encoder


def split_data(X, y):
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)


def input_schema(X_train):
    """Chỉ học các lựa chọn nhập liệu từ Train (không truy cập Test)."""
    result = {}
    for col in X_train.columns:
        s = X_train[col]
        if pd.api.types.is_numeric_dtype(s):
            nonnull = s.dropna().astype(float)
            result[col] = {
                "kind": "numeric", "min": float(nonnull.min()),
                "max": float(nonnull.max()), "median": float(nonnull.median()),
                "integer": bool(pd.api.types.is_integer_dtype(s)),
            }
        else:
            vals = sorted(map(str, s.dropna().unique().tolist()))
            result[col] = {"kind": "categorical", "choices": vals,
                           "default": str(s.mode().iloc[0]) if not s.mode().empty else vals[0]}
    return result


def json_dump(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
