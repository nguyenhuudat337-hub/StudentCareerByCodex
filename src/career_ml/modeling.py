"""Tạo 13 cấu hình có preprocessing và FS nằm BÊN TRONG CV."""
from __future__ import annotations
from functools import partial
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from .config import SEED, FOLDS


def make_preprocessor(X_train):
    num = X_train.select_dtypes(include=np.number).columns.tolist()
    cat = X_train.select_dtypes(exclude=np.number).columns.tolist()
    if X_train.isna().any().any():
        raise ValueError("Có missing trong X: cần xác định cách xử lý có căn cứ trước khi fit")
    return ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
    ], remainder="drop", sparse_threshold=0, verbose_feature_names_out=True)


def make_cv():
    return StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)


def make_estimator(name):
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=600, solver="lbfgs", random_state=SEED)
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=SEED)
    if name == "XGBoost":
        return XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                             subsample=1.0, colsample_bytree=1.0,
                             objective="multi:softprob", eval_metric="mlogloss",
                             tree_method="hist", n_jobs=1, random_state=SEED)
    if name == "Dummy":
        return DummyClassifier(strategy="stratified", random_state=SEED)
    raise KeyError(name)


def mixed_mutual_info(X, y, *, n_numeric):
    """Đánh dấu các cột One-Hot là rời rạc khi tính MI."""
    X = np.asarray(X)

    if X.ndim != 2 or not 0 <= n_numeric <= X.shape[1]:
        raise ValueError("Số cột số không khớp dữ liệu sau tiền xử lý.")

    # ColumnTransformer xuất nhánh num trước, nhánh cat sau.
    discrete_mask = np.zeros(X.shape[1], dtype=bool)
    discrete_mask[n_numeric:] = True

    return mutual_info_classif(
        X,
        y,
        discrete_features=discrete_mask,
        random_state=SEED,
    )


def make_selector(fs, n_numeric=None):
    if fs == "None":
        return "passthrough"
    if fs == "Filter":
        if n_numeric is None:
            raise ValueError("Filter cần biết số cột của nhánh num.")

        return SelectKBest(
            score_func=partial(
                mixed_mutual_info,
                n_numeric=n_numeric,
            ),
            k=16,
        )
    if fs == "Wrapper":
        return RFE(estimator=LogisticRegression(max_iter=400, solver="lbfgs", random_state=SEED),
                   n_features_to_select=16, step=0.3)
    if fs == "Embedded":
        return SelectFromModel(estimator=RandomForestClassifier(
            n_estimators=60, max_depth=12, n_jobs=1, random_state=SEED),
            threshold="median")
    raise KeyError(fs)


def make_pipeline(X_train, name, fs):
    n_numeric = len(
        X_train.select_dtypes(include=np.number).columns
    )

    return Pipeline([
        ("preprocess", make_preprocessor(X_train)),
        ("fs", make_selector(fs, n_numeric=n_numeric)),
        ("clf", make_estimator(name)),
    ])


def param_grid(name, fs, preset="quick"):
    """Grid không tham chiếu Test; mọi FS có 2 k/threshold khi nâng cấp preset."""
    if preset == "quick":
        base = {
            "Logistic Regression": {"clf__C": [0.5, 1.0]},
            "Random Forest": {"clf__n_estimators": [60, 100]},
            "XGBoost": {"clf__learning_rate": [0.07, 0.12]},
        }
        options = {"Filter": [12], "Wrapper": [12], "Embedded": ["median"]}
    elif preset == "standard":
        base = {
            "Logistic Regression": {"clf__C": [0.3, 1, 3]},
            "Random Forest": {"clf__n_estimators": [100, 160], "clf__max_depth": [None, 12]},
            "XGBoost": {"clf__learning_rate": [0.05, 0.1], "clf__max_depth": [3, 5]},
        }
        options = {"Filter": [12, 24], "Wrapper": [12, 24], "Embedded": ["median", "mean"]}
    elif preset == "full":
        base = {
            "Logistic Regression": {"clf__C": [0.1, 1, 10], "clf__solver": ["lbfgs"], "clf__penalty": ["l2"]},
            "Random Forest": {"clf__n_estimators": [100, 200], "clf__max_depth": [None, 12],
                              "clf__min_samples_split": [2, 5]},
            "XGBoost": {"clf__n_estimators": [100, 200], "clf__max_depth": [3, 5],
                        "clf__learning_rate": [0.05, 0.1], "clf__subsample": [0.8, 1.0]},
        }
        options = {"Filter": [12, 24, 32], "Wrapper": [12, 24, 32], "Embedded": ["median", "mean"]}
    else:
        raise ValueError(preset)
    grid = dict(base[name])
    if fs == "Filter":
        grid["fs__k"] = options[fs]
    elif fs == "Wrapper":
        grid["fs__n_features_to_select"] = options[fs]
    elif fs == "Embedded":
        grid["fs__threshold"] = options[fs]
    return grid


def feature_names_after_fs(model):
    all_names = model.named_steps["preprocess"].get_feature_names_out()
    selector = model.named_steps["fs"]
    mask = np.ones(len(all_names), dtype=bool) if selector == "passthrough" else selector.get_support()
    return np.asarray(all_names)[mask].tolist()
