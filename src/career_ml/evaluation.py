"""Đánh giá Test chỉ sau khi chốt cấu hình từ CV trên Train."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             log_loss, confusion_matrix, classification_report)
from sklearn.calibration import calibration_curve
from .data import json_dump


def validate_proba(model, X, n_classes):
    proba = np.asarray(model.predict_proba(X))
    labels = np.asarray(model.classes_, dtype=int)
    if proba.ndim != 2 or proba.shape != (len(X), n_classes):
        raise AssertionError(f"Sai kích thước predict_proba: {proba.shape}")
    if not np.isfinite(proba).all() or (proba < -1e-10).any() or (proba > 1+1e-10).any():
        raise AssertionError("Có xác suất không hợp lệ")
    if not np.allclose(proba.sum(axis=1), 1, atol=1e-5):
        raise AssertionError("Tổng xác suất theo dòng khác 1")
    if not np.array_equal(labels, np.arange(n_classes)):
        raise AssertionError("Thứ tự classes_ không khớp LabelEncoder 0..K-1")
    return proba


def summary_metrics(y_true, y_pred, proba=None, labels=None):
    d = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if proba is not None:
        d["log_loss_multiclass"] = float(log_loss(y_true, proba, labels=labels))
    return d


def one_vs_rest_calibration(y_true, proba, class_names, output_dir, bins=10):
    """Reliability cho từng lớp; ECE là tóm tắt mô tả, không phải chuẩn kiểm chứng tuyệt đối."""
    rows = []
    fig, ax = plt.subplots(3, 4, figsize=(18, 13))
    axes = ax.flat
    for c, label in enumerate(class_names):
        binary = (np.asarray(y_true) == c).astype(int)
        frac, mean = calibration_curve(binary, proba[:, c], n_bins=bins, strategy="uniform")
        a = axes[c]
        a.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
        a.plot(mean, frac, "o-", markersize=3)
        a.set_title(str(label), fontsize=10)
        a.set(xlim=(0,1), ylim=(0,1), xlabel="Xác suất dự đoán", ylabel="Tần suất thực tế")
        a.grid(alpha=0.2)
        ids = np.minimum((proba[:, c] * bins).astype(int), bins-1)
        ece = 0.0
        for i in range(bins):
            m = ids == i
            if not m.any():
                continue
            confidence = float(proba[m, c].mean())
            observed = float(binary[m].mean())
            ece += float(m.mean()) * abs(confidence - observed)
            rows.append({"class":label, "bin":i, "count":int(m.sum()),
                         "mean_probability":confidence, "observed_frequency":observed})
        a.text(0.04,0.94,f"ECE={ece:.3f}",transform=a.transAxes,fontsize=9,va="top")
    for a in list(axes)[len(class_names):]:
        a.set_visible(False)
    fig.suptitle("Calibration one-vs-rest theo từng nhóm nghề (chỉ phân tích Test)")
    fig.tight_layout(rect=(0,0,1,0.98))
    fig.savefig(output_dir / "calibration_12_classes.png", dpi=150)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(output_dir.parent / "tables" / "calibration_bins.csv",index=False)


def test_evaluation(model, X_test, y_test, label_encoder, out):
    names = label_encoder.classes_.tolist()
    preds = np.asarray(model.predict(X_test), dtype=int)
    proba = validate_proba(model, X_test, len(names))
    metrics = summary_metrics(y_test, preds, proba, labels=np.arange(len(names)))
    tables = out / "tables"
    figures = out / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    json_dump(metrics, tables / "test_metrics.json")
    report = classification_report(y_test, preds, labels=np.arange(len(names)),
                                   target_names=names, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(tables / "classification_report.csv")
    cm = confusion_matrix(y_test,preds,labels=np.arange(len(names)))
    pd.DataFrame(cm,index=names,columns=names).to_csv(tables / "confusion_matrix.csv")
    plt.figure(figsize=(13, 11))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",xticklabels=names,yticklabels=names)
    plt.title("Confusion matrix — mô hình được chọn từ Train CV")
    plt.ylabel("Nhãn thực tế"); plt.xlabel("Nhãn dự đoán")
    plt.xticks(rotation=65,ha="right"); plt.yticks(rotation=0)
    plt.tight_layout(); plt.savefig(figures / "confusion_matrix.png",dpi=150); plt.close()
    one_vs_rest_calibration(y_test,proba,names,figures)
    pd.DataFrame({"true":label_encoder.inverse_transform(np.asarray(y_test)),
                  "predicted":label_encoder.inverse_transform(preds),
                  "predicted_max_probability":proba.max(axis=1)})\
      .to_csv(tables / "test_predictions.csv",index=False)
    return metrics


def plot_cv_comparison(results, out):
    valid = pd.DataFrame(results)
    if valid.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=valid, x="model", y="best_cv_f1_macro", hue="fs", ax=ax,
                order=["Logistic Regression","Random Forest","XGBoost"],
                hue_order=["None","Filter","Wrapper","Embedded"])
    ax.set_title("So sánh Macro F1 bằng 5-fold CV trên Train — KHÔNG PHẢI Test")
    ax.set(xlabel="Thuật toán",ylabel="Macro F1 (CV)")
    plt.tight_layout(); (out / "figures").mkdir(parents=True,exist_ok=True)
    plt.savefig(out / "figures" / "cv_model_comparison.png",dpi=150); plt.close(fig)
