"""EDA và lưu hình; không sử dụng Test để chọn đặc trưng/mô hình."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from .config import OUTPUT, TARGET
from .data import load_data, audit_data, json_dump


def run_eda(data_path=None, output=OUTPUT):
    df = load_data(data_path) if data_path else load_data()
    fig_dir = output / "figures"
    table_dir = output / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    audit = audit_data(df)
    json_dump(audit, table_dir / "data_quality_audit.json")
    df.describe(include="all").transpose().to_csv(table_dir / "data_description.csv")
    df.dtypes.astype(str).to_csv(table_dir / "column_dtypes.csv",header=["dtype"])
    vc = df[TARGET].value_counts()
    vc.rename("count").to_csv(table_dir / "class_distribution.csv")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=vc.index,y=vc.values,ax=ax,color="#347C8C")
    ax.set(title="Phân bố nhãn recommended_career",xlabel="Nhóm nghề",ylabel="Số sinh viên")
    ax.tick_params(axis="x",rotation=60)
    plt.setp(ax.get_xticklabels(),ha="right")
    fig.tight_layout(); fig.savefig(fig_dir / "eda_label_distribution.png",dpi=150); plt.close(fig)

    numeric = df.select_dtypes(include=np.number).columns.tolist()
    preferred = [c for c in ["cgpa","technical_skill_score","career_interest_technology"] if c in numeric]
    if preferred:
        fig, axes = plt.subplots(1,len(preferred),figsize=(5*len(preferred),4))
        for ax, col in zip(np.atleast_1d(axes),preferred):
            sns.histplot(df[col],bins=30,ax=ax,kde=True,color="#347C8C")
            ax.set_title(col)
        fig.tight_layout(); fig.savefig(fig_dir / "eda_histograms.png",dpi=150); plt.close(fig)
        fig, axes = plt.subplots(len(preferred),1,figsize=(12,4*len(preferred)))
        for ax, col in zip(np.atleast_1d(axes),preferred):
            sns.boxplot(data=df,x=TARGET,y=col,ax=ax,color="#5F9EA0")
            ax.set_title(f"{col} theo nhóm nghề")
            ax.tick_params(axis="x",rotation=55)
            plt.setp(ax.get_xticklabels(),ha="right")
        fig.tight_layout(); fig.savefig(fig_dir / "eda_boxplots.png",dpi=150); plt.close(fig)
    if "gender" in df.columns:
        fig,ax = plt.subplots(figsize=(7,4))
        sns.countplot(data=df,x="gender",ax=ax,color="#347C8C")
        ax.set_title("Phân bố giới tính")
        fig.tight_layout(); fig.savefig(fig_dir / "eda_gender.png",dpi=150); plt.close(fig)
    corr = df[numeric].corr(numeric_only=True)
    corr.to_csv(table_dir / "numeric_correlations.csv")
    fig,ax = plt.subplots(figsize=(17,14))
    sns.heatmap(corr, cmap="coolwarm",center=0,vmin=-1,vmax=1,ax=ax)
    ax.set_title("Ma trận tương quan các cột số (để KHÁM PHÁ, không tự chứng minh leakage)")
    fig.tight_layout(); fig.savefig(fig_dir / "eda_correlation_heatmap.png",dpi=120); plt.close(fig)
    print(f"EDA xong: {len(df):,} dòng, {df.shape[1]} cột, {df[TARGET].nunique()} lớp")
    print(f"Missing={audit['missing_total']}; trùng dòng={audit['duplicate_rows']}; "
          f"trùng bỏ ID={audit['duplicate_rows_without_student_id']}")
    return audit
