"""Feature Importance, Permutation Importance và SHAP global/local."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from sklearn.inspection import permutation_importance
from .config import DATA_PATH,OUTPUT,SEED
from .data import load_data,prepare_xy,split_data
from .modeling import feature_names_after_fs


def explain(output=OUTPUT, data_path=DATA_PATH, sample_size=80):
    import shap
    selection_path=output / "models" / "uncalibrated_selected_pipeline.joblib"
    if not selection_path.exists():
        raise FileNotFoundError("Cần chạy train.py hoàn chỉnh trước explain.py")
    model=joblib.load(selection_path)  # chỉ tải tệp joblib do CHÍNH dự án này tạo
    bundle=joblib.load(output / "models" / "career_classifier.joblib")
    df=load_data(data_path)
    X,y,_=prepare_xy(df,excluded=bundle["excluded_columns"])
    X_train,X_test,y_train,y_test=split_data(X,y)
    X_test=X_test.sample(n=min(sample_size,len(X_test)),random_state=SEED)
    names=feature_names_after_fs(model)
    pre=model.named_steps["preprocess"]
    selector=model.named_steps["fs"]
    classifier=model.named_steps["clf"]
    Xt=pre.transform(X_test)
    if selector != "passthrough":
        Xt=selector.transform(Xt)
    Xt=np.asarray(Xt,dtype=float)
    target_index=int(model.predict(X_test.iloc[[0]])[0])
    fig_dir=output / "figures"; table_dir=output / "tables"
    fig_dir.mkdir(exist_ok=True,parents=True);table_dir.mkdir(exist_ok=True,parents=True)
    if hasattr(classifier,"feature_importances_"):
        imp=np.asarray(classifier.feature_importances_)
    elif hasattr(classifier,"coef_"):
        imp=np.mean(np.abs(classifier.coef_),axis=0)
    else:
        imp=None
    if imp is not None:
        table=pd.DataFrame({"feature":names,"importance":imp}).sort_values("importance",ascending=False)
        table.to_csv(table_dir / "feature_importance.csv",index=False)
        ax=table.head(20).iloc[::-1].plot.barh(x="feature",y="importance",figsize=(11,8),legend=False)
        ax.set_title("Độ quan trọng đặc trưng (của mô hình cuối)")
        plt.tight_layout();plt.savefig(fig_dir / "feature_importance.png",dpi=150);plt.close()
    # Permutation importance theo F1 Macro trên tập Test: dùng để GIẢI THÍCH, không chọn lại mô hình.
    prm=permutation_importance(model,X_test,y_test.loc[X_test.index],
                               n_repeats=3,random_state=SEED,scoring="f1_macro",n_jobs=1)
    ptab=pd.DataFrame({"raw_feature":X_test.columns,"importance_mean":prm.importances_mean,
                       "importance_std":prm.importances_std}).sort_values("importance_mean",ascending=False)
    ptab.to_csv(table_dir / "permutation_importance.csv",index=False)
    ax=ptab.head(20).iloc[::-1].plot.barh(x="raw_feature",y="importance_mean",figsize=(11,8),legend=False)
    ax.set_title("Permutation Importance trên Test (phân tích sau đánh giá)")
    plt.tight_layout();plt.savefig(fig_dir / "permutation_importance.png",dpi=150);plt.close()

    if hasattr(classifier,"coef_"):
        background=pre.transform(X_train.sample(n=min(80,len(X_train)),random_state=SEED))
        if selector != "passthrough":
            background=selector.transform(background)
        explainer=shap.LinearExplainer(classifier,np.asarray(background,dtype=float))
    else:
        explainer=shap.TreeExplainer(classifier)
    explanation = explainer(Xt) if hasattr(classifier,"coef_") else explainer(Xt,check_additivity=False)
    values=np.asarray(explanation.values)
    base=np.asarray(explanation.base_values)
    # SHAP cho đa lớp: (N,F,C) ở bản mới; một số phiên bản trả (N,F) cho một lớp.
    if values.ndim==3:
        class_values=values[:,:,target_index]
        base_local=base[0,target_index] if base.ndim==2 else base[target_index]
    elif values.ndim==2:
        class_values=values
        base_local=base[0] if base.ndim else float(base)
    else:
        raise ValueError(f"SHAP có shape không được hỗ trợ: {values.shape}")
    shap.summary_plot(class_values,Xt,feature_names=names,max_display=20,show=False)
    plt.title(f"SHAP global — nhóm: {bundle['label_encoder'].classes_[target_index]}")
    plt.tight_layout();plt.savefig(fig_dir / "shap_summary.png",dpi=150,bbox_inches="tight");plt.close()
    local=shap.Explanation(values=class_values[0],base_values=float(base_local),
                           data=Xt[0],feature_names=names)
    shap.plots.waterfall(local,max_display=18,show=False)
    plt.tight_layout();plt.savefig(fig_dir / "shap_waterfall.png",dpi=150,bbox_inches="tight");plt.close()
    meta={"explained_class":str(bundle['label_encoder'].classes_[target_index]),
          "shap_array_shape":list(values.shape),"n_samples_explained":int(len(Xt)),
          "shap_scale":"đầu ra thô của mô hình (logit/margin với nhiều tree models); không mặc nhiên là xác suất",
          "warning":"SHAP mô tả đóng góp theo mô hình, không chứng minh nguyên nhân hay tỷ lệ đỗ tuyển sinh."}
    (table_dir / "shap_metadata.json").write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding="utf-8")
    print("XAI hoàn tất:",json.dumps(meta,ensure_ascii=False))
    return meta
