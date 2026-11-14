"""Kiểm thử artifact mô hình đã huấn luyện, không thay đổi hoặc chấm điểm lại Test."""
import json
import joblib
import numpy as np
import pandas as pd
from career_ml.config import DATA_PATH, FS_NAMES, MODEL_NAMES, OUTPUT
from career_ml.data import load_data,prepare_xy,split_data


def main():
    root=OUTPUT
    df=load_data(DATA_PATH)
    runs=pd.read_csv(root/"tables"/"cv_experiment_results.csv",keep_default_na=False)
    expected_pairs = {(model, fs) for model in MODEL_NAMES for fs in FS_NAMES}
    actual_pairs = set(runs[["model", "fs"]].itertuples(index=False, name=None))
    assert len(runs) == len(expected_pairs), (
        f"Có {len(runs)} dòng kết quả, cần {len(expected_pairs)} cấu hình."
    )
    assert actual_pairs == expected_pairs, (
        f"Cấu hình thiếu: {expected_pairs - actual_pairs}. "
        f"Cấu hình thừa: {actual_pairs - expected_pairs}."
    )
    model_path=root/"models"/"career_classifier.joblib"
    bundle=joblib.load(model_path)  # artifact do dự án tự sinh
    model=bundle["model"]
    encoder=bundle["label_encoder"]
    assert len(encoder.classes_)==12
    raw_features=bundle["raw_features"]
    X,y,_=prepare_xy(df,excluded=bundle["excluded_columns"])
    assert X.columns.tolist()==raw_features
    X_train,_,_,_=split_data(X,y)
    one_row=X_train.iloc[[0]]
    probs=np.asarray(model.predict_proba(one_row))
    assert probs.shape==(1,12) and np.allclose(probs.sum(axis=1),1)
    assert np.array_equal(np.asarray(model.classes_),np.arange(12))
    classes=encoder.inverse_transform(np.asarray(model.classes_,dtype=int))
    assert len(set(classes))==12
    assert all(col in bundle["input_schema"] for col in raw_features)
    required=["calibration_12_classes.png","confusion_matrix.png","shap_summary.png","shap_waterfall.png"]
    assert all((root/"figures"/p).exists() for p in required)
    assert (root/"reports"/"bao_cao_thuc_nghiem.md").exists()
    metrics=json.loads((root/"tables"/"test_metrics.json").read_text())
    print("Artifact OK",len(runs),"configurations; labels",len(classes),
          "prediction",classes[probs.argmax()],"probability_sum",float(probs.sum()))
    print("Stored Test metrics:",metrics)
if __name__=="__main__":main()
