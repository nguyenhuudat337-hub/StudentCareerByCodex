"""Kiểm tra preprocessing và xác suất trên tập nhỏ — KHÔNG có kết quả nghiên cứu."""
import numpy as np
from career_ml.data import load_data,prepare_xy,split_data,audit_data
from career_ml.modeling import make_pipeline,feature_names_after_fs

def main():
    data=load_data(); audit=audit_data(data)
    print("Rows:",len(data),"Classes:",len(audit["target_counts"]))
    print("Missing:",audit["missing_total"],"Duplicates:",audit["duplicate_rows"],
          "Duplicates w/o ID:",audit["duplicate_rows_without_student_id"])
    X,y,enc=prepare_xy(data)
    train,test,yt,ys=split_data(X,y)
    # Train nhỏ chỉ nhằm xác minh pipeline/label/proba.
    sample=train.sample(n=min(240,len(train)),random_state=42)
    target=yt.loc[sample.index]
    for model_name in ["Logistic Regression","Random Forest","XGBoost"]:
        for fs in ["None","Filter","Wrapper","Embedded"]:
            pipe=make_pipeline(sample,model_name,fs)
            pipe.fit(sample,target)
            prob=pipe.predict_proba(test.iloc[:4])
            assert np.isfinite(prob).all() and np.allclose(prob.sum(axis=1),1)
            assert prob.shape[1]==len(enc.classes_)
            print(model_name,fs,"OK",len(feature_names_after_fs(pipe)),"features")
    print("Smoke test thành công (không dùng score này trong báo cáo)")
if __name__=="__main__":main()
