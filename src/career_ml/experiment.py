"""Huấn luyện 1 Dummy + 12 cấu hình; Test chỉ gọi sau khi chọn Train CV."""
from __future__ import annotations
import json
import platform
import time
import warnings
from pathlib import Path
from uuid import uuid4
import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from .config import (DATA_PATH, OUTPUT, EXCLUDE_COLUMNS, MODEL_NAMES, FS_NAMES,
                     SEED, TEST_SIZE, FOLDS)
from .data import load_data, audit_data, prepare_xy, split_data, input_schema, json_dump
from .modeling import (make_cv, make_pipeline, param_grid, feature_names_after_fs)
from .evaluation import summary_metrics, test_evaluation, plot_cv_comparison

SCORING = {"accuracy":"accuracy", "precision_macro":make_scorer(precision_score,average="macro",zero_division=0),
           "recall_macro":make_scorer(recall_score,average="macro",zero_division=0),
            "recall_weighted": make_scorer(recall_score, average="weighted", zero_division=0),
           "f1_macro":make_scorer(f1_score,average="macro",zero_division=0),
           "f1_weighted":make_scorer(f1_score,average="weighted",zero_division=0)}


def compare_calibration_on_train(selected, X_train, y_train, output, delta=0.005):
    """Chỉ validation được rút từ Train; tuyệt đối không chọn calibration trên Test."""
    from sklearn.metrics import log_loss
    A, B, ya, yb = train_test_split(X_train,y_train,test_size=0.25,
                                     stratify=y_train,random_state=SEED+1)
    plain = clone(selected).fit(A,ya)
    before = float(log_loss(yb,plain.predict_proba(B),labels=selected.classes_))
    calibrated = CalibratedClassifierCV(estimator=clone(selected), method="sigmoid", cv=3)
    calibrated.fit(A,ya)
    after = float(log_loss(yb,calibrated.predict_proba(B),labels=selected.classes_))
    chosen = bool(after + delta < before)
    report = {"validation_from_train":True, "method":"sigmoid", "cv_inner":3,
              "log_loss_uncalibrated":before, "log_loss_calibrated":after,
              "improvement_required":delta,"use_calibrated":chosen}
    json_dump(report,output / "tables" / "calibration_train_validation.json")
    if chosen:
        # Hiệu chỉnh lại bằng toàn bộ Train; chưa nhìn Test.
        final = CalibratedClassifierCV(estimator=clone(selected),method="sigmoid",cv=3)
        final.fit(X_train,y_train)
        return final, report
    return selected, report


def _features_count(estimator):
    return len(feature_names_after_fs(estimator))


def run_experiment(data_path=DATA_PATH, output=OUTPUT, preset="quick", jobs=2,
                   limit=None, debug_rows=None, compare_calibration=False):
    """Mặc định đủ 13 cấu hình trên toàn bộ dữ liệu, 5-fold CV.

    debug_rows/limit chỉ kiểm thử và KHÔNG tạo model chính thức hoặc đánh giá Test.
    """
    data = load_data(data_path)
    run_id = uuid4().hex
    print(f"Mã lần huấn luyện: {run_id}", flush=True)
    data_audit = audit_data(data)
    if debug_rows is not None:
        if debug_rows < 120:
            raise ValueError("--debug-rows phải >= 120")
        data = data.groupby("recommended_career",group_keys=False).sample(
            n=max(10,debug_rows//data["recommended_career"].nunique()),random_state=SEED)
    dev = debug_rows is not None or limit is not None
    if dev:
        output = output / "debug"
    for folder in ["tables","models","figures","reports"]:
        (output / folder).mkdir(parents=True,exist_ok=True)
    json_dump(data_audit,output / "tables" / "data_quality_audit.json")
    X,y,encoder = prepare_xy(data)
    X_train,X_test,y_train,y_test = split_data(X,y)
    json_dump({"run_id": run_id,
               "dataset_rows":len(data),"n_train":len(X_train),"n_test":len(X_test),
               "class_labels":encoder.classes_.tolist(),"excluded_columns":EXCLUDE_COLUMNS,
               "feature_count_raw":len(X.columns),"feature_names":X.columns.tolist(),
               "seed":SEED,"test_size":TEST_SIZE,"folds":FOLDS,"preset":preset,
               "debug":dev,"versions":{"python":platform.python_version(),
                   "sklearn":sklearn.__version__,"xgboost":xgboost.__version__}},
              output / "tables" / "experiment_config.json")
    cv=make_cv()
    if not dev:
        # 3 baseline không FS, siêu tham số khởi tạo, cùng Train CV như các GridSearchCV.
        from .initial import run_initial_baselines
        run_initial_baselines(X_train,y_train,output=output,jobs=jobs)
    rows=[]
    print("Đang chạy 5-fold CV; Test được giữ riêng đến khi chọn mô hình.",flush=True)
    dummy=make_pipeline(X_train,"Dummy","None")
    t=time.perf_counter()
    baseline_cv=cross_validate(dummy,X_train,y_train,cv=cv,scoring=SCORING,n_jobs=jobs)
    baseline={"exp":0,"model":"Dummy","fs":"None", "n_candidates":0,
              "best_cv_f1_macro":float(baseline_cv["test_f1_macro"].mean()),
              "std_cv_f1_macro":float(baseline_cv["test_f1_macro"].std()),
              "best_cv_accuracy":float(baseline_cv["test_accuracy"].mean()),
              "best_cv_precision_macro":float(baseline_cv["test_precision_macro"].mean()),
              "best_cv_recall_macro":float(baseline_cv["test_recall_macro"].mean()),
            "best_cv_recall_weighted": float(baseline_cv["test_recall_weighted"].mean()),
              "best_cv_f1_weighted":float(baseline_cv["test_f1_weighted"].mean()),
              "seconds":round(time.perf_counter()-t,2),"num_selected_features":None,
              "best_params":"{}"}
    baseline["run_id"] = run_id
    rows.append(baseline)
    print(f"[00/12] Dummy CV Macro F1 = {baseline['best_cv_f1_macro']:.4f}",flush=True)
    selected=None
    bestscore=-float("inf")
    exp=0
    for model_name in MODEL_NAMES:
        for fs in FS_NAMES:
            exp+=1
            if limit is not None and exp>limit:
                break
            pipe=make_pipeline(X_train,model_name,fs)
            grid=param_grid(model_name,fs,preset)
            search=GridSearchCV(pipe,grid,cv=cv,scoring=SCORING,refit="f1_macro",
                                n_jobs=jobs,error_score="raise",return_train_score=False)
            tic=time.perf_counter()
            print(f"[{exp:02d}/12] {model_name} + {fs}: bắt đầu GridSearchCV ...",flush=True)
            search.fit(X_train,y_train)
            took=time.perf_counter()-tic
            ix=search.best_index_
            res=search.cv_results_
            row={"exp":exp,"model":model_name,"fs":fs,"n_candidates":len(res["params"]),
                 "best_cv_f1_macro":float(search.best_score_),
                 "std_cv_f1_macro":float(res["std_test_f1_macro"][ix]),
                 "best_cv_accuracy":float(res["mean_test_accuracy"][ix]),
                 "best_cv_precision_macro":float(res["mean_test_precision_macro"][ix]),
                 "best_cv_recall_macro":float(res["mean_test_recall_macro"][ix]),
                "best_cv_recall_weighted": float(res["mean_test_recall_weighted"][ix]),
                 "best_cv_f1_weighted":float(res["mean_test_f1_weighted"][ix]),
                 "seconds":round(took,2),
                 "num_selected_features":_features_count(search.best_estimator_),
                 "best_params":json.dumps(search.best_params_,ensure_ascii=False)}
            row["run_id"] = run_id
            rows.append(row)
            frame=pd.DataFrame(rows)
            frame.to_csv(output / "tables" / "cv_experiment_results.csv",index=False)
            # Toàn bộ kết quả từng tổ hợp grid cho từng phương án.
            pd.DataFrame(res).to_csv(output / "tables" / f"grid_{exp:02d}.csv",index=False)
            if search.best_score_ > bestscore:
                bestscore=float(search.best_score_)
                selected=search.best_estimator_
                best_config={"exp":exp,"model":model_name,"fs":fs,
                             "cv_f1_macro":bestscore,"best_params":search.best_params_}
            print(f"[{exp:02d}/12] CV F1={search.best_score_:.4f} | {took:.1f}s",flush=True)
        if limit is not None and exp>=limit:
            break
    plot_cv_comparison(rows,output)
    if dev:
        print("Đã chạy DEBUG: không tạo model dùng cho demo; không đánh giá Test.",flush=True)
        return rows
    if len(rows)!=13:
        raise AssertionError("Phải có đúng 13 cấu hình trước khi đánh giá Test")
    # Chỉ chọn từ CV Train, chưa hề quan sát nhãn Test khi chốt.
    json_dump(best_config,output / "tables" / "selected_model_train_cv.json")
    calibrated_report=None
    final_model=selected
    if compare_calibration:
        final_model,calibrated_report=compare_calibration_on_train(
            selected,X_train,y_train,output)
    # Đánh giá Test DUY NHẤT ở đây, sau khi chọn mô hình và quyết định calibration.
    metrics=test_evaluation(final_model,X_test,y_test,encoder,output)
    selected.training_run_id_ = run_id
    final_model.training_run_id_ = run_id
    bundle={"run_id": run_id,
            "dummy_cv": baseline.copy(),
            "model":final_model,"label_encoder":encoder,
            "input_schema":input_schema(X_train), "raw_features":X_train.columns.tolist(),
            "excluded_columns":EXCLUDE_COLUMNS,"selected_cv":best_config,
            "test_metrics":metrics,"calibration_train_selection":calibrated_report,
            "trained_on_train_only": True,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "dataset_rows": len(data),
            "preset": preset}
    joblib.dump(bundle,output / "models" / "career_classifier.joblib",compress=3)
    # SHAP/intrinsic importance sử dụng mô hình gốc đã fit trước calibration, nếu chọn hiệu chỉnh.
    # Mô hình chính để giải thích SHAP giữ riêng vì CalibratedClassifierCV gồm nhiều base estimators.
    joblib.dump(selected,output / "models" / "uncalibrated_selected_pipeline.joblib",compress=3)
    _write_report(output,data_audit,best_config,metrics,baseline,rows,calibrated_report)
    print("HOÀN THÀNH: Test Macro F1={:.4f}; model={}".format(metrics["f1_macro"],best_config),flush=True)
    return rows


def _write_report(output,audit,selection,metrics,baseline,rows,calibration):
    report=output / "reports" / "bao_cao_thuc_nghiem.md"
    txt=["# Báo cáo thực nghiệm — Phân loại nhóm nghề nghiệp (v3.1)",
         "","> Chỉ dự đoán `recommended_career`. **Không dự đoán đậu/trượt tuyển sinh.**",
         "",f"## 1. Dữ liệu và kiểm tra chất lượng",
         f"- {audit['rows']} bản ghi, {audit['columns']} cột; {len(audit['target_counts'])} lớp.",
         f"- Missing: {audit['missing_total']}; trùng hoàn toàn: {audit['duplicate_rows']}; trùng bỏ ID: {audit['duplicate_rows_without_student_id']}.",
         "- Loại student_id và 9 cột nghi ngờ chứa tín hiệu hậu quyết định (chưa chứng minh nhân quả/leakage).",
         "- Chia 80/20 có stratify; tiền xử lý + FS fit trong từng fold; Test giữ riêng.",
         "", "## 2. Thực nghiệm CV", "- Dummy stratified, không tuning; 3 mô hình × 4 FS, dùng GridSearchCV 5 folds.",
         "- Chỉ số chọn: F1 Macro. Điểm CV không phải Test và có thể lạc quan do chọn tham số.",
         f"- Dummy CV F1 Macro: {baseline['best_cv_f1_macro']:.4f}.",
         f"- Mô hình chọn theo CV: **{selection['model']} + {selection['fs']}**, "
         f"Macro F1 CV={selection['cv_f1_macro']:.4f}.",
         f"- Tham số tốt nhất: `{json.dumps(selection['best_params'],ensure_ascii=False)}`",
         "", "## 3. Đánh giá TEST (chỉ thực hiện sau khi chốt từ Train CV)"]
    txt += [f"- {k}: {v:.6f}" for k,v in metrics.items()]
    txt += ["", "## 4. Xác suất và calibration",
            "- Xác minh xác suất hợp lệ, cùng thứ tự classes_, tổng mỗi dòng xấp xỉ 1.",
            "- Log loss và biểu đồ reliability one-vs-rest 12 lớp nằm trong outputs/figures, bảng bins trong outputs/tables.",
            "- Diễn giải reliability thận trọng: mỗi bin có thể có ít mẫu, ECE phụ thuộc cách chia bins.",
            f"- Calibration chọn bằng validation từ Train: `{calibration}`" if calibration is not None else
            "- Chưa chủ động hiệu chỉnh: chỉ báo cáo calibration và hạn chế, không chọn phương pháp theo Test.",
            "", "## 5. Diễn giải và giới hạn",
            "- Dữ liệu không có nhãn đỗ/trượt thực tế nên không hỗ trợ xác suất tuyển sinh.",
            "- Xác suất là dự đoán theo dữ liệu, không có nghĩa xác suất phù hợp nghề đã được kiểm định.",
            "- Các cột có thể hình thành sau quyết định nghề được loại thận trọng; cần thẩm định nguồn sinh dữ liệu.",
            "- Nếu kết quả gần baseline, phải nêu dữ liệu có thể thiếu tín hiệu; không quảng bá tư vấn chắc chắn.",
            "- SHAP/Permutation là diễn giải mô hình, không chứng minh quan hệ nhân quả.",
            "", "## 6. Tệp minh chứng", "- outputs/tables/cv_experiment_results.csv (đủ 13 dòng)",
            "- outputs/tables/grid_01.csv ... grid_12.csv (từng tổ hợp grid)",
            "- outputs/tables/classification_report.csv, confusion_matrix.csv, test_metrics.json",
            "- outputs/figures/cv_model_comparison.png, confusion_matrix.png, calibration_12_classes.png",
            "- outputs/models/career_classifier.joblib (pipeline + schema + nhãn)", ""]
    init_path=output / "tables" / "initial_baselines_cv.csv"
    if init_path.exists():
        initial=pd.read_csv(init_path)
        txt += ["", "## 7. Baseline khởi tạo (trước GridSearchCV; đánh giá bằng Train CV)"]
        for _,r in initial.iterrows():
            txt.append(f"- {r['model']} mặc định: Macro F1 CV={r['cv_f1_macro']:.4f}; "
                       f"Accuracy CV={r['cv_accuracy']:.4f}; {r['seconds']:.1f} giây.")
    if selection["cv_f1_macro"] <= baseline["best_cv_f1_macro"]:
        txt += ["", "## KẾT LUẬN GIỚI HẠN", 
                "- **Mô hình chính tốt nhất chưa vượt Dummy baseline trên Macro F1 của Train CV.**",
                "- Không có bằng chứng rằng các đặc trưng hiện tại hỗ trợ khuyến nghị nghề đáng tin cậy. "
                "Demo chỉ có giá trị MINH HỌA KỸ THUẬT, tuyệt đối không dùng tư vấn thực tế."]
    report.write_text("\n".join(txt),encoding="utf-8")
