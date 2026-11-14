"""Streamlit demo: EDA, dự đoán 12 nhóm, so sánh CV, giải thích."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from career_ml.config import ROOT, OUTPUT, DATA_PATH

st.set_page_config(page_title="Phân loại nhóm nghề nghiệp — V3.1", page_icon="🎓",layout="wide")
st.title("🎓 Phân loại nhóm nghề nghiệp của sinh viên")
st.caption("Đề tài Phân tích dữ liệu lớn — 1 bài toán phân loại đa lớp, 12 nhóm nghề nghiệp")
st.warning("**Lưu ý:** Xác suất mô hình gán cho nhóm nghề nghiệp được khuyến nghị, "
           "KHÔNG phải xác suất trúng tuyển vào ngành/trường và không bảo đảm mức độ phù hợp nghề.")

@st.cache_resource
def load_bundle(path, modified_time_ns, file_size):
    """Tải lại mô hình khi thời gian sửa hoặc kích thước file thay đổi."""
    return joblib.load(path)

@st.cache_data
def load_csv():
    return pd.read_csv(DATA_PATH)

model_path=OUTPUT / "models" / "career_classifier.joblib"
choices=["📊 Tổng quan dữ liệu", "🔮 Dự đoán nghề nghiệp", "📈 So sánh thực nghiệm", "🧠 Giải thích mô hình"]
page=st.sidebar.radio("Điều hướng",choices)
st.sidebar.markdown("---")
st.sidebar.write("Kế hoạch v3.1 | Seed=42 | Train/Test=80/20 | 5-fold CV")

if page==choices[0]:
    st.subheader("Tổng quan dữ liệu và EDA")
    if not DATA_PATH.exists():
        st.error("Không tìm thấy CSV trong data/raw");st.stop()
    df=load_csv()
    a,b,c=st.columns(3)
    a.metric("Số bản ghi",f"{len(df):,}")
    b.metric("Số cột",len(df.columns))
    c.metric("Số lớp",df["recommended_career"].nunique())
    quality_path=OUTPUT / "tables" / "data_quality_audit.json"
    st.write("Missing values:",int(df.isna().sum().sum()))
    st.write("Trùng hoàn toàn:",int(df.duplicated().sum()))
    st.write("Trùng nếu bỏ student_id:",int(df.drop(columns="student_id",errors="ignore").duplicated().sum()))
    counts=df["recommended_career"].value_counts().reset_index()
    counts.columns=["Nhóm nghề","Số mẫu"]
    st.plotly_chart(px.bar(counts,x="Nhóm nghề",y="Số mẫu",title="Phân bố nhãn"),width="stretch")
    st.dataframe(df.head(20),width="stretch")
    for fig_name,title in [("eda_boxplots.png","Boxplots"),("eda_correlation_heatmap.png","Tương quan")]:
        path=OUTPUT / "figures" / fig_name
        if path.exists():
            st.subheader(title);st.image(str(path),width="stretch")

elif page==choices[1]:
    st.subheader("Nhập thông tin sinh viên và dự đoán nhóm nghề")
    if not model_path.exists():
        st.info("Chưa có mô hình hoàn chỉnh. Chạy `python scripts/train.py --profile quick` ở thư mục dự án.")
        st.stop()
    model_stat = model_path.stat()
    bundle = load_bundle(
        str(model_path),
        model_stat.st_mtime_ns,
        model_stat.st_size,
    )
    model=bundle["model"]
    label_encoder=bundle["label_encoder"]
    schema=bundle["input_schema"]
    st.write("Mô hình:",bundle["selected_cv"]["model"],"| Feature Selection:",bundle["selected_cv"]["fs"])
    metrics = bundle.get("test_metrics", {})
    if metrics:
        st.info(
            f"Kết quả kiểm tra Test: Macro F1 = {metrics['f1_macro']:.3f}; "
            f"Log loss = {metrics['log_loss_multiclass']:.3f}. "
            "Chất lượng xác suất cần xem cùng calibration; không suy ra độ phù hợp hay tỷ lệ đỗ."
        )
    with st.form("student_data"):
        vals={}
        numeric_cols=[k for k,v in schema.items() if v["kind"]=="numeric"]
        categorical_cols=[k for k,v in schema.items() if v["kind"]=="categorical"]
        with st.expander("Thông tin học tập, kỹ năng và chỉ số số học",expanded=True):
            cols=st.columns(3)
            for idx,key in enumerate(numeric_cols):
                settings=schema[key]
                if settings["integer"]:
                    vals[key]=cols[idx%3].number_input(key,min_value=int(np.floor(settings["min"])),
                                max_value=int(np.ceil(settings["max"])),
                                value=int(round(settings["median"])),step=1)
                else:
                    vals[key]=cols[idx%3].number_input(key,
                                min_value=float(settings["min"]),max_value=float(settings["max"]),
                                value=float(settings["median"]),format="%.2f")
        with st.expander("Thông tin phân loại và sở thích",expanded=True):
            cols=st.columns(2)
            for idx,key in enumerate(categorical_cols):
                settings=schema[key]
                vals[key]=cols[idx%2].selectbox(key,options=settings["choices"],
                     index=settings["choices"].index(settings["default"]))
        submit=st.form_submit_button("Dự đoán 12 nhóm nghề",type="primary",width="stretch")
    if submit:
        student=pd.DataFrame([{name:vals[name] for name in bundle["raw_features"]}])
        proba=np.asarray(model.predict_proba(student))[0]
        classes=np.asarray(model.classes_,dtype=int)
        if len(classes)!=len(label_encoder.classes_) or not np.isclose(proba.sum(),1,atol=1e-5):
            st.error("Vector xác suất sai cấu trúc. Hãy kiểm tra mô hình.");st.stop()
        labels=label_encoder.inverse_transform(classes)
        ranking=np.argsort(proba)[::-1]
        best=ranking[0]
        st.success(f"**Nhóm nghề được dự đoán:** {labels[best]}")
        st.markdown("#### Top 3 nhóm nghề (theo xác suất mô hình)")
        topcols=st.columns(3)
        for i,ix in enumerate(ranking[:3]):
            topcols[i].metric(f"#{i+1} {labels[ix]}",f"{proba[ix]*100:.2f}%")
        chart=pd.DataFrame({"Nhóm nghề":labels,"Xác suất":proba}).sort_values("Xác suất",ascending=False)
        fig=px.bar(chart,x="Xác suất",y="Nhóm nghề",orientation="h",
                   title="Xác suất dự đoán của mô hình cho đủ 12 nhóm nghề",
                   text=chart["Xác suất"].map(lambda x:f"{x*100:.1f}%"))
        fig.update_layout(yaxis={"categoryorder":"total ascending"},xaxis_tickformat=".0%")
        st.plotly_chart(fig,width="stretch")
        st.caption("Xác suất có thể chưa được hiệu chỉnh tốt. Không dùng làm tỷ lệ trúng tuyển hoặc bảo đảm nghề phù hợp.")
        st.subheader("Giải thích dự đoán cho hồ sơ vừa nhập")

        calibration = bundle.get("calibration_train_selection") or {}
        if calibration.get("use_calibrated", False):
            st.info(
                "Phần giải thích trực tiếp chưa hỗ trợ mô hình "
                "đã hiệu chỉnh xác suất."
            )
        else:
            from career_ml.explain import explain_student
            from career_ml.data import load_data, prepare_xy, split_data
            from career_ml.config import SEED
            import shap
            import matplotlib.pyplot as plt

            waterfall_fig = None
            try:
                with st.spinner("Đang tính đóng góp của các đặc trưng..."):
                    background = None

                    # Logistic Regression cần dữ liệu nền lấy từ Train.
                    if hasattr(model.named_steps["clf"], "coef_"):
                        X, y, _ = prepare_xy(
                            load_data(),
                            excluded=bundle["excluded_columns"],
                        )
                        X_train, _, _, _ = split_data(X, y)
                        background = X_train.sample(
                            n=min(80, len(X_train)),
                            random_state=SEED,
                        )

                    explanation_result = explain_student(
                        bundle,
                        student,
                        background_train=background,
                    )

                    if (
                        explanation_result["class_name"] != str(labels[best])
                        or not np.isclose(
                            explanation_result["probability"], proba[best]
                        )
                    ):
                        raise ValueError(
                            "Kết quả giải thích không khớp dự đoán vừa hiển thị."
                        )

                    waterfall_fig = plt.figure()
                    shap.plots.waterfall(
                        explanation_result["explanation"],
                        max_display=15,
                        show=False,
                    )
                    waterfall_fig = plt.gcf()

                st.write(
                    "Nhóm nghề đang được giải thích:",
                    explanation_result["class_name"],
                )
                st.caption(
                    f"Thang đo SHAP: {explanation_result['scale']}. "
                    "Các giá trị đặc trưng trên hình đã được chuẩn hóa "
                    "hoặc mã hóa theo pipeline."
                )
                st.pyplot(waterfall_fig)

                st.caption(
                    "Thanh đỏ làm tăng đầu ra của lớp đang giải thích; "
                    "thanh xanh làm giảm đầu ra so với mức nền. "
                    "Đây là giải thích hành vi mô hình, "
                    "không chứng minh mức độ phù hợp nghề nghiệp."
                )
            except Exception as exc:
                st.warning(
                    "Đã có kết quả dự đoán nhưng chưa tạo được giải thích."
                )
                st.code(f"{type(exc).__name__}: {exc}")
            finally:
                if waterfall_fig is not None:
                    plt.close(waterfall_fig)

elif page==choices[2]:
    st.subheader("So sánh 12 cấu hình theo Train CV")
    result_path = OUTPUT / "tables" / "cv_experiment_results.csv"

    if not result_path.exists() or not model_path.exists():
        st.info("Hãy hoàn tất huấn luyện để xem kết quả.")
        st.stop()

    model_stat = model_path.stat()
    bundle = load_bundle(
        str(model_path),
        model_stat.st_mtime_ns,
        model_stat.st_size,
    )

    try:
        results = pd.read_csv(result_path, keep_default_na=False)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        st.warning("Chưa đọc được bảng kết quả. Hãy chờ huấn luyện hoàn tất.")
        st.stop()

    run_id = bundle.get("run_id")
    if (
        not run_id
        or "run_id" not in results.columns
        or len(results) != 12
        or not results["run_id"].eq(run_id).all()
    ):
        st.warning(
            "Bảng thực nghiệm chưa đầy đủ hoặc không cùng lần huấn luyện "
            "với mô hình. Hãy hoàn tất huấn luyện rồi mở lại trang."
        )
        st.stop()

    st.dataframe(results.drop(columns="run_id"), width="stretch")
    st.caption(
        "Các điểm CV trên tập Train, KHÔNG phải điểm Test. "
        "Mô hình được chọn bằng CV trước khi đánh giá Test."
    )

    st.plotly_chart(
        px.bar(
            results,
            x="model",
            y="best_cv_f1_macro",
            color="fs",
            barmode="group",
            title="Macro F1 bằng StratifiedKFold(5)",
        ),
        width="stretch",
    )

    metrics = bundle.get("test_metrics")
    if metrics:
        st.subheader("Chỉ số Test của mô hình cuối")
        st.json(metrics)

elif page==choices[3]:
    st.subheader("Giải thích mô hình — Feature Importance, Permutation Importance và SHAP")
    st.caption("SHAP giải thích đầu ra mô hình, không chứng minh quan hệ nhân quả hoặc trúng tuyển thực tế.")
    metadata_path = OUTPUT / "tables" / "shap_metadata.json"

    if not model_path.exists() or not metadata_path.exists():
        st.warning(
            "Chưa có đủ mô hình và kết quả giải thích. "
            "Hãy hoàn tất huấn luyện và chạy scripts/run_explain.py."
        )
        st.stop()

    model_stat = model_path.stat()
    bundle = load_bundle(
        str(model_path),
        model_stat.st_mtime_ns,
        model_stat.st_size,
    )

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        st.warning("Chưa đọc được thông tin XAI. Hãy tạo lại kết quả giải thích.")
        st.stop()

    model_run_id = bundle.get("run_id")
    xai_run_id = metadata.get("run_id") if isinstance(metadata, dict) else None

    if not model_run_id or xai_run_id != model_run_id:
        st.warning(
            "Hình giải thích chưa khớp với mô hình hiện tại. "
            "Hãy chạy lại scripts/run_explain.py sau khi huấn luyện."
        )
        st.stop()
    if metadata.get("data_source") != "test_sample":
        st.warning(
            "Thông tin về mẫu giải thích chưa được cập nhật. "
            "Hãy chạy lại scripts/run_explain.py."
        )
        st.stop()

    explained_class = metadata["explained_class"]
    sample_count = metadata["n_samples_explained"]

    st.info(
        "Các hình bên dưới là kết quả giải thích đã tạo sẵn. "
        "Feature Importance lấy từ mô hình đã học trên Train; "
        "Permutation Importance và SHAP được phân tích trên mẫu Test. "
        "Chúng không giải thích hồ sơ vừa nhập ở trang Dự đoán nghề nghiệp."
    )

    st.markdown(
        f"**SHAP Summary:** đóng góp của các đặc trưng đối với lớp "
        f"**{explained_class}**, trên **{sample_count} hồ sơ Test**."
    )

    st.markdown(
        f"**SHAP Waterfall:** giải thích lớp **{explained_class}** "
        f"được pipeline dự đoán cho hồ sơ có chỉ mục "
        f"**{metadata['local_sample_index']}**. "
        f"Nhãn thực tế trong dữ liệu: **{metadata['local_true_class']}**."
    )

    st.caption(
        "Chỉ mục trên là chỉ mục dòng dữ liệu, không phải mã sinh viên. "
        "SHAP Summary này chỉ mô tả lớp được ghi tên, "
        "không tổng hợp độ quan trọng cho cả 12 lớp."
    )

    if metadata.get("prediction_model_calibrated"):
        st.warning(
            "Mô hình dự đoán đã được hiệu chỉnh xác suất. "
            "Các hình này giải thích pipeline trước hiệu chỉnh, "
            "không trực tiếp giải thích xác suất cuối cùng."
        )

    st.caption(
        "Giá trị SHAP không mặc nhiên là phần trăm xác suất. "
        "Với XGBoost hiện tại, chúng giải thích đầu ra margin của mô hình."
    )
    for title,file in [("Feature Importance","feature_importance.png"),
                       ("Permutation Importance","permutation_importance.png"),
                       ("SHAP Summary (global)","shap_summary.png"),
                       ("SHAP Waterfall (local)","shap_waterfall.png")]:
        p=OUTPUT / "figures" / file
        if p.exists():st.subheader(title);st.image(str(p),width="stretch")
    st.caption("Nếu chưa có SHAP: chạy `python scripts/run_explain.py` sau khi huấn luyện đầy đủ.")
