"""Streamlit demo: EDA, dự đoán 12 nhóm, so sánh CV, giải thích."""
from pathlib import Path
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
def load_bundle(path):
    return joblib.load(path)  # chỉ mở artifact do chính project này huấn luyện

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
    st.plotly_chart(px.bar(counts,x="Nhóm nghề",y="Số mẫu",title="Phân bố nhãn"),use_container_width=True)
    st.dataframe(df.head(20),use_container_width=True)
    for fig_name,title in [("eda_boxplots.png","Boxplots"),("eda_correlation_heatmap.png","Tương quan")]:
        path=OUTPUT / "figures" / fig_name
        if path.exists():
            st.subheader(title);st.image(str(path),use_container_width=True)

elif page==choices[1]:
    st.subheader("Nhập thông tin sinh viên và dự đoán nhóm nghề")
    if not model_path.exists():
        st.info("Chưa có mô hình hoàn chỉnh. Chạy `python scripts/train.py --profile quick` ở thư mục dự án.")
        st.stop()
    bundle=load_bundle(str(model_path))
    model=bundle["model"]
    label_encoder=bundle["label_encoder"]
    schema=bundle["input_schema"]
    st.write("Mô hình:",bundle["selected_cv"]["model"],"| Feature Selection:",bundle["selected_cv"]["fs"])
    metrics=bundle.get("test_metrics",{})
    if metrics:
        cv_path=OUTPUT / "tables" / "cv_experiment_results.csv"
        if cv_path.exists():
            cv_comparison=pd.read_csv(cv_path,keep_default_na=False)
            base_row=cv_comparison[cv_comparison["model"]=="Dummy"]
            if not base_row.empty and float(bundle["selected_cv"]["cv_f1_macro"])<=float(base_row.iloc[0]["best_cv_f1_macro"]):
                st.error("Mô hình chưa vượt baseline ngẫu nhiên trên Macro F1 CV. "
                         "Chỉ xem dự đoán như minh họa, KHÔNG dùng để tư vấn nghề thực tế.")
        st.info(f"Kết quả kiểm tra Test: Macro F1 = {metrics['f1_macro']:.3f}; "
                f"Log loss = {metrics['log_loss_multiclass']:.3f}. "
                "Chất lượng xác suất cần xem cùng calibration; không suy ra độ phù hợp hay tỷ lệ đỗ.")
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
        submit=st.form_submit_button("Dự đoán 12 nhóm nghề",type="primary",use_container_width=True)
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
        st.plotly_chart(fig,use_container_width=True)
        st.caption("Xác suất có thể chưa được hiệu chỉnh tốt. Không dùng làm tỷ lệ trúng tuyển hoặc bảo đảm nghề phù hợp.")

elif page==choices[2]:
    st.subheader("So sánh 13 cấu hình theo Train CV")
    result_path=OUTPUT / "tables" / "cv_experiment_results.csv"
    if not result_path.exists():
        st.info("Chạy train.py để tạo bảng thực nghiệm đầy đủ.");st.stop()
    results=pd.read_csv(result_path,keep_default_na=False)
    st.dataframe(results,use_container_width=True)
    st.caption("Các điểm CV trên tập Train, KHÔNG phải điểm Test. Chọn model bằng CV trước khi đánh giá Test.")
    group=results[results["model"]!="Dummy"]
    st.plotly_chart(px.bar(group,x="model",y="best_cv_f1_macro",color="fs",barmode="group",
                           title="Macro F1 bằng StratifiedKFold(5)"),use_container_width=True)
    metrics_path=OUTPUT / "tables" / "test_metrics.json"
    if metrics_path.exists():
        import json
        st.subheader("Chỉ số Test của mô hình cuối")
        st.json(json.loads(metrics_path.read_text(encoding="utf-8")))

elif page==choices[3]:
    st.subheader("Giải thích mô hình — Feature Importance, Permutation Importance và SHAP")
    st.caption("SHAP giải thích đầu ra mô hình, không chứng minh quan hệ nhân quả hoặc trúng tuyển thực tế.")
    for title,file in [("Feature Importance","feature_importance.png"),
                       ("Permutation Importance","permutation_importance.png"),
                       ("SHAP Summary (global)","shap_summary.png"),
                       ("SHAP Waterfall (local)","shap_waterfall.png")]:
        p=OUTPUT / "figures" / file
        if p.exists():st.subheader(title);st.image(str(p),use_container_width=True)
    st.caption("Nếu chưa có SHAP: chạy `python scripts/run_explain.py` sau khi huấn luyện đầy đủ.")
