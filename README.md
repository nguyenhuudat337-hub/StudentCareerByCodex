# Phân loại nhóm nghề nghiệp và kết quả tuyển sinh của sinh viên — Kế hoạch 3.1

**Phạm vi khoa học:** một bài toán phân loại đa lớp từ dữ liệu sinh viên, nhãn `recommended_career` (12 nhóm). Dữ liệu **KHÔNG** có biến đậu/trượt tuyển sinh. `predict_proba()` là phân bố xác suất mô hình dự đoán nghề, **không phải xác suất trúng tuyển** và chưa chắc là xác suất phù hợp nghề được hiệu chỉnh.

## 1. Cài đặt

Yêu cầu Python 3.10+ (khuyên 3.11 hoặc 3.12), đứng ở thư mục gốc của dự án.

```bash
python -m venv .venv
source .venv/bin/activate            # macOS/Linux
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

## 2. Chạy theo thứ tự

```bash
python scripts/run_eda.py
python scripts/smoke_test.py
python scripts/train.py --profile quick --jobs 2
python scripts/run_explain.py --samples 80
streamlit run app.py
```

**`quick` vẫn chạy 100% dữ liệu, đủ 5 folds, 13 cấu hình (Dummy + 12 GridSearchCV).** Grid nhỏ, 2 ứng viên cho mỗi mô hình chính nên chỉ thử tham số cơ bản. Nếu muốn lưới rộng hơn, chạy lại với `--profile standard` hoặc `--profile full` (rất tốn tài nguyên/thời gian); các lần chạy trên cùng output sẽ ghi đè tệp kết quả. `--jobs 2` an toàn với RAM cá nhân hơn `-1`.

Chạy **debug** nhanh (không có Test, không tạo model demo và không dùng số liệu debug để báo cáo):

```bash
python scripts/train.py --debug-rows 480 --limit 2 --jobs 1
```

Đánh giá hiệu chỉnh xác suất trên **validation nằm trong Train**, trước Test:

```bash
python scripts/train.py --profile quick --jobs 2 --compare-calibration
```

Chức năng trên so sánh log loss của sigmoid calibration với pipeline gốc trên tập validation lấy từ Train; chỉ áp dụng nếu cải thiện đủ mức đã cấu hình; khi áp dụng, hiệu chỉnh lại với Train bằng CV 3 folds, **không dùng Test để chọn cách hiệu chỉnh**. Mặc định KHÔNG áp dụng vì không thể kết luận cần calibration chỉ dựa trên Test.

## 3. Cấu trúc

- `data/raw/`: CSV gốc, luôn giữ nguyên.
- `docs/`: kế hoạch v3.1 đã cung cấp.
- `src/career_ml/config.py`: nhãn, seed, danh sách loại cột nguy cơ leakage.
- `src/career_ml/data.py`: đọc dữ liệu, missing/duplicates, split, schema form.
- `src/career_ml/eda.py`: histogram, countplot, boxplot, heatmap, class distribution.
- `src/career_ml/modeling.py`: Dummy, LR, RF, XGBoost; None/Filter/Wrapper/Embedded.
- `src/career_ml/experiment.py`: 12 GridSearchCV, 5-fold CV, chọn từ Train rồi Test duy nhất.
- `src/career_ml/evaluation.py`: metrics, confusion, calibration 12 lớp, log loss.
- `src/career_ml/explain.py`: feature importance, permutation, SHAP summary/waterfall.
- `app.py`: trang EDA, nhập thông tin, top-3 và đầy đủ 12 xác suất, so sánh, XAI.
- `notebooks/`: notebook Jupyter cho 7 giai đoạn.
- `outputs/tables/`, `outputs/figures/`, `outputs/reports/`, `outputs/models/`: được sinh từ thực nghiệm thật; chưa có kết quả thì không có số liệu.
- `tests/`: kiểm tra model pipeline, mapping xác suất, leakage, schema.

## 4. Tính công bằng và giới hạn dữ liệu

- Chia Train/Test 80/20, stratify, seed=42; giữ Test độc lập đến bước đánh giá cuối.
- Mỗi GridSearchCV bao gồm **toàn bộ** `preprocess → feature selection → classifier`, với `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, `scoring=f1_macro`.
- **Không** chạy FS/OHE/Scaler trên toàn dữ liệu trước CV. Với bản 3.1, lựa chọn mô hình chỉ dùng Train CV.
- Bắt buộc kiểm tra `isnull().sum()`, `duplicated().sum()` và trùng khi bỏ `student_id`; không tự xóa bản ghi.
- `config.py` mặc định loại `student_id` và 9 cột **nghi ngờ** được tính sau quyết định nghề: `skill_demand_match_score`, `academic_career_match_score`, `interest_career_match_score`, `competency_career_match_score`, `career_suitability_score`, `decision_step`, `reward_score`, `decision_consistency_score`, `recommendation_reliability_score`. Đó là biện pháp phòng tránh thận trọng, **không phải kết luận chắc chắn** các cột đã leakage. Cần kiểm tra nguồn gốc/tài liệu dữ liệu trước khi thay đổi.
- CV và Test có thể gần mức ngẫu nhiên nếu nhãn thiếu tín hiệu. **Không được chỉnh model sau khi xem Test để làm đẹp số.**
- Log loss và biểu đồ calibration chỉ đo đạc trên Test **sau khi đã chốt** model/calibration. Reliability one-vs-rest phụ thuộc số lượng mẫu trong bin; không chứng minh xác suất đáng tin cậy tuyệt đối.
- Với SHAP, đại lượng giải thích có thể là margin/logit thay vì xác suất. Không diễn giải thành tác động nhân quả.
- File `.joblib` dùng cơ chế pickle: **không tải mô hình từ người lạ**.

## 5. Kết quả sau khi chạy

`initial_baselines_cv.csv` (3 mô hình chưa tuning); `cv_experiment_results.csv` (13 dòng); `grid_01.csv` đến `grid_12.csv`; `selected_model_train_cv.json`; `test_metrics.json`, `classification_report.csv`, `confusion_matrix.csv`, `calibration_bins.csv`, `test_predictions.csv`; hình EDA/CV/confusion/calibration/XAI; báo cáo `bao_cao_thuc_nghiem.md`; mô hình `career_classifier.joblib`.

## 6. Kiểm thử

```bash
pytest -q
```

**Chú ý:** nếu chỉ có mã nguồn mà chưa chạy train đầy đủ, Streamlit vẫn xem EDA được nhưng chưa có kết quả dự đoán. Mọi số liệu thực nghiệm phải sinh từ chạy thuật toán thật.
