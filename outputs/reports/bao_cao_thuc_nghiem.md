# Báo cáo thực nghiệm — Phân loại nhóm nghề nghiệp (v3.1)

> Chỉ dự đoán `recommended_career`. **Không dự đoán đậu/trượt tuyển sinh.**

## 1. Dữ liệu và kiểm tra chất lượng
- 12000 bản ghi, 50 cột; 12 lớp.
- Missing: 0; trùng hoàn toàn: 0; trùng bỏ ID: 0.
- Loại student_id và 9 cột nghi ngờ chứa tín hiệu hậu quyết định (chưa chứng minh nhân quả/leakage).
- Chia 80/20 có stratify; tiền xử lý + FS fit trong từng fold; Test giữ riêng.

## 2. Thực nghiệm CV
- Dummy stratified, không tuning; 3 mô hình × 4 FS, dùng GridSearchCV 5 folds.
- Chỉ số chọn: F1 Macro. Điểm CV không phải Test và có thể lạc quan do chọn tham số.
- Dummy CV F1 Macro: 0.0884.
- Mô hình chọn theo CV: **XGBoost + None**, Macro F1 CV=0.0882.
- Tham số tốt nhất: `{"clf__learning_rate": 0.12}`

## 3. Đánh giá TEST (chỉ thực hiện sau khi chốt từ Train CV)
- accuracy: 0.080000
- precision_macro: 0.080304
- recall_macro: 0.080000
- f1_macro: 0.079747
- f1_weighted: 0.079747
- log_loss_multiclass: 2.542523

## 4. Xác suất và calibration
- Xác minh xác suất hợp lệ, cùng thứ tự classes_, tổng mỗi dòng xấp xỉ 1.
- Log loss và biểu đồ reliability one-vs-rest 12 lớp nằm trong outputs/figures, bảng bins trong outputs/tables.
- Diễn giải reliability thận trọng: mỗi bin có thể có ít mẫu, ECE phụ thuộc cách chia bins.
- Chưa chủ động hiệu chỉnh: chỉ báo cáo calibration và hạn chế, không chọn phương pháp theo Test.

## 5. Diễn giải và giới hạn
- Dữ liệu không có nhãn đỗ/trượt thực tế nên không hỗ trợ xác suất tuyển sinh.
- Xác suất là dự đoán theo dữ liệu, không có nghĩa xác suất phù hợp nghề đã được kiểm định.
- Các cột có thể hình thành sau quyết định nghề được loại thận trọng; cần thẩm định nguồn sinh dữ liệu.
- Nếu kết quả gần baseline, phải nêu dữ liệu có thể thiếu tín hiệu; không quảng bá tư vấn chắc chắn.
- SHAP/Permutation là diễn giải mô hình, không chứng minh quan hệ nhân quả.

## 6. Tệp minh chứng
- outputs/tables/cv_experiment_results.csv (đủ 13 dòng)
- outputs/tables/grid_01.csv ... grid_12.csv (từng tổ hợp grid)
- outputs/tables/classification_report.csv, confusion_matrix.csv, test_metrics.json
- outputs/figures/cv_model_comparison.png, confusion_matrix.png, calibration_12_classes.png
- outputs/models/career_classifier.joblib (pipeline + schema + nhãn)


## 7. Baseline khởi tạo (trước GridSearchCV; đánh giá bằng Train CV)
- Logistic Regression mặc định: Macro F1 CV=0.0793; Accuracy CV=0.0806; 1.8 giây.
- Random Forest mặc định: Macro F1 CV=0.0823; Accuracy CV=0.0830; 11.7 giây.
- XGBoost mặc định: Macro F1 CV=0.0905; Accuracy CV=0.0909; 8.3 giây.

## KẾT LUẬN GIỚI HẠN
- **Mô hình chính tốt nhất chưa vượt Dummy baseline trên Macro F1 của Train CV.**
- Không có bằng chứng rằng các đặc trưng hiện tại hỗ trợ khuyến nghị nghề đáng tin cậy. Demo chỉ có giá trị MINH HỌA KỸ THUẬT, tuyệt đối không dùng tư vấn thực tế.