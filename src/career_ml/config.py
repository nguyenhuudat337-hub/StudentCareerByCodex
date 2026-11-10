"""Cấu hình chung của dự án. Không sửa dữ liệu gốc."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "college_student_career_selection_dataset.csv"
OUTPUT = ROOT / "outputs"
TARGET = "recommended_career"
ID_COL = "student_id"
SEED = 42
TEST_SIZE = 0.20
FOLDS = 5

# Loại THẬN TRỌNG các biến có nguy cơ được tính sau khi lựa chọn/đánh giá nghề.
# Tên cột không chứng minh leakage; cần xác minh quy trình tạo dữ liệu để quyết định cuối.
POTENTIAL_LEAKAGE = [
    "skill_demand_match_score", "academic_career_match_score",
    "interest_career_match_score", "competency_career_match_score",
    "career_suitability_score", "decision_step", "reward_score",
    "decision_consistency_score", "recommendation_reliability_score",
]
EXCLUDE_COLUMNS = [ID_COL, *POTENTIAL_LEAKAGE]
MODEL_NAMES = ("Logistic Regression", "Random Forest", "XGBoost")
FS_NAMES = ("None", "Filter", "Wrapper", "Embedded")

# Quick: 2 ứng viên / cấu hình, đủ 12 GridSearchCV và 5-fold, dùng toàn bộ Train.
# Standard: lưới rộng hơn; Full: lưới lớn, có thể tốn nhiều giờ.
GRID_PRESETS = ("quick", "standard", "full")
