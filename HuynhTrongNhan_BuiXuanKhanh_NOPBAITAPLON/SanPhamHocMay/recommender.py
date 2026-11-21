"""SmartCourse: phân loại chủ đề và gợi ý môn học.

Phân loại chủ đề: TF-IDF + Multinomial Naive Bayes.
Xếp hạng môn học: hệ thống tính điểm theo quy tắc, không phải mô hình học máy.
"""

from pathlib import Path
import re
import unicodedata

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
COURSES_PATH = DATA_DIR / "courses.csv"
TOPICS_PATH = DATA_DIR / "topics.csv"

REQUIRED_COURSE_COLUMNS = {
    "ma_mon", "ten_mon", "hoc_ky_goi_y", "so_tin_chi",
    "nhom_chu_de", "mo_ta", "mon_tien_quyet",
    "kien_thuc_nen", "loai_mon", "trang_thai_xac_minh"
}
REQUIRED_TOPIC_COLUMNS = {"mo_ta", "nhom_chu_de"}


def _clean_text(value):
    """Chuẩn hóa văn bản đầu vào, giữ nguyên dấu tiếng Việt."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFC", str(value))
    return re.sub(r"\s+", " ", text).strip()


def load_data():
    """Đọc hai file CSV và kiểm tra cấu trúc dữ liệu cơ bản."""
    if not COURSES_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {COURSES_PATH}")
    if not TOPICS_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {TOPICS_PATH}")

    courses = pd.read_csv(COURSES_PATH, encoding="utf-8-sig").fillna("")
    topics = pd.read_csv(TOPICS_PATH, encoding="utf-8-sig").fillna("")

    missing_courses = REQUIRED_COURSE_COLUMNS - set(courses.columns)
    missing_topics = REQUIRED_TOPIC_COLUMNS - set(topics.columns)
    if missing_courses:
        raise ValueError(f"courses.csv thiếu cột: {', '.join(sorted(missing_courses))}")
    if missing_topics:
        raise ValueError(f"topics.csv thiếu cột: {', '.join(sorted(missing_topics))}")
    if courses.empty:
        raise ValueError("courses.csv chưa có dữ liệu môn học.")
    if topics.empty:
        raise ValueError("topics.csv chưa có dữ liệu huấn luyện.")

    # Bỏ các dòng huấn luyện thiếu mô tả hoặc nhãn để tránh huấn luyện dữ liệu rỗng.
    topics = topics.copy()
    topics["mo_ta"] = topics["mo_ta"].map(_clean_text)
    topics["nhom_chu_de"] = topics["nhom_chu_de"].map(_clean_text)
    topics = topics[(topics["mo_ta"] != "") & (topics["nhom_chu_de"] != "")]
    if topics.empty:
        raise ValueError("topics.csv không có dòng nào chứa đủ mô tả và nhóm chủ đề.")

    return courses, topics.reset_index(drop=True)


def train_topic_model(topics=None):
    """Huấn luyện mô hình TF-IDF kết hợp Multinomial Naive Bayes."""
    if topics is None:
        _, topics = load_data()

    if not {"mo_ta", "nhom_chu_de"}.issubset(topics.columns):
        raise ValueError("Dữ liệu huấn luyện cần có cột mo_ta và nhom_chu_de.")

    training = topics[["mo_ta", "nhom_chu_de"]].copy().fillna("")
    training["mo_ta"] = training["mo_ta"].map(_clean_text)
    training["nhom_chu_de"] = training["nhom_chu_de"].map(_clean_text)
    training = training[(training["mo_ta"] != "") & (training["nhom_chu_de"] != "")]
    if training["nhom_chu_de"].nunique() < 2:
        raise ValueError("Cần ít nhất hai nhóm chủ đề để huấn luyện mô hình phân loại.")

    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("naive_bayes", MultinomialNB(alpha=1.0))
    ])
    model.fit(training["mo_ta"], training["nhom_chu_de"])
    return model


def classify_topic(text, model=None):
    """Phân loại mô tả; trả về nhãn rỗng nếu không có đủ dấu hiệu từ vựng."""
    text = _clean_text(text)
    if not text:
        return {
            "nhom_chu_de": "", "do_tin_cay": 0.0,
            "trang_thai": "khong_co_du_lieu",
            "ly_do": "Cậu hãy nhập mô tả mục tiêu học tập."
        }

    if model is None:
        model = train_topic_model()

    vectorizer = model.named_steps["tfidf"]
    vector = vectorizer.transform([text])
    # Nếu không từ/nhóm từ nào trùng với dữ liệu huấn luyện, không đoán bừa một nhãn.
    if vector.nnz == 0:
        return {
            "nhom_chu_de": "", "do_tin_cay": 0.0,
            "trang_thai": "khong_nhan_dien_duoc",
            "ly_do": "Nội dung chưa khớp với từ vựng của dữ liệu chủ đề. Hãy mô tả cụ thể hơn."
        }

    probabilities = model.predict_proba([text])[0]
    best_index = int(probabilities.argmax())
    best_probability = float(probabilities[best_index])
    # Số đặc trưng được khớp chỉ là tín hiệu hỗ trợ, không phải thước đo độ chính xác.
    if vector.nnz < 2:
        return {
            "nhom_chu_de": "", "do_tin_cay": best_probability,
            "trang_thai": "can_mo_ta_ro_hon",
            "ly_do": "Mô tả còn quá ngắn hoặc chung chung để phân loại đáng tin cậy."
        }

    return {
        "nhom_chu_de": str(model.classes_[best_index]),
        "do_tin_cay": best_probability,
        "trang_thai": "da_du_doan",
        "ly_do": "Xác suất do mô hình ước lượng; chưa phải độ chính xác đã được kiểm chứng."
    }


def _parse_prerequisites(value):
    """Tách danh sách mã môn tiên quyết, chấp nhận dấu phẩy/chấm phẩy/gạch đứng."""
    value = _clean_text(value)
    if not value:
        return []
    return [code.strip().upper() for code in re.split(r"[,;|]", value) if code.strip()]


def _semester_position(value):
    """Đổi học kỳ gợi ý thành vị trí gần đúng trên lộ trình.

    Học kỳ chính dùng số nguyên; Hè năm 1 nằm sau học kỳ 2,
    Hè năm 2 nằm sau học kỳ 4. Đây chỉ là quy tắc sắp xếp/gợi ý.
    """
    text = _clean_text(value).lower()
    match = re.search(r"học\s*kỳ\s*(\d+)", text)
    if match:
        return float(match.group(1))
    match = re.search(r"hè\s*năm\s*(\d+)", text)
    if match:
        year = int(match.group(1))
        if year == 1:
            return 2.5
        if year == 2:
            return 4.5
        return None
    return None


def get_recommendations(
    interests=None,
    goal_text="",
    completed_courses=None,
    knowledge_level="co_ban",
    current_semester=None,
    top_n=8,
    courses=None,
    model=None
):
    """Xếp hạng môn học theo sở thích, mục tiêu, nền tảng và học kỳ.

    Điểm là điểm quy tắc để sắp xếp, không phải xác suất AI.
    Các tiên quyết trống không được xem là đã xác minh rằng môn không có tiên quyết.
    """
    if courses is None:
        courses, _ = load_data()
    if model is None:
        model = train_topic_model()
    if courses.empty:
        return courses.copy()

    interests = {_clean_text(item) for item in (interests or []) if _clean_text(item)}
    completed = {str(code).strip().upper() for code in (completed_courses or []) if str(code).strip()}
    goal_text = _clean_text(goal_text)
    goal_result = classify_topic(goal_text, model) if goal_text else {"nhom_chu_de": ""}
    goal_topic = goal_result["nhom_chu_de"]

    try:
        semester_now = int(current_semester) if current_semester is not None else None
    except (TypeError, ValueError):
        semester_now = None

    recommendations = []
    for _, course in courses.iterrows():
        code = _clean_text(course["ma_mon"]).upper()
        if code and code in completed:
            continue

        category = _clean_text(course["nhom_chu_de"])
        score = 0
        reasons = []

        if category in interests:
            score += 40
            reasons.append("Phù hợp với chủ đề quan tâm")
        if goal_topic and category == goal_topic:
            score += 30
            reasons.append("Gần với mục tiêu học tập")

        course_type = _clean_text(course["loai_mon"])
        if knowledge_level == "co_ban" and course_type == "nen_tang_AI":
            score += 20
            reasons.append("Hỗ trợ củng cố kiến thức nền")
        elif knowledge_level == "trung_binh" and course_type in {"nen_tang_AI", "mon_bo_tro"}:
            score += 12
            reasons.append("Phù hợp để bổ sung kiến thức")
        elif knowledge_level == "kha" and course_type == "chuyen_nganh_AI":
            score += 20
            reasons.append("Phù hợp với trình độ hiện tại")

        course_position = _semester_position(course["hoc_ky_goi_y"])
        semester_delta = None
        if semester_now is not None and course_position is not None:
            semester_delta = course_position - semester_now
            if 0 <= semester_delta <= 1.0:
                score += 10
                reasons.append("Thời điểm học phù hợp với học kỳ hiện tại hoặc kế tiếp")
            elif semester_delta < 0:
                score -= 5
                reasons.append("Môn nằm ở giai đoạn trước trong lộ trình gợi ý")

        prerequisites = _parse_prerequisites(course["mon_tien_quyet"])
        missing = [item for item in prerequisites if item not in completed]
        if prerequisites and missing:
            prerequisite_status = "Chưa đủ điều kiện theo dữ liệu hiện có"
            prerequisite_priority = 2
            score -= 25
        elif prerequisites:
            prerequisite_status = "Đã hoàn thành các môn tiên quyết đã ghi nhận"
            prerequisite_priority = 0
            score += 5
        else:
            prerequisite_status = "Chưa ghi nhận môn tiên quyết trong dữ liệu"
            # Không suy diễn rằng môn chắc chắn không có tiên quyết.
            prerequisite_priority = 1

        verification = _clean_text(course["trang_thai_xac_minh"])
        if verification == "ten_bo_sung_chua_xac_minh":
            score -= 15
            reasons.append("Môn bổ sung chưa được xác minh theo lộ trình")

        recommendations.append({
            **course.to_dict(),
            "diem_goi_y": max(score, 0),
            "ly_do_goi_y": "; ".join(reasons) if reasons else "Chưa có đủ thông tin để cá nhân hóa",
            "trang_thai_tien_quyet": prerequisite_status,
            "mon_tien_quyet_thieu": ", ".join(missing),
            "_uu_tien_tien_quyet": prerequisite_priority,
            "_khop_muc_tieu": bool(goal_topic and category == goal_topic),
            "_vi_tri_hoc_ky": course_position if course_position is not None else 999,
        })

    result = pd.DataFrame(recommendations)
    if result.empty:
        return result

    # Môn thiếu tiên quyết xếp sau; trong mỗi nhóm ưu tiên điểm cao hơn.
    result = result.sort_values(
        by=["_uu_tien_tien_quyet", "diem_goi_y", "_khop_muc_tieu", "_vi_tri_hoc_ky", "ten_mon"],
        ascending=[True, False, False, True, True],
        kind="stable"
    )
    result = result.drop(columns=["_uu_tien_tien_quyet", "_khop_muc_tieu", "_vi_tri_hoc_ky"])
    try:
        limit = max(1, int(top_n))
    except (TypeError, ValueError):
        limit = 8
    return result.head(limit).reset_index(drop=True)


if __name__ == "__main__":
    courses, topics = load_data()
    model = train_topic_model(topics)

    print(f"Đã đọc {len(courses)} môn học.")
    print(f"Đã đọc {len(topics)} mô tả chủ đề thuộc {topics['nhom_chu_de'].nunique()} nhóm.")
    print("\nThử phân loại:")
    for sample in [
        "Tôi muốn học máy và phân tích dữ liệu",
        "Tôi muốn làm web frontend",
        "abc xyz qqq",
        ""
    ]:
        print(f"- {sample or '[để trống]'}: {classify_topic(sample, model)}")

    print("\nMột số môn được gợi ý:")
    result = get_recommendations(
        interests=["AI_học_máy", "dữ_liệu"],
        goal_text="Tôi muốn học máy và phân tích dữ liệu",
        knowledge_level="co_ban",
        current_semester=3,
        top_n=5,
        courses=courses,
        model=model
    )
    columns = ["ten_mon", "nhom_chu_de", "diem_goi_y", "trang_thai_tien_quyet"]
    print(result[columns].to_string(index=False))
