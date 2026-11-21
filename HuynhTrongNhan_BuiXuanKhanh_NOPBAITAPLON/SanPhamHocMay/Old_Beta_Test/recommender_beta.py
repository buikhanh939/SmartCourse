"""
SmartCourse: phân loại chủ đề và gợi ý môn học.
Mô hình Naive Bayes phân loại chủ đề; điểm gợi ý được tính bằng quy tắc.
"""

from pathlib import Path
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
COURSES_PATH = DATA_DIR / "courses.csv"
TOPICS_PATH = DATA_DIR / "topics.csv"


def load_data():
    """Đọc hai file CSV và kiểm tra các cột cần thiết."""
    if not COURSES_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {COURSES_PATH}")
    if not TOPICS_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {TOPICS_PATH}")

    courses = pd.read_csv(COURSES_PATH, encoding="utf-8-sig").fillna("")
    topics = pd.read_csv(TOPICS_PATH, encoding="utf-8-sig").fillna("")

    required_courses = {
        "ma_mon", "ten_mon", "hoc_ky_goi_y", "so_tin_chi",
        "nhom_chu_de", "mo_ta", "mon_tien_quyet",
        "kien_thuc_nen", "loai_mon", "trang_thai_xac_minh"
    }
    required_topics = {"mo_ta", "nhom_chu_de"}

    missing_courses = required_courses - set(courses.columns)
    missing_topics = required_topics - set(topics.columns)

    if missing_courses:
        raise ValueError(f"courses.csv thiếu cột: {', '.join(sorted(missing_courses))}")
    if missing_topics:
        raise ValueError(f"topics.csv thiếu cột: {', '.join(sorted(missing_topics))}")
    if topics.empty:
        raise ValueError("topics.csv chưa có dữ liệu.")

    return courses, topics


def train_topic_model(topics=None):
    """Huấn luyện mô hình TF-IDF kết hợp Multinomial Naive Bayes."""
    if topics is None:
        _, topics = load_data()

    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("naive_bayes", MultinomialNB())
    ])
    model.fit(topics["mo_ta"].astype(str), topics["nhom_chu_de"].astype(str))
    return model


def classify_topic(text, model=None):
    """Phân loại một mô tả thành nhóm chủ đề."""
    text = str(text).strip()
    if not text:
        return {"nhom_chu_de": "", "do_tin_cay": 0.0}

    if model is None:
        model = train_topic_model()

    probabilities = model.predict_proba([text])[0]
    best_index = probabilities.argmax()
    return {
        "nhom_chu_de": str(model.classes_[best_index]),
        "do_tin_cay": float(probabilities[best_index])
    }


def _parse_prerequisites(value):
    """Tách danh sách mã môn tiên quyết."""
    value = str(value).strip()
    if not value:
        return []
    return [
        code.strip().upper()
        for code in re.split(r"[,;|]", value)
        if code.strip()
    ]


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
    """
    Xếp hạng môn học theo sở thích, mục tiêu, nền tảng và học kỳ.
    Môn thiếu tiên quyết vẫn có thể hiện ra nhưng được xếp sau.
    """
    if courses is None:
        courses, _ = load_data()
    if model is None:
        model = train_topic_model()

    interests = set(interests or [])
    completed = {str(code).strip().upper() for code in (completed_courses or [])}
    goal_topic = classify_topic(goal_text, model)["nhom_chu_de"] if goal_text.strip() else ""

    recommendations = []

    for _, course in courses.iterrows():
        code = str(course["ma_mon"]).strip().upper()
        if code in completed:
            continue

        category = str(course["nhom_chu_de"]).strip()
        score = 0
        reasons = []

        if category in interests:
            score += 40
            reasons.append("Phù hợp với chủ đề quan tâm")

        if goal_topic and category == goal_topic:
            score += 30
            reasons.append("Gần với mục tiêu học tập")

        course_type = str(course["loai_mon"]).strip()
        if knowledge_level == "co_ban" and course_type == "nen_tang_AI":
            score += 20
            reasons.append("Hỗ trợ củng cố kiến thức nền")
        elif knowledge_level == "trung_binh" and course_type in {"nen_tang_AI", "mon_bo_tro"}:
            score += 12
            reasons.append("Phù hợp để bổ sung kiến thức")
        elif knowledge_level == "kha" and course_type == "chuyen_nganh_AI":
            score += 20
            reasons.append("Phù hợp với trình độ hiện tại")

        semester_text = str(course["hoc_ky_goi_y"])
        match = re.search(r"Học kỳ\s*(\d+)", semester_text, flags=re.IGNORECASE)
        course_semester = int(match.group(1)) if match else None
        if current_semester is not None and course_semester is not None:
            if course_semester <= int(current_semester) + 1:
                score += 10
                reasons.append("Thời điểm học tương đối phù hợp")

        prerequisites = _parse_prerequisites(course["mon_tien_quyet"])
        missing = [item for item in prerequisites if item not in completed]

        if prerequisites and missing:
            prerequisite_status = "Chưa đủ điều kiện theo dữ liệu hiện có"
            score -= 25
        elif prerequisites:
            prerequisite_status = "Đã hoàn thành các môn tiên quyết đã ghi nhận"
            score += 5
        else:
            prerequisite_status = "Chưa ghi nhận môn tiên quyết trong dữ liệu"

        verification = str(course["trang_thai_xac_minh"]).strip()
        if verification == "ten_bo_sung_chua_xac_minh":
            reasons.append("Môn bổ sung chưa được xác minh theo lộ trình")

        recommendations.append({
            **course.to_dict(),
            "diem_goi_y": round(max(score, 0), 1),
            "ly_do_goi_y": "; ".join(reasons) if reasons else "Chưa có đủ thông tin để cá nhân hóa",
            "trang_thai_tien_quyet": prerequisite_status,
            "mon_tien_quyet_thieu": ", ".join(missing),
            "_thieu_tien_quyet": bool(missing),
            "_khop_muc_tieu": bool(goal_topic and category == goal_topic)
        })

    result = pd.DataFrame(recommendations)
    if result.empty:
        return result

    result = result.sort_values(
        by=["_thieu_tien_quyet", "diem_goi_y", "_khop_muc_tieu", "ten_mon"],
        ascending=[True, False, False, True]
    )
    result = result.drop(columns=["_thieu_tien_quyet", "_khop_muc_tieu"])
    return result.head(max(1, int(top_n))).reset_index(drop=True)


if __name__ == "__main__":
    courses, topics = load_data()
    model = train_topic_model(topics)

    print(f"Đã đọc {len(courses)} môn học.")
    print(f"Đã đọc {len(topics)} mô tả chủ đề.")
    print("\nThử phân loại:")
    print(classify_topic("Tôi muốn học máy và phân tích dữ liệu", model))

    print("\nMột số môn được gợi ý:")
    result = get_recommendations(
        interests=["AI_học_máy", "dữ_liệu"],
        goal_text="Tôi muốn học máy và phân tích dữ liệu",
        knowledge_level="co_ban",
        top_n=5,
        courses=courses,
        model=model
    )
    print(result[["ten_mon", "nhom_chu_de", "diem_goi_y", "trang_thai_tien_quyet"]].to_string(index=False))
