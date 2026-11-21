from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, session, flash, make_response
import pandas as pd
import re
import os

from recommender import load_data, train_topic_model, get_recommendations

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__)
app.secret_key = os.environ.get("SMARTCOURSE_SECRET", "change-this-secret-key-for-local-use")

# Nạp dữ liệu và huấn luyện mô hình một lần khi khởi động server.
courses, topics = load_data()
model = train_topic_model(topics)

LEVELS = {
    "co_ban": "Cơ bản",
    "trung_binh": "Trung bình",
    "kha": "Khá",
}


def current_profile():
    return {
        "interests": session.get("interests", []),
        "goal_text": session.get("goal_text", ""),
        "knowledge_level": session.get("knowledge_level", "co_ban"),
        "current_semester": session.get("current_semester"),
        "completed_courses": session.get("completed_courses", []),
        "profile_saved": session.get("profile_saved", False),
    }


def recommendations():
    profile = current_profile()
    return get_recommendations(
        interests=profile["interests"],
        goal_text=profile["goal_text"],
        completed_courses=profile["completed_courses"],
        knowledge_level=profile["knowledge_level"],
        current_semester=profile["current_semester"],
        top_n=10,
        courses=courses,
        model=model,
    )


@app.context_processor
def inject_globals():
    return {"profile": current_profile(), "level_labels": LEVELS}


@app.route("/")
def index():
    return render_template(
        "index.html",
        course_count=len(courses),
        topic_count=len(topics),
        category_count=topics["nhom_chu_de"].nunique(),
    )


@app.route("/profile", methods=["GET", "POST"])
def profile():
    course_rows = courses[courses["ma_mon"].astype(str).str.strip() != ""]
    course_options = [
        {
            "code": str(row["ma_mon"]).strip(),
            "name": str(row["ten_mon"]).strip(),
        }
        for _, row in course_rows.iterrows()
    ]
    categories = sorted(courses["nhom_chu_de"].dropna().astype(str).unique().tolist())

    if request.method == "POST":
        session["interests"] = request.form.getlist("interests")
        session["goal_text"] = request.form.get("goal_text", "").strip()
        level = request.form.get("knowledge_level", "co_ban")
        session["knowledge_level"] = level if level in LEVELS else "co_ban"

        semester = request.form.get("current_semester", "")
        session["current_semester"] = int(semester) if semester in {"1", "2", "3", "4", "5", "6"} else None

        mode = request.form.get("student_mode", "current")
        session["completed_courses"] = (
            request.form.getlist("completed_courses") if mode == "current" else []
        )
        session["profile_saved"] = True
        flash("Đã lưu hồ sơ trong phiên làm việc hiện tại.", "success")
        return redirect(url_for("recommendation_page"))

    return render_template(
        "profile.html",
        categories=categories,
        course_options=course_options,
        selected_interests=current_profile()["interests"],
        selected_courses=current_profile()["completed_courses"],
    )


@app.route("/roadmap", methods=["GET", "POST"])
def roadmap():
    if request.method == "POST":
        session["completed_courses"] = request.form.getlist("completed_courses")
        session["profile_saved"] = True
        flash("Đã cập nhật tiến độ học tập.", "success")
        return redirect(url_for("roadmap"))

    include_unverified = request.args.get("all") == "1"
    if include_unverified:
        roadmap_df = courses.copy()
    else:
        roadmap_df = courses[
            courses["trang_thai_xac_minh"].astype(str) != "ten_bo_sung_chua_xac_minh"
        ].copy()

    roadmap_df = roadmap_df.copy()
    roadmap_df["ma_mon"] = roadmap_df["ma_mon"].astype(str).str.strip()
    roadmap_df = roadmap_df[roadmap_df["ma_mon"] != ""]
    completed = set(current_profile()["completed_courses"])

    total = len(roadmap_df)
    done = sum(code in completed for code in roadmap_df["ma_mon"])
    credits_total = pd.to_numeric(roadmap_df["so_tin_chi"], errors="coerce").fillna(0).sum()
    done_rows = roadmap_df[roadmap_df["ma_mon"].isin(completed)]
    done_credits = pd.to_numeric(done_rows["so_tin_chi"], errors="coerce").fillna(0).sum()
    progress = round(done / total * 100) if total else 0

    # Chỉ dựng quan hệ tiên quyết được ghi rõ trong CSV.
    known_codes = set(roadmap_df["ma_mon"].str.upper())
    edges = []
    for _, row in roadmap_df.iterrows():
        target = str(row.get("ma_mon", "")).strip().upper()
        raw = str(row.get("mon_tien_quyet", "")).strip()
        for prerequisite in re.split(r"[,;|]", raw) if raw else []:
            source = prerequisite.strip().upper()
            if source and source in known_codes:
                edges.append({"source": source, "target": target})

    semesters = []
    for semester_name, group in roadmap_df.groupby("hoc_ky_goi_y", dropna=False):
        label = str(semester_name).strip() or "Chưa xác định học kỳ"
        rows = group.sort_values("ten_mon").to_dict("records")
        semesters.append({
            "name": label,
            "rows": rows,
            "done": sum(str(row["ma_mon"]).strip() in completed for row in rows),
            "credits": pd.to_numeric(group["so_tin_chi"], errors="coerce").fillna(0).sum(),
        })

    semesters.sort(key=lambda item: (
        int(re.search(r"\d+", item["name"]).group()) if re.search(r"\d+", item["name"]) else 99,
        1 if "hè" in item["name"].lower() else 0,
        item["name"]
    ))

    return render_template(
        "roadmap.html",
        semesters=semesters,
        course_options=roadmap_df.to_dict("records"),
        edges=edges,
        done=done,
        total=total,
        done_credits=done_credits,
        credits_total=credits_total,
        progress=progress,
        include_unverified=include_unverified,
    )


@app.route("/recommendations")
def recommendation_page():
    if not current_profile()["profile_saved"]:
        flash("Hãy tạo và lưu hồ sơ trước khi xem gợi ý cá nhân hóa.", "warning")
        return redirect(url_for("profile"))

    result = recommendations()
    rows = result.to_dict("records")
    return render_template(
        "recommendations.html",
        results=rows,
        interest_count=len(current_profile()["interests"]),
        completed_count=len(current_profile()["completed_courses"]),
    )


@app.route("/courses")
def course_explorer():
    query = request.args.get("q", "").strip()
    category = request.args.get("category", "Tất cả")
    verification = request.args.get("verification", "Tất cả")
    filtered = courses.copy()

    if query:
        searchable = filtered[["ma_mon", "ten_mon", "mo_ta", "nhom_chu_de"]].astype(str).agg(" ".join, axis=1)
        filtered = filtered[searchable.str.contains(query, case=False, regex=False)]

    if category != "Tất cả":
        filtered = filtered[filtered["nhom_chu_de"].astype(str) == category]

    if verification == "Theo lộ trình":
        filtered = filtered[filtered["trang_thai_xac_minh"].astype(str) != "ten_bo_sung_chua_xac_minh"]
    elif verification == "Môn bổ sung chưa xác minh":
        filtered = filtered[filtered["trang_thai_xac_minh"].astype(str) == "ten_bo_sung_chua_xac_minh"]

    categories = sorted(courses["nhom_chu_de"].dropna().astype(str).unique().tolist())
    return render_template(
        "courses.html",
        results=filtered.to_dict("records"),
        query=query,
        category=category,
        verification=verification,
        categories=categories,
    )


@app.route("/export")
def export_recommendations():
    if not current_profile()["profile_saved"]:
        flash("Hãy lưu hồ sơ trước khi tải danh sách gợi ý.", "warning")
        return redirect(url_for("profile"))

    result = recommendations()
    csv_data = result.to_csv(index=False).encode("utf-8-sig")
    response = make_response(csv_data)
    response.headers["Content-Disposition"] = "attachment; filename=smartcourse_goi_y.csv"
    response.headers["Content-Type"] = "text/csv; charset=utf-8"
    return response


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
