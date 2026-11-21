from pathlib import Path
import streamlit as st
import pandas as pd

from recommender import load_data, train_topic_model, get_recommendations

# -------------------- Cau hinh trang --------------------
st.set_page_config(
    page_title="SmartCourse | Your Learning Path",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent

# Giao dien dark futuristic. Co the tach CSS nay sang assets/styles.css sau.
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --sc-bg: #080B14;
        --sc-panel: #12192B;
        --sc-panel-2: #0D1322;
        --sc-blue: #527BFF;
        --sc-purple: #A78BFA;
        --sc-text: #F3F5FF;
        --sc-muted: #A5AEC5;
        --sc-border: rgba(130, 153, 255, 0.18);
    }
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background:
          radial-gradient(ellipse at 10% 0%, rgba(82,123,255,.16), transparent 34%),
          radial-gradient(ellipse at 90% 8%, rgba(167,139,250,.12), transparent 30%),
          var(--sc-bg);
        color: var(--sc-text);
    }
    [data-testid="stHeader"] { background: rgba(8,11,20,.72); }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0D1322 0%, #090D18 100%);
        border-right: 1px solid var(--sc-border);
    }
    [data-testid="stSidebar"] * { color: var(--sc-text); }
    h1, h2, h3 { letter-spacing: -0.035em; color: var(--sc-text); }
    p, label, .stMarkdown { color: var(--sc-text); }
    .sc-eyebrow {
        color: #9DB3FF; text-transform: uppercase; letter-spacing: .18em;
        font-size: .76rem; font-weight: 800;
    }
    .sc-hero {
        padding: 2.4rem 2.3rem;
        border: 1px solid rgba(130,153,255,.25);
        border-radius: 26px;
        background: linear-gradient(120deg, rgba(82,123,255,.17), rgba(167,139,250,.09) 55%, rgba(18,25,43,.88));
        box-shadow: 0 0 55px rgba(82,123,255,.08), inset 0 1px rgba(255,255,255,.04);
        animation: sc-fade-up .7s ease both;
    }
    .sc-title {
        font-size: clamp(2.5rem, 5vw, 4.5rem); line-height: 1.02;
        font-weight: 800; margin: .7rem 0 1rem;
        background: linear-gradient(95deg, #FFFFFF 10%, #AFC2FF 55%, #C9AFFF 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .sc-subtitle { color: #BBC5DD; font-size: 1.08rem; max-width: 680px; line-height: 1.8; }
    .sc-card {
        background: linear-gradient(145deg, rgba(18,25,43,.96), rgba(13,19,34,.96));
        border: 1px solid var(--sc-border); border-radius: 18px;
        padding: 1.2rem 1.25rem; margin-bottom: .8rem;
        transition: transform .22s ease, border-color .22s ease, box-shadow .22s ease;
        animation: sc-fade-up .5s ease both;
    }
    .sc-card:hover {
        transform: translateY(-3px); border-color: rgba(130,153,255,.58);
        box-shadow: 0 10px 32px rgba(82,123,255,.10);
    }
    .sc-tag {
        display: inline-block; border: 1px solid rgba(130,153,255,.28);
        border-radius: 999px; padding: .25rem .65rem; color: #B8C7FF;
        background: rgba(82,123,255,.10); font-size: .75rem; margin: .15rem .25rem .15rem 0;
    }
    .sc-muted { color: var(--sc-muted); font-size: .9rem; line-height: 1.6; }
    .sc-score { color: #B6C5FF; font-weight: 800; font-size: 1.45rem; }
    .sc-section { margin-top: 1.2rem; margin-bottom: .6rem; }
    div.stButton > button, div.stDownloadButton > button {
        border: 1px solid rgba(130,153,255,.42);
        border-radius: 12px; font-weight: 700;
        background: linear-gradient(100deg, #527BFF, #8265F5);
        color: white; transition: all .2s ease;
    }
    div.stButton > button:hover, div.stDownloadButton > button:hover {
        border-color: #C4B5FD; box-shadow: 0 0 24px rgba(130,101,245,.28);
        transform: translateY(-1px); color: white;
    }
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div,
    div[data-baseweb="textarea"] > div {
        background-color: #0D1322; border-color: var(--sc-border);
    }
    [data-testid="stMetric"] {
        background: rgba(18,25,43,.78); border: 1px solid var(--sc-border);
        padding: 1rem; border-radius: 16px;
    }
    hr { border-color: var(--sc-border); }
    @keyframes sc-fade-up {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after { animation: none !important; transition: none !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------- Doc du lieu --------------------
@st.cache_data
def get_course_data():
    courses, topics = load_data()
    return courses, topics

@st.cache_resource
def get_model():
    _, topics = load_data()
    return train_topic_model(topics)

try:
    courses, topics = get_course_data()
    model = get_model()
except Exception as error:
    st.error("Không đọc được dữ liệu SmartCourse. Hãy kiểm tra hai file CSV trong thư mục data.")
    st.code(str(error))
    st.stop()

if "completed_courses" not in st.session_state:
    st.session_state.completed_courses = []
if "interests" not in st.session_state:
    st.session_state.interests = []
if "goal_text" not in st.session_state:
    st.session_state.goal_text = ""
if "knowledge_level" not in st.session_state:
    st.session_state.knowledge_level = "co_ban"
if "student_mode" not in st.session_state:
    st.session_state.student_mode = "Sinh viên hiện tại"

# -------------------- Thanh dieu huong --------------------
with st.sidebar:
    st.markdown(
        '<div style="padding:.5rem 0 1.3rem">'
        '<div style="font-size:1.55rem;font-weight:800;letter-spacing:-.04em">✦ Smart<span style="color:#8EA6FF">Course</span></div>'
        '<div class="sc-muted">Your Future. Your Learning Path.</div></div>',
        unsafe_allow_html=True,
    )
    page = st.radio(
        "KHÁM PHÁ",
        ["Trang chủ", "Tạo lộ trình", "Gợi ý môn học", "Khám phá môn học"],
        label_visibility="visible",
    )
    st.markdown("---")
    st.markdown('<div class="sc-muted">BẢN THỬ NGHIỆM · VĂN LANG UNIVERSITY</div>', unsafe_allow_html=True)

# -------------------- Cac ham hien thi --------------------
def show_course_card(row, index=None):
    score = row.get("diem_goi_y", None)
    score_html = f'<div class="sc-score">{score:.0f}<span style="font-size:.8rem;color:#A5AEC5"> / 100 điểm</span></div>' if score is not None else ""
    code = str(row.get("ma_mon", "")).strip()
    name = str(row.get("ten_mon", "Môn học"))
    category = str(row.get("nhom_chu_de", "Chưa phân loại"))
    semester = str(row.get("hoc_ky_goi_y", "")).strip()
    credits = str(row.get("so_tin_chi", "")).strip()
    description = str(row.get("mo_ta", "")).strip()
    prereq_status = str(row.get("trang_thai_tien_quyet", "")).strip()
    missing = str(row.get("mon_tien_quyet_thieu", "")).strip()
    reasons = str(row.get("ly_do_goi_y", "")).strip()
    status = str(row.get("trang_thai_xac_minh", "")).strip()
    if status == "ten_bo_sung_chua_xac_minh":
        status_html = '<span class="sc-tag">Chưa xác minh theo lộ trình</span>'
    else:
        status_html = '<span class="sc-tag">Theo dữ liệu lộ trình</span>'
    st.markdown(
        f"""
        <div class="sc-card">
          <div style="display:flex;justify-content:space-between;gap:1rem;align-items:flex-start;flex-wrap:wrap">
            <div style="flex:1;min-width:220px">
              <div class="sc-eyebrow">{code if code else "MÔN BỔ SUNG"}</div>
              <h3 style="margin:.45rem 0 .5rem">{name}</h3>
              <div><span class="sc-tag">{category}</span><span class="sc-tag">{credits + " tín chỉ" if credits else "Chưa có tín chỉ"}</span><span class="sc-tag">{semester if semester else "Chưa xác định học kỳ"}</span></div>
            </div>
            <div>{score_html}</div>
          </div>
          <p class="sc-muted" style="margin:.9rem 0">{description}</p>
          {status_html}
          {f'<p class="sc-muted"><b>Tiên quyết:</b> {prereq_status}</p>' if prereq_status else ""}
          {f'<p class="sc-muted"><b>Môn tiên quyết còn thiếu:</b> {missing}</p>' if missing else ""}
          {f'<p class="sc-muted"><b>Vì sao được gợi ý:</b> {reasons}</p>' if reasons else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )

def make_recommendations():
    return get_recommendations(
        interests=st.session_state.interests,
        goal_text=st.session_state.goal_text,
        completed_courses=st.session_state.completed_courses,
        knowledge_level=st.session_state.knowledge_level,
        current_semester=st.session_state.get("current_semester"),
        top_n=10,
        courses=courses,
        model=model,
    )

# -------------------- Trang chu --------------------
if page == "Trang chủ":
    st.markdown(
        """
        <div class="sc-hero">
          <div class="sc-eyebrow">AI-POWERED LEARNING COMPANION</div>
          <div class="sc-title">Your Future.<br>Your Learning Path.</div>
          <div class="sc-subtitle">
            Khám phá những môn học phù hợp với sở thích, mục tiêu và kiến thức hiện tại.
            Từng bước xây dựng lộ trình học tập của riêng bạn cùng SmartCourse.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    left, right = st.columns([1.15, 1])
    with left:
        st.markdown("## Học có định hướng. Tiến xa hơn.")
        st.markdown(
            "SmartCourse kết hợp phân loại chủ đề bằng Naive Bayes và bộ quy tắc gợi ý "
            "để giúp sinh viên CNTT khám phá môn học, nhận biết kiến thức nền và xem xét "
            "các môn tiên quyết đã được ghi nhận."
        )
        if st.button("Bắt đầu xây dựng lộ trình  →", use_container_width=True):
            st.session_state["go_to_path"] = True
            st.info("Chọn mục **Tạo lộ trình** ở thanh điều hướng bên trái để tiếp tục.")
    with right:
        st.markdown(
            """
            <div class="sc-card">
              <div class="sc-eyebrow">YOUR LEARNING SPACE</div>
              <h3>✦ Cá nhân hóa</h3><p class="sc-muted">Chọn chủ đề và mục tiêu học tập của bạn.</p>
              <hr>
              <h3>⌘ Gợi ý thông minh</h3><p class="sc-muted">Xem điểm phù hợp cùng lý do đề xuất.</p>
              <hr>
              <h3>⌁ Lộ trình rõ ràng</h3><p class="sc-muted">Theo dõi môn đã học và thông tin tiên quyết.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("### SmartCourse hiện có")
    c1, c2, c3 = st.columns(3)
    c1.metric("Môn học trong dữ liệu", len(courses))
    c2.metric("Mẫu phân loại chủ đề", len(topics))
    c3.metric("Nhóm chủ đề", topics["nhom_chu_de"].nunique())

# -------------------- Tao lo trinh --------------------
elif page == "Tạo lộ trình":
    st.markdown('<div class="sc-eyebrow">PERSONAL LEARNING PROFILE</div>', unsafe_allow_html=True)
    st.title("Tạo lộ trình của bạn")
    st.markdown("Cung cấp một vài thông tin để SmartCourse đề xuất môn học sát với nhu cầu hơn.")
    with st.form("profile_form"):
        mode = st.radio(
            "Bạn đang ở giai đoạn nào?",
            ["Sinh viên mới / đang tìm hiểu", "Sinh viên hiện tại"],
            horizontal=True,
            index=1 if st.session_state.student_mode == "Sinh viên hiện tại" else 0,
        )
        interest_options = sorted(courses["nhom_chu_de"].dropna().astype(str).unique().tolist())
        interests = st.multiselect(
            "Chủ đề bạn quan tâm",
            options=interest_options,
            default=[item for item in st.session_state.interests if item in interest_options],
            help="Có thể chọn nhiều chủ đề. Nhãn chủ đề đang dùng trong bản thử nghiệm.",
        )
        goal_text = st.text_area(
            "Mục tiêu học tập",
            value=st.session_state.goal_text,
            placeholder="Ví dụ: Mình muốn học máy, phân tích dữ liệu và xây dựng mô hình AI...",
            height=100,
        )
        knowledge_label = st.select_slider(
            "Mức kiến thức hiện tại",
            options=["Cơ bản", "Trung bình", "Khá"],
            value={"co_ban": "Cơ bản", "trung_binh": "Trung bình", "kha": "Khá"}.get(st.session_state.knowledge_level, "Cơ bản"),
        )
        semester = st.selectbox("Học kỳ hiện tại (nếu đã biết)", ["Chưa chọn", "1", "2", "3", "4", "5", "6"])
        course_choices = courses[courses["ma_mon"].astype(str).str.strip() != ""]
        completed_labels = {
            f'{row["ma_mon"]} — {row["ten_mon"]}': str(row["ma_mon"]).strip()
            for _, row in course_choices.iterrows()
        }
        completed_default = [
            label for label, code in completed_labels.items()
            if code in st.session_state.completed_courses
        ]
        if mode == "Sinh viên hiện tại":
            completed_selected = st.multiselect(
                "Các môn bạn đã hoàn thành",
                options=list(completed_labels.keys()),
                default=completed_default,
            )
        else:
            completed_selected = []
        submitted = st.form_submit_button("Tạo gợi ý cho tôi  ✦", use_container_width=True)

    if submitted:
        st.session_state.student_mode = "Sinh viên hiện tại" if mode == "Sinh viên hiện tại" else "Sinh viên mới"
        st.session_state.interests = interests
        st.session_state.goal_text = goal_text
        st.session_state.knowledge_level = {
            "Cơ bản": "co_ban", "Trung bình": "trung_binh", "Khá": "kha"
        }[knowledge_label]
        st.session_state.current_semester = int(semester) if semester != "Chưa chọn" else None
        st.session_state.completed_courses = [
            completed_labels[label] for label in completed_selected
        ]
        st.session_state.profile_saved = True
        st.success("Đã lưu thông tin trong phiên hiện tại. Mở mục **Gợi ý môn học** để xem kết quả.")

    if st.session_state.get("profile_saved"):
        st.markdown("### Tóm tắt hồ sơ hiện tại")
        st.write("**Chủ đề quan tâm:**", ", ".join(st.session_state.interests) if st.session_state.interests else "Chưa chọn")
        st.write("**Mục tiêu:**", st.session_state.goal_text or "Chưa nhập")
        st.write("**Mức kiến thức:**", {"co_ban": "Cơ bản", "trung_binh": "Trung bình", "kha": "Khá"}.get(st.session_state.knowledge_level))
        st.write("**Môn đã hoàn thành:**", len(st.session_state.completed_courses))

# -------------------- Goi y mon hoc --------------------
elif page == "Gợi ý môn học":
    st.markdown('<div class="sc-eyebrow">RECOMMENDATION ENGINE</div>', unsafe_allow_html=True)
    st.title("Môn học dành cho bạn")
    st.markdown("Kết quả dựa trên chủ đề quan tâm, mục tiêu, mức kiến thức và dữ liệu tiên quyết hiện có.")
    if not st.session_state.get("profile_saved"):
        st.info("Cậu hãy vào **Tạo lộ trình** và lưu hồ sơ trước để cá nhân hóa gợi ý.")
        if st.button("Dùng hồ sơ mẫu để xem thử"):
            st.session_state.interests = ["AI_học_máy", "dữ_liệu"]
            st.session_state.goal_text = "Tôi muốn học máy và phân tích dữ liệu"
            st.session_state.knowledge_level = "co_ban"
            st.session_state.completed_courses = []
            st.session_state.current_semester = None
            st.session_state.profile_saved = True
            st.rerun()
    else:
        results = make_recommendations()
        if results.empty:
            st.warning("Không tìm thấy môn phù hợp với hồ sơ hiện tại.")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Môn được hiển thị", len(results))
            m2.metric("Chủ đề đã chọn", len(st.session_state.interests))
            m3.metric("Môn đã hoàn thành", len(st.session_state.completed_courses))
            st.markdown("### Danh sách đề xuất")
            st.caption("Điểm là điểm xếp hạng theo quy tắc của bản thử nghiệm, không phải xác suất bạn sẽ thành công.")
            for i, (_, row) in enumerate(results.iterrows(), start=1):
                st.markdown(f"#### {i:02d}  ·  Đề xuất")
                show_course_card(row)

            csv = results.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "Tải danh sách gợi ý dạng CSV",
                data=csv,
                file_name="smartcourse_goi_y.csv",
                mime="text/csv",
            )

# -------------------- Kham pha mon hoc --------------------
elif page == "Khám phá môn học":
    st.markdown('<div class="sc-eyebrow">COURSE EXPLORER</div>', unsafe_allow_html=True)
    st.title("Khám phá môn học")
    st.markdown("Tìm kiếm và lọc danh mục hiện có trong dữ liệu SmartCourse.")
    search_text = st.text_input("Tìm theo tên môn, mã môn hoặc mô tả", placeholder="Ví dụ: học máy, cơ sở dữ liệu...")
    category_options = ["Tất cả"] + sorted(courses["nhom_chu_de"].dropna().astype(str).unique().tolist())
    selected_category = st.selectbox("Lọc theo nhóm chủ đề", category_options)
    verification_filter = st.selectbox(
        "Trạng thái dữ liệu",
        ["Tất cả", "Theo lộ trình", "Môn bổ sung chưa xác minh"],
    )

    filtered = courses.copy()
    if search_text.strip():
        search_lower = search_text.strip().lower()
        searchable = filtered[["ma_mon", "ten_mon", "mo_ta", "nhom_chu_de"]].astype(str).agg(" ".join, axis=1).str.lower()
        filtered = filtered[searchable.str.contains(search_lower, regex=False)]
    if selected_category != "Tất cả":
        filtered = filtered[filtered["nhom_chu_de"].astype(str) == selected_category]
    if verification_filter == "Theo lộ trình":
        filtered = filtered[filtered["trang_thai_xac_minh"].astype(str) != "ten_bo_sung_chua_xac_minh"]
    elif verification_filter == "Môn bổ sung chưa xác minh":
        filtered = filtered[filtered["trang_thai_xac_minh"].astype(str) == "ten_bo_sung_chua_xac_minh"]

    st.markdown(f"**Tìm thấy {len(filtered)} môn học**")
    if filtered.empty:
        st.warning("Không có môn nào khớp bộ lọc. Hãy thử từ khóa khác.")
    else:
        for _, row in filtered.iterrows():
            show_course_card(row)

    with st.expander("Xem bảng dữ liệu"):
        display_cols = ["ma_mon", "ten_mon", "hoc_ky_goi_y", "so_tin_chi", "nhom_chu_de", "mon_tien_quyet", "trang_thai_xac_minh"]
        st.dataframe(filtered[display_cols], use_container_width=True, hide_index=True)

st.markdown(
    '<div style="text-align:center;color:#78839E;padding:2rem 0 1rem;font-size:.78rem">'
    'SMARTCOURSE · BẢN THỬ NGHIỆM HỌC TẬP · DỮ LIỆU CẦN ĐƯỢC ĐỐI CHIẾU VỚI CHƯƠNG TRÌNH CHÍNH THỨC'
    '</div>',
    unsafe_allow_html=True,
)
