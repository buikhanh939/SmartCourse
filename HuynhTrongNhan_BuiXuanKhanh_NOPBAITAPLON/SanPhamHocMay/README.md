# SmartCourse

**SmartCourse -- Your Future. Your Learning Path.** là ứng dụng web thử
nghiệm hỗ trợ sinh viên Công nghệ thông tin khám phá môn học và tham
khảo lộ trình học tập dựa trên sở thích, mục tiêu, mức kiến thức và dữ
liệu môn học hiện có.

> **Lưu ý:** SmartCourse là công cụ tham khảo trong giai đoạn thử
> nghiệm. Thông tin môn học, học kỳ và môn tiên quyết cần được đối chiếu
> với chương trình đào tạo chính thức của Văn Lang University. Ứng dụng
> không thay thế tư vấn học vụ hoặc hệ thống đăng ký học phần của
> trường.

## 1. Tính năng chính

-   **Trang chủ:** giới thiệu ứng dụng và thống kê dữ liệu hiện có.
-   **Tạo lộ trình:** nhập chủ đề quan tâm, mục tiêu học tập, mức kiến
    thức, học kỳ hiện tại và các môn đã hoàn thành.
-   **Lộ trình trực quan:** xem danh mục môn theo học kỳ, đánh dấu môn
    đã hoàn thành, theo dõi số môn/tín chỉ và phần trăm tiến độ trên
    danh mục đang hiển thị.
-   **Sơ đồ môn tiên quyết:** trực quan hóa các quan hệ tiên quyết được
    ghi rõ trong dữ liệu; ứng dụng không tự suy đoán các quan hệ còn
    thiếu.
-   **Gợi ý môn học:** xếp hạng môn dựa trên hồ sơ người dùng và hiển
    thị điểm cùng lý do gợi ý; có thể tải kết quả dạng CSV.
-   **Khám phá môn học:** tìm theo tên, mã môn hoặc mô tả; lọc theo nhóm
    chủ đề và trạng thái xác minh dữ liệu.

## 2. Công nghệ sử dụng

-   **Python:** ngôn ngữ lập trình chính.
-   **Streamlit:** xây dựng giao diện web và điều phối các thao tác của
    người dùng.
-   **Pandas:** đọc, xử lý và lọc dữ liệu CSV.
-   **scikit-learn:** xây dựng quy trình xử lý văn bản và mô hình phân
    loại chủ đề.
-   **CSS/HTML:** tạo phong cách giao diện. Trong phiên bản hiện tại,
    phần lớn CSS được khai báo trực tiếp trong `app.py`.
-   **Graphviz chart của Streamlit:** hiển thị sơ đồ quan hệ môn tiên
    quyết.

## 3. Cấu trúc dự án

SanPhamHocMay/
│
├── server.py
├── app.py
├── recommender.py
├── requirements.txt
│
├── data/
│   ├── courses.csv
│   └── topics.csv
│
├── models/
├── templates/
│   ├── index.html
│   └── ...
│
└── static/
    ├── styles.css
    └── ...

### Vai trò của từng thành phần

| Thành phần | Vai trò |
| `app.py` | Thành phần chính xây dựng giao diện bằng Streamlit, tiếp nhận thông tin người dùng, điều hướng giữa các trang chức năng và hiển thị kết quả đề xuất môn học. |
| `recommender.py` | Xử lý dữ liệu môn học, xây dựng và huấn luyện mô hình phân loại chủ đề bằng TF-IDF và Multinomial Naive Bayes, đồng thời tính điểm và xếp hạng môn học theo các quy tắc được thiết lập. |
| `data/courses.csv` | Lưu trữ dữ liệu môn học, bao gồm mã môn, tên môn, học kỳ gợi ý, số tín chỉ, nhóm chủ đề, mô tả, môn tiên quyết, kiến thức nền, loại môn và trạng thái xác minh. |
| `data/topics.csv` | Chứa các ví dụ mô tả văn bản và nhãn nhóm chủ đề tương ứng, được sử dụng làm dữ liệu huấn luyện mô hình phân loại. |
| `models/` | Thư mục lưu trữ các mô hình học máy hoặc thành phần đã huấn luyện để tái sử dụng. Trong phiên bản hiện tại, mô hình được khởi tạo khi ứng dụng chạy nên thư mục này có thể chưa chứa dữ liệu. |
| `assets/styles.css` | Thư mục chứa mã CSS dùng để tùy chỉnh giao diện, màu sắc, bố cục và kiểu hiển thị. Trong phiên bản hiện tại, phần lớn CSS vẫn được khai báo trực tiếp trong `app.py`. |
| `requirements.txt` | Chứa danh sách các thư viện và gói phụ thuộc cần thiết để cài đặt và triển khai dự án. |
| `README.md` | Tài liệu giới thiệu dự án, mô tả chức năng, cấu trúc mã nguồn, công nghệ sử dụng và hướng dẫn cài đặt, khởi chạy ứng dụng. |
| `Flask` | Framework Python hỗ trợ xây dựng ứng dụng web và API phía máy chủ. Flask không được sử dụng trong phiên bản SmartCourse hiện tại nếu hệ thống chỉ triển khai giao diện bằng Streamlit. |
| `Streamlit` | Framework xây dựng giao diện web trực tiếp bằng Python, hỗ trợ tương tác với người dùng và hiển thị kết quả xử lý của hệ thống. Đây là framework giao diện chính của SmartCourse. |

## 4. Dữ liệu đầu vào

### `courses.csv`

Mỗi dòng mô tả một môn học. Một số trường quan trọng:

-   `ma_mon`, `ten_mon`: mã và tên môn.
-   `hoc_ky_goi_y`, `so_tin_chi`: học kỳ gợi ý và số tín chỉ.
-   `nhom_chu_de`, `mo_ta`: nhóm chủ đề và mô tả môn học.
-   `mon_tien_quyet`: môn tiên quyết được ghi nhận trong dữ liệu.
-   `kien_thuc_nen`, `loai_mon`: kiến thức nền và loại môn.
-   `trang_thai_xac_minh`: trạng thái xác minh của thông tin môn học.

### `topics.csv`

Gồm hai cột chính:

-   `mo_ta`: câu mô tả chủ đề.
-   `nhom_chu_de`: nhãn chủ đề tương ứng.

Chất lượng mô hình phụ thuộc vào độ đa dạng, độ chính xác và số lượng ví
dụ trong dữ liệu huấn luyện. Dữ liệu hiện tại phù hợp cho bản thử
nghiệm, chưa đủ cơ sở để khẳng định mô hình có độ chính xác cao trên mọi
mô tả.

## 5. Thuật toán và logic xử lý

### 5.1. Phân loại chủ đề: TF-IDF + Multinomial Naive Bayes

Trong `recommender.py`, mô hình được xây dựng bằng
`scikit-learn Pipeline` gồm hai bước:

1.  **TF-IDF (`TfidfVectorizer`, n-gram 1--2):** chuyển văn bản thành
    các đặc trưng số dựa trên mức độ quan trọng của từ và cụm từ.
2.  **Multinomial Naive Bayes (`MultinomialNB`):** học mối liên hệ giữa
    các đặc trưng văn bản và nhãn nhóm chủ đề từ dữ liệu trong
    `topics.csv`.

Khi nhận mô tả mục tiêu, hệ thống chuẩn hóa văn bản, sau đó dùng mô hình
để dự đoán nhóm chủ đề. Nếu đầu vào trống, không khớp từ vựng huấn luyện
hoặc quá ngắn, hệ thống có thể yêu cầu người dùng mô tả rõ hơn thay vì
đưa ra nhãn dự đoán.

Giá trị xác suất do mô hình trả về **không đồng nghĩa với độ chính xác
đã được kiểm chứng**. Cần đánh giá trên tập dữ liệu riêng trước khi báo
cáo hiệu năng chính thức.

### 5.2. Gợi ý môn học: hệ thống tính điểm theo quy tắc

Phần xếp hạng môn học hiện tại **không phải một mô hình Machine Learning
riêng**. Hệ thống cộng/trừ điểm dựa trên các quy tắc được lập trình, ví
dụ:

-   Chủ đề môn học khớp chủ đề người dùng quan tâm: cộng điểm.
-   Nhóm chủ đề khớp với mục tiêu được mô hình phân loại: cộng điểm.
-   Loại môn phù hợp với mức kiến thức hiện tại: cộng điểm.
-   Học kỳ gợi ý phù hợp với học kỳ hiện tại hoặc kế tiếp: cộng điểm;
    môn nằm ở giai đoạn trước có thể bị trừ điểm.
-   Môn có tiên quyết được ghi nhận nhưng người dùng chưa hoàn thành: bị
    trừ điểm và xếp sau nhóm đáp ứng tiên quyết.
-   Môn bổ sung chưa xác minh theo lộ trình: bị trừ điểm.
-   Môn đã đánh dấu hoàn thành sẽ không xuất hiện trong danh sách gợi ý.

Sau khi tính điểm, kết quả được sắp xếp theo mức ưu tiên của môn tiên
quyết, điểm gợi ý và các tiêu chí phụ. **Điểm gợi ý là điểm xếp hạng
theo quy tắc, không phải xác suất thành công, xác suất phù hợp hay điểm
số do mô hình AI dự đoán.**

Các ô môn tiên quyết để trống được hiểu là **chưa ghi nhận thông tin
trong dữ liệu**, không tự động kết luận rằng môn đó không có tiên quyết.

## 6. Quy trình hoạt động

1.  `app.py` đọc `courses.csv` và `topics.csv` thông qua các hàm trong
    `recommender.py`.
2.  Hệ thống kiểm tra sự tồn tại của tệp, các cột cần thiết và dữ liệu
    cơ bản.
3.  Mô hình TF-IDF + Multinomial Naive Bayes được huấn luyện từ các ví
    dụ trong `topics.csv` khi ứng dụng cần khởi tạo mô hình. Streamlit
    cache tài nguyên mô hình trong phiên chạy để tránh huấn luyện lại
    không cần thiết ở mỗi lần tương tác.
4.  Người dùng nhập hồ sơ học tập trong trang **Tạo lộ trình**.
5.  Khi mở trang **Gợi ý môn học**, hệ thống phân loại mục tiêu nếu có,
    tính điểm theo quy tắc, sắp xếp môn học và hiển thị kết quả.
6.  Các trang **Lộ trình trực quan** và **Khám phá môn học** hỗ trợ xem,
    tìm kiếm, lọc và theo dõi dữ liệu môn học.

## 7. Cài đặt và chạy ứng dụng

### Yêu cầu

-   Python được cài đặt trên máy.
-   Terminal/PowerShell có thể chạy lệnh `python` và `pip`.
-   Các tệp dự án được đặt đúng cấu trúc thư mục.

### Bước 1: Mở thư mục dự án

Mở PowerShell hoặc terminal tại thư mục `SanPhamHocMay`, nơi chứa
`app.py` và `requirements.txt`.

### Bước 2: Tạo môi trường Python

python -m venv venv
venv\Scripts\activate 

### Bước 3: Cài đặt thư viện

python -m pip install -r requirements.txt

### Bước 4: Khởi chạy
python server.py

Khi Flask khởi động thành công, Terminal sẽ hiển thị địa chỉ tương tự:
Running on http://127.0.0.1:5000
### Bước 4: Dừng ứng dụng

Quay lại cửa sổ terminal đang chạy ứng dụng và nhấn `Ctrl + C`.

## 8. Hướng dẫn sử dụng

1.  Mở **Trang chủ** để xem phần giới thiệu và thống kê dữ liệu.
2.  Vào **Tạo lộ trình**, chọn giai đoạn học tập, chủ đề quan tâm, nhập
    mục tiêu, chọn mức kiến thức và học kỳ nếu biết.
3.  Nếu đang học đại học, chọn các môn đã hoàn thành rồi nhấn nút
    tạo/lưu hồ sơ.
4.  Mở **Gợi ý môn học** để xem danh sách môn được xếp hạng. Đọc cả
    điểm, lý do gợi ý và trạng thái tiên quyết; có thể tải kết quả thành
    CSV.
5.  Mở **Lộ trình trực quan** để đánh dấu các môn hoàn thành, theo dõi
    tiến độ danh mục và xem sơ đồ tiên quyết.
6.  Mở **Khám phá môn học** để tìm kiếm theo tên, mã môn hoặc mô tả, sau
    đó lọc theo chủ đề hoặc trạng thái xác minh.

Thông tin hồ sơ và tiến độ trong phiên bản hiện tại được quản lý bằng
trạng thái phiên của Streamlit. Không nên xem đây là hệ thống lưu hồ sơ
lâu dài giữa các phiên hoặc tài khoản.

## 9. Giới hạn hiện tại

-   Dữ liệu chủ đề còn nhỏ và có thể chưa bao quát nhiều cách diễn đạt
    của người dùng.
-   Chưa có kết quả đánh giá mô hình trên một tập kiểm thử độc lập để
    báo cáo độ chính xác khách quan.
-   Điểm gợi ý môn học được xây dựng bằng quy tắc do người phát triển
    thiết lập; chưa được tối ưu bằng đánh giá người dùng hoặc dữ liệu
    thực tế.
-   Một số môn bổ sung chưa được xác minh theo chương trình đào tạo
    chính thức.
-   Các môn tiên quyết chỉ được hiển thị và xử lý khi thông tin đã được
    ghi trong dữ liệu.
-   Tiến độ tín chỉ/môn học chỉ phản ánh danh mục đang hiển thị, không
    phải tiến độ tốt nghiệp chính thức.
-   Ứng dụng chưa phải hệ thống đăng ký học phần và không xác nhận điều
    kiện đăng ký thực tế.

## 10. Hướng phát triển

-   Đối chiếu và cập nhật dữ liệu môn học với nguồn chương trình đào tạo
    chính thức.
-   Bổ sung và đa dạng hóa dữ liệu huấn luyện cho các nhóm chủ đề.
-   Tạo tập đánh giá độc lập và báo cáo các chỉ số phù hợp cho mô hình
    phân loại.
-   Cân nhắc lưu mô hình đã huấn luyện vào `models/` nếu cần tái sử dụng
    giữa các lần chạy.
-   Bổ sung kiểm thử và cải thiện khả năng xử lý lỗi khi dữ liệu đầu vào
    không hợp lệ.

------------------------------------------------------------------------

**SmartCourse --- Bản thử nghiệm hỗ trợ khám phá môn học và tham khảo lộ
trình học tập.**
