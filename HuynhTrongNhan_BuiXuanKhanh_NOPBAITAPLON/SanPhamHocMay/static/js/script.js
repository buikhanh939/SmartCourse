const menuToggle = document.getElementById("menuToggle");
const mainNav = document.getElementById("mainNav");

if (menuToggle && mainNav) {
  menuToggle.addEventListener("click", () => {
    mainNav.classList.toggle("open");
  });
}

// Ẩn danh sách môn đã hoàn thành nếu người dùng chọn "Sinh viên mới".
document.querySelectorAll('input[name="student_mode"]').forEach((radio) => {
  radio.addEventListener("change", () => {
    const completed = document.getElementById("completedCourses");
    if (completed) {
      completed.style.display = radio.value === "new" && radio.checked ? "none" : "block";
    }
  });
});
