# MBD-mini: dữ liệu có đủ tín hiệu cho propensity?

[Mở deck HTML](index.html) · 19 slide chính + 7 slide phụ lục · tiếng Việt · 16:9.

HTML chứa toàn bộ CSS, SVG và JavaScript, mở trực tiếp bằng trình duyệt, không cần mạng hoặc build. Tải file HTML từ GitHub để trình chiếu; trang GitHub chỉ hiển thị mã nguồn.

- `←` / `→` / Space: chuyển slide; Home / End: đầu / cuối.
- `O`: mục lục; `S`: cửa sổ presenter; `N`: ghi chú; `F`: toàn màn hình.
- Nút **Phụ lục** mở slide 20; hash `#/7` mở trực tiếp slide 7.
- Điện thoại hỗ trợ vuốt; in trình duyệt để xuất từng trang.

Nội dung chính: bài toán và câu hỏi → toàn cảnh bốn nguồn → target / trx / dialog / geo → bằng chứng baseline, lỗi và snapshot → độ phù hợp, rủi ro, việc tiếp theo. Phụ lục chứa phân phối số tháng dương, cohort, sparsity, leakage, tín hiệu geo, thiết kế ablation và cách tính metric.

Số liệu mới được tính bằng DuckDB từ bảy tháng feature train (02–08/2022), gồm nhóm R/F/M, tần suất dialog và đa dạng mã geo. Persistence được tính đúng cặp tháng liền trước trong validation 09–11/2022. Phân phối số tháng dương dùng nhãn toàn kỳ chỉ để mô tả. Không đọc dự đoán test, huấn luyện lại, sửa dữ liệu nguồn hoặc thay lựa chọn model.

Các đính chính so với đề cương ban đầu:

- Targets có từ 02/2022, muộn hơn trx khoảng 13 tháng; không có bằng chứng SLA nhãn trễ 6 tháng. Không coi giai đoạn chưa có nhãn là mẫu âm.
- Nhãn nhị phân theo tháng chỉ đo số tháng dương, không đo số lần mua trong tháng.
- Chưa có ablation riêng từng nguồn hoặc thử embedding, nên không công bố ΔAP / giá trị nội dung chưa đo.
- Phân tích lỗi dùng baseline hành vi gốc 153 biến và được ghi rõ; kết quả B+H 156 biến dùng báo cáo mới nhất.

`aggregate_data.json` chứa số liệu tổng hợp mới; `outline.json` chứa nội dung, cách tính và nguồn theo slide. Không chứa ID hay dự đoán cá nhân. `verification.json` ghi kết quả kiểm tra trình duyệt.

Tái dựng từ root (cần dữ liệu và feature cục bộ):

```powershell
.\.venv\Scripts\python.exe -B -X utf8 scripts/build_data_story.py
```

Thiết kế dựa trên skill **html-ppt**, template `tech-sharing` và các layout chart/table/flow/gantt, điều chỉnh nền sáng navy/teal. Runtime và base CSS được lưu trong `assets/` để tái dựng; file trình chiếu đã nhúng cả hai. Xem [LICENSE.txt](LICENSE.txt).
