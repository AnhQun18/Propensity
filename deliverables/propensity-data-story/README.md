# MBD-mini: dữ liệu có đủ tín hiệu cho propensity?

[Bản tóm tắt 19 slide](index.html) · [Bản đầy đủ 23 slide](detailed.html) · tiếng Việt · 16:9.

Bản đầy đủ theo đúng A3 + B3 + C11 + D3 + E3. Bản tóm tắt giữ phần chính dưới 20 slide và chuyển các trang chi tiết sang phụ lục. Mỗi trang có một biểu đồ/sơ đồ/bảng, một dòng ý nghĩa cho mô hình, định nghĩa cách tính và nguồn. [Bảng đối chiếu 26 yêu cầu](coverage_ledger.md) chỉ rõ nội dung nằm ở trang nào và giới hạn nào thuộc dữ liệu.

HTML chứa toàn bộ CSS, SVG và JavaScript, mở trực tiếp bằng trình duyệt, không cần mạng hoặc build. Tải file HTML từ GitHub để trình chiếu; trang GitHub chỉ hiển thị mã nguồn.

- `←` / `→` / Space: chuyển slide; Home / End: đầu / cuối.
- `O`: mục lục; `S`: cửa sổ presenter; `N`: ghi chú; `F`: toàn màn hình.
- Nút **Phụ lục** mở phần phụ lục của từng bản; nút đầu thanh điều hướng đổi bản tóm tắt / đầy đủ. Hash `#/7` mở trực tiếp slide 7.
- Điện thoại hỗ trợ vuốt; in trình duyệt để xuất từng trang.

Nội dung chính: bài toán và câu hỏi → toàn cảnh bốn nguồn → target / trx / dialog / geo → đóng góp theo nguồn, phân tích lỗi và snapshot → độ phù hợp, rủi ro, việc tiếp theo. Phụ lục có R/F/M decile, chất lượng geo, kết quả từng tháng, dương giả theo nhóm và giới hạn nhãn tại cutoff bất kỳ.

Các bổ sung sau rà soát:

- Cohort cố định theo lịch sử tại 01/02/2022 và tỷ lệ dương qua 10 tháng; phân phối số tháng dương trong số người từng mua.
- Trx có hai trục với đơn vị thật, sparsity theo cửa sổ 30/84 ngày, R/F/M decile giữ ties cùng nhóm và lift riêng từng event_type.
- Dialog có lift frequency/recency/giờ/cuối tuần, với các nhóm có chồng lấn được ghi rõ. Geo có đa dạng/tập trung mã vùng và audit proxy nhiễu trên mẫu hash ID.
- **9 lần fit mới**: bỏ lần lượt trx/geo/dialog trên cùng nền 15 biến lịch sử mua và cùng full B+H đã lưu; báo ΔAP và kết quả từng tháng.
- **12 lần fit mới**: so tháng lịch, 30 ngày, 84 ngày và toàn lịch sử với cùng **31 predictor**, cùng split/cohort/config.
- Phân tích lỗi dùng đúng dự đoán **B+H 156 biến**, gồm recall, dương giả, giao danh sách với rule và nhóm thiếu nguồn.

Thống kê liên hệ dùng train 02–08/2022; thí nghiệm dùng validation 09–11/2022. Các đối chứng cố định depth 4, 200 cây, learning rate 0,04, L2 5, seed 20261110. Không đọc feature/nhãn calibration hoặc test cho thí nghiệm mới, không sửa dữ liệu nguồn, không ghi đè model cuối hoặc run baseline cũ. Nhãn toàn kỳ của biểu đồ số tháng dương chỉ phục vụ mô tả EDA.

Các đính chính so với đề cương ban đầu:

- Targets có từ 02/2022, muộn hơn trx khoảng 13 tháng; không có bằng chứng SLA nhãn trễ 6 tháng. Không coi giai đoạn chưa có nhãn là mẫu âm.
- Nhãn nhị phân theo tháng chỉ đo số tháng dương, không đo số lần mua trong tháng.
- Đã có ablation riêng từng nguồn trong bản sửa này; embedding/PCA chưa thử nên không công bố kết quả nội dung.
- Không có text, duration hoặc độ dài hội thoại; không có tọa độ vị trí thực. Giờ được suy từ timestamp theo giả định UTC; mã vùng không được diễn giải thành quãng đường.
- Với T giữa tháng, chưa thể suy nhãn mua 30 ngày tới từ nhãn nhị phân theo tháng. Đã so độ dài cửa sổ tại cutoff tháng; chưa báo metric cho một horizon sản phẩm không quan sát được.

[ANALYSIS.md](ANALYSIS.md) tổng hợp kết quả đối chứng mới. `analysis/` chứa CSV và manifest; `aggregate_data.json` giữ thống kê nền. `outline.json` / `detailed_outline.json` chứa nội dung, cách tính và nguồn từng slide. `coverage_ledger.json` ánh xạ yêu cầu; `content_verification.json` kiểm tra bảo toàn số lượng và contract; `verification.json` kiểm tra trình duyệt và SHA-256. Các file công bố không chứa ID hoặc dự đoán cá nhân; model/dự đoán cục bộ nằm trong `outputs/` đã ignore.

Tái dựng từ root (cần dữ liệu và feature cục bộ):

```powershell
.\.venv\Scripts\python.exe -B -X utf8 scripts/build_data_story.py
.\.venv\Scripts\python.exe -B -X utf8 scripts/validate_data_story.py
```

Để tính lại từ feature/cache cục bộ: chạy `scripts/analyze_data_story.py`, sau đó `scripts/run_data_story_models.py` trước bước build. Run mô hình đã complete được giữ bất biến; dùng output mới nếu muốn chạy lại. `scripts/verify_data_story.mjs NODE_MODULES_DIR` render hai deck bằng Playwright Core/Edge, kiểm tra overflow, bàn phím, mục lục, presenter, điện thoại và ghi checksum.

Thiết kế dựa trên skill **html-ppt**, template `tech-sharing` và các layout chart/table/flow/gantt, điều chỉnh nền sáng navy/teal. Runtime và base CSS được lưu trong `assets/` để tái dựng; file trình chiếu đã nhúng cả hai. Xem [LICENSE.txt](LICENSE.txt).
