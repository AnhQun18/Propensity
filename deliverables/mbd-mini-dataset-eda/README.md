# MBD-mini — Dataset Understanding & EDA

Mở `index.html` trực tiếp bằng Microsoft Edge hoặc Chrome. Bản HTML gồm **30 slide chính + 10 phụ lục**, nhúng CSS, JavaScript, SVG và toàn bộ số liệu tổng hợp; không cần mạng hoặc máy chủ.

- `←` / `→` / Space / PageUp / PageDown: chuyển slide; Home / End: đầu / cuối.
- `O`: mục lục; `F`: toàn màn hình; `S`: cửa sổ người trình bày; `N`: ghi chú.
- `#/17`: liên kết thẳng slide 17. Nút **Phụ lục** mở A1.
- **Tải số liệu** xuất JSON tổng hợp ngay cả khi chỉ giữ một file HTML.
- Nguồn dưới mỗi slide liên kết CSV trong `tables/`; giữ cả thư mục để mở các CSV này.
- In bằng trình duyệt để xuất các trang. Font Segoe UI / Arial có tiếng Việt, không tải webfont.

## Phạm vi và quy ước

Toàn bộ 100.224 khách và 12 tháng nhãn 02/2022–01/2023. Chi tiết sự kiện lấy từ `detail/*`; không cộng `ptls/*`. Mapping T là tháng chứa mon; cutoff đầu tháng T và cửa sổ `[đầu T−1, đầu T)` theo quy ước nghiên cứu UTC. Timezone ETL nhãn gốc chưa xác nhận. Recency dùng sự kiện gần nhất trước cutoff. Số đếm 0 chỉ nghĩa không tìm thấy log, không chứng minh không hoạt động thực tế.

Wilson CI cho tỷ lệ từng tháng; cluster bootstrap theo client_id với 2.000 lần, seed 42 cho tỷ lệ gộp, lift và tỷ số chuyển tiếp. Lift tính lại baseline trên cùng lượt bootstrap. Nhóm n+ <30 gắn cờ. Dữ liệu chính không được lọc theo việc có đủ ba nguồn.

Returning / first-time observed / history unknown cùng chia cho toàn bộ người mua hiện tại; tháng nhãn đầu thuộc history unknown. Không suy số lần mua thực tế từ số tháng dương. Amount không cộng khác currency; ngoại lai giữ phương pháp log1p-IQR đã đính chính. Không huấn luyện mô hình.

## Tái dựng và đối soát

Chạy từ project root bằng Python `.venv`:

```powershell
.\.venv\Scripts\python.exe -B -u -X utf8 scripts/analyze_dataset_eda.py
.\.venv\Scripts\python.exe -B -X utf8 scripts/build_dataset_eda.py
.\.venv\Scripts\python.exe -B -X utf8 scripts/validate_dataset_eda.py
node scripts/verify_dataset_eda.mjs NODE_MODULES_DIR
```

`manifest.json` lưu phiên bản đầu vào, checksum CSV, quy tắc và kiểm tra tính toán. `coverage_ledger.csv` nối slide với bảng và phương pháp. `outline.json` chứa nội dung và ghi chú. `verification.json` là kết quả kiểm tra 40 slide bằng Edge/Playwright.

Các snapshot cấp khách chỉ lưu trong `outputs/dataset_eda/`, không nằm trong gói bàn giao. Cache quét mới tách khỏi cache cũ; raw được đọc, giữ nguyên.

Nguồn dataset: https://huggingface.co/datasets/ai-lab/MBD-mini (CC BY 4.0); bài báo https://arxiv.org/abs/2409.17587. Thiết kế: HTML PPT Studio, template presenter-mode-reveal, theme corporate-clean (MIT).
