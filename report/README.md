# Report — Đầu ra phân tích theo brief MBD-mini

Gói này chỉ chứa bảng, biểu đồ và kiểm tra do code phân tích tạo ra.

- `tables/`: kết quả theo TARGETS, TRX, DIALOG, GEO, temporal và cross-modal.
- `figures/`: 45 biểu đồ/bảng hình, mỗi mục có PNG và SVG.
- `data/`: hai bảng khách–tháng, tách snapshot T−1 và lịch sử tích lũy trước cutoff.
- `brief_audit.csv`: đối chiếu toàn bộ 45 mục ở phần 12 của brief với output tương ứng.
- `checks/validation.json`: kiểm tra số liệu, cỡ mẫu, cutoff và tính đầy đủ.
- `methods/EDA_METHODS.md`: định nghĩa, cửa sổ, CI và giới hạn.

Encoding GEO vẫn có trạng thái `audited_unverified`, vì kiểm tra dữ liệu không thể tự khôi phục quy tắc mã hóa gốc. Các biểu đồ dùng mã vùng như category. PCA embedding không xác nhận ngữ nghĩa hội thoại.

Code tái lập: `scripts/complete_report_eda.py` tại project root, chạy bằng Python của `.venv`. Các bảng EDA đã có được kiểm tra SHA-256 và fingerprint trước khi tái sử dụng. Không có baseline, ảnh chụp slide hoặc tài sản thiết kế trình bày trong gói này.
