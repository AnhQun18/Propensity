# Mã nguồn

| Vị trí | Vai trò |
| --- | --- |
| `propensity_data/` | Helper dùng chung và CLI cho notebook 02 và 03 |
| `mbd_mini_eda/` | Package 7 bước thống kê mô tả |
| `audit_*.py`, `research_*.py` | Các audit/nghiên cứu độc lập đã được tài liệu tham chiếu |
| `review_mbd_eda.py`, `mbd_eda_review_report.py`, `validate_mbd_review.py` | Rà soát và xác nhận báo cáo EDA |
| `plot_*.py`, `render_*.py`, `build_*canva*.py`, `prepare_canva_requested_charts.py` | Tạo hình và nội dung trình bày báo cáo |

Giữ các script audit ở đường dẫn hiện tại để các lệnh tái lập trong báo cáo vẫn dùng
được. Không đặt output, raw hoặc cache vào `src/`. Các lệnh cho pipeline mới được
ghi trong hướng dẫn; hai notebook đã chạy raw ngày 10/11/2026. Bản sửa schema v2
đã được kiểm thử tổng hợp và kiểm tra lại output thật, chưa fit mô hình.

Trong `propensity_data/`: `config.py` cấu hình; `schema.py` định nghĩa cột và
whitelist; `storage.py` đọc/ghi; `quality.py` nhãn và chuẩn hóa sự kiện;
`features.py` đặc trưng trước cutoff; `validation.py` kiểm tra schema/giá trị/nhãn
dùng chung trước ready; `protocol.py` lịch đánh giá;
`pipeline.py` điều phối CLI. Notebook chứa đầy đủ mã và các bước chạy tách cell.
