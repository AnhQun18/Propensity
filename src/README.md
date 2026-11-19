# Mã nguồn hiện hành

| Vị trí | Vai trò |
| --- | --- |
| `propensity_data/` | Chuẩn bị dữ liệu trước cutoff, kiểm tra schema và CLI; không huấn luyện mô hình |
| `mbd_mini_eda/` | Package thống kê mô tả 7 bước |
| `propensity_eda.py` | EDA raw và đặc trưng khách–tháng |
| `audit_*.py`, `research_*.py` | Kiểm chứng cấu trúc, thời gian, nhãn, missing và giả thuyết nghiệp vụ |
| `review_mbd_eda.py`, `mbd_eda_review_report.py`, `validate_mbd_review.py` | Cache/audit đầu vào cho báo cáo hiện hành |
| `plot_behavior_coverage.py`, `behavior_coverage_viewer.html` | Tạo cache độ phủ và viewer; cache còn được audit đọc |

Các renderer/plot phiên bản cũ trùng với ảnh mới đã được xóa.
Xem [mục lục tài liệu](../docs/README.md) và [danh mục notebook](../notebooks/README.md).
Dùng notebook 03 cho chuẩn bị, notebook 04 nếu cần hợp nhất dữ liệu.
