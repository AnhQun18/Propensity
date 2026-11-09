# Mã nguồn

| Vị trí | Vai trò |
| --- | --- |
| `propensity_data/` | Pipeline chuẩn bị bảng khách–tháng, quarantine, feature và protocol; xem `docs/propensity_data_preparation.md` |
| `mbd_mini_eda/` | Package 7 bước thống kê mô tả |
| `audit_*.py`, `research_*.py` | Các audit/nghiên cứu độc lập đã được tài liệu tham chiếu |
| `review_mbd_eda.py`, `mbd_eda_review_report.py`, `validate_mbd_review.py` | Rà soát và xác nhận báo cáo EDA |
| `plot_*.py`, `render_*.py`, `build_*canva*.py`, `prepare_canva_requested_charts.py` | Tạo hình và nội dung trình bày báo cáo |

Giữ các script audit ở đường dẫn hiện tại để các lệnh tái lập trong báo cáo vẫn dùng
được. Không đặt output, raw hoặc cache vào `src/`. Các lệnh cho pipeline mới được
ghi trong hướng dẫn; không chạy trên máy hiện tại theo yêu cầu người dùng.
