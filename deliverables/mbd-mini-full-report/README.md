# Báo cáo phân tích MBD-mini đầy đủ

Bản báo cáo 125 trang, mở rộng từ bản Canva tóm tắt 27 trang. Nguồn số liệu được cố định tại commit `3e4b95d79d172d2ef525ff3d26e2528bce290e8e` của repository này; không quét lại hay thay đổi dữ liệu gốc khi dựng báo cáo.

- `MBD_mini_Bao_cao_day_du.pdf`: bản đọc và xuất bản, có liên kết CSV/hình nguồn.
- `MBD_mini_Bao_cao_day_du_Canva.html`: bản nhập Canva; chữ có thể sửa, nền đồ họa giữ bố cục của PDF.
- `coverage_ledger.csv` và `.json`: đối chiếu 61 bảng, số dòng, tên biến, trang phân tích, cách biểu diễn và đường dẫn nguồn; JSON có SHA-256 của từng CSV và đối chiếu 24 hình.
- `canva_content_verification.json`: kết quả đối chiếu nội dung sau nhập Canva: đủ 125 trang, 10.416 khối chữ được tìm thấy, không thiếu hoặc thừa lượt xuất hiện của giá trị số.
- `build_summary.json`: danh sách tên 125 trang.

## Nội dung

| Trang | Phạm vi |
| --- | --- |
| 1–26 | Tổng quan; cấu trúc; null; duplicate; detail/ptls; ma trận tỷ lệ và nhóm có chủ đích; nhịp hành vi; Amount tuyến tính; target zoom Y; mẫu; ngoại lai |
| 27 | Hướng dẫn tra cứu và quy ước |
| 28–30 | Quy mô B0–B6 và toàn bộ 175 ô độ phủ nhóm–tháng |
| 31–49 | 20 biến tổng hợp theo khách, phân vị, độ dài lịch sử, toàn bộ tháng hoạt động và phân bố tháng đầu/cuối |
| 50–57 | Phân phối từng cột mã và 5 mã thường gặp nhất mỗi cột |
| 58–68 | Nhãn dương/âm, từng dương, đồng xuất hiện, toàn bộ nhóm tháng–fold và số liệu target theo tháng |
| 69–86 | Từ điển 61 cột cùng null, 55 event_type, giá trị không hợp lệ, khóa, duplicate, chuỗi và vector |
| 87–102 | Đủ 24 chẩn đoán mẫu, 178 tầng kể cả tầng không nhận mẫu, trọng số và đối chiếu nhãn |
| 103–117 | Amount từng currency, phân vị, hai bộ bin, ngoại lai, phân mảnh nguồn, số nguồn có log và thời gian quét |
| 118–124 | Danh mục đối chiếu 24 hình và 61 bảng, gồm toàn bộ tên cột và liên kết |
| 125 | Kết luận và điều kiện sử dụng |

## Phạm vi của “đầy đủ”

Mọi chủ đề phân tích và mọi bảng/hình trong manifest đều có trang đối chiếu. Các bảng dài được chuyển thành ma trận hoặc thống kê; báo cáo không in tuần tự 752.805 dòng tần suất vị trí, 12.829 dòng tần suất giao dịch hay 2.193 dòng chuỗi ngày. Các CSV đầy đủ được liên kết ở phiên bản nguồn cố định. Tên biến và cách biểu diễn được ghi ở phụ lục, tránh đồng nhất “đủ phân tích” với “in mọi dòng”.

Giữ toàn bộ thời gian và khách không có log. Biểu đồ không dùng biến đổi log. Amount giữ giá trị nguồn; currency là mã ẩn danh, chưa quy đổi tiền tệ. Trong lưới khách–tháng, ô geo/dialog không có log trước 01/2022 là null có cờ nguồn sớm; các số đếm raw vẫn được giữ. Không tự loại ứng viên trùng hoặc ngoại lai.

## Tái tạo

Cần Python với `numpy`, `pandas`, `reportlab`, `pypdf`, `pdfplumber`, font Arial trên Windows và Poppler (`pdftoppm`) trong PATH. Chạy tại thư mục gốc repository:

```powershell
python deliverables/mbd-mini-full-report/build_complete.py
python deliverables/mbd-mini-full-report/build_canva_html.py
New-Item -ItemType Directory -Force deliverables/mbd-mini-full-report/graphics | Out-Null
pdftoppm -r 144 -png deliverables/mbd-mini-full-report/graphics_only.pdf deliverables/mbd-mini-full-report/graphics/page
python deliverables/mbd-mini-full-report/assemble_html.py
```

PDF/HTML được tạo ở `output/pdf/`. Dữ liệu chi tiết nguồn đã có trong `docs/mbd_mini_eda/`; quy trình này không cần truy cập raw hoặc cache quét. Cache quét hiện có vẫn được giữ cục bộ, không đưa lên Git.

Nguồn: ai-lab/MBD-mini; giấy phép CC BY 4.0 theo README của bộ dữ liệu. Báo cáo chỉ dùng số liệu tổng hợp và mã đã ẩn danh; không chứa ID khách gốc.
