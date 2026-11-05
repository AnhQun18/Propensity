# Kiểm đếm biến và quan sát MBD-mini

Thời điểm: 2026-11-05T21:32:09.730703+07:00. Quét đủ 3,820 file Parquet; kiểm tra schema của từng file và số dòng trong footer. Đọc toàn bộ khóa khách hàng–tháng để kiểm tra số khách, tháng và trùng khóa.

| Bảng | Quan sát (dòng) | Đơn vị quan sát | Cột trong file | Cột khi gồm partition |
|---|---:|---|---:|---:|
| `client_split` | 100,224 | Khách hàng | 1 | 2 |
| `targets` | 1,202,688 | Khách hàng–tháng | 8 | 9 |
| `detail/trx` | 38,764,930 | Sự kiện giao dịch | 14 | 15 |
| `detail/geo` | 66,295,724 | Sự kiện vị trí | 5 | 6 |
| `detail/dialog` | 217,205 | Sự kiện hội thoại | 3 | 4 |
| `ptls/trx` | 98,721 | Chuỗi giao dịch của khách | 14 | 15 |
| `ptls/geo` | 72,573 | Chuỗi vị trí của khách | 5 | 6 |
| `ptls/dialog` | 46,320 | Chuỗi hội thoại của khách | 3 | 4 |

Có **100,224 khách hàng**, **12 tháng nhãn**, **1,202,688 mẫu khách hàng–tháng** và **105,277,859 sự kiện detail**.

`fold` được lưu trong đường dẫn partition, nên số cột khi đọc cả bảng nhiều hơn số cột vật lý trong file một cột. `embedding` được tính là một cột mảng; theo audit nội dung trước, mỗi vector hội thoại có 768 phần tử. Các cột mảng trong ptls cũng được tính là một cột mỗi trường, không trải phẳng thành số phần tử.

Không cộng detail và ptls thành tổng quan sát độc lập: đó là hai dạng biểu diễn của lịch sử. Cũng không cộng các bảng có đơn vị quan sát khác nhau để suy ra số mẫu train. Số cột nguồn chưa phải số feature sau tổng hợp lịch sử.

Không trùng ID trong client_split: True. Không trùng khóa khách hàng–tháng trong targets: True.

Tên cột và kiểu dữ liệu của từng bảng nằm trong `summary.json`. Chạy lại: `.\.venv\Scripts\python.exe -X utf8 src/audit_mbd_inventory.py`.
