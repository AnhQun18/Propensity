# Phạm vi 7 bước và phép đo tốc độ

## Thay đổi ở phiên bản 0.2.0

- Bỏ tạo đặc trưng khách hàng–tháng theo các cửa sổ lịch sử và toàn bộ bước exploration.
- Giữ bảy mục mô tả, dùng chung một lần quét sự kiện.
- Tính tổng hợp theo khách, số ngày hoạt động và số vị trí một lần rồi dùng lại.
- Lưu vị trí khác nhau theo khách trên toàn lịch sử, thay cho khách–ngày–vị trí.
- Bỏ tính missingness theo nhóm và ma trận ngày/giờ chỉ phục vụ exploration cũ.
- Bỏ xây các index ngày chỉ dùng cho truy vấn cửa sổ cũ.
- Chọn ứng viên lấy mẫu trước khi copy/concat DataFrame; chuẩn hóa embedding numpy theo mảng.
- Mặc định batch 250000 dòng, batch hội thoại tối đa 4096 dòng, cache SQLite 256 MiB.
- Báo tiến độ theo nguồn và ghi thời gian từng bước vào metadata.

## Phép đo giới hạn ngày 2026-11-05

Đo trên máy phát triển Windows, i3-8145U, RAM khoảng 8 GB. Đây không phải phép
đo trên máy i7-U/RAM 16 GB của người dùng. Hai bản đọc cùng các file dữ liệu
nguồn và cùng số dòng; không sửa dữ liệu gốc.

| Nguồn | Số dòng quét |
|---|---:|
| Giao dịch | 597479 |
| Vị trí | 792771 |
| Hội thoại | 5171 |
| Tổng | 1395421 |

Điều kiện: `max_batches_per_source=10`, `batch_size=100000`, `sample_size=5000`,
`random_seed=42`, `duplicate_mode=sample`; đọc toàn bộ bảng khách/nhãn. Không
tính thời gian xuất báo cáo/PNG. Bản cũ chạy 16 bước, giới hạn phần feature
theo nhãn ở tháng `2023-01-31`, với ba cửa sổ 30/84/180 ngày. Bản mới chạy
đúng bảy bước; giới hạn batch hội thoại 4096. Các shard hội thoại trong phần
thử đủ nhỏ nên hai bản vẫn đọc đúng cùng 5171 dòng.

| Chỉ số | Bản cũ | Bản 7 bước đã tối ưu |
|---|---:|---:|
| Thời gian phân tích | 100,57 giây | 54,62 giây |
| Peak working set tiến trình | 729,32 MiB | 812,09 MiB |
| Số bước | 16 | 7 |

Trong lần đo này, tổng thời gian giảm khoảng **45,7%**. Đây là hiệu quả gộp
của việc cắt phạm vi và tối ưu xử lý; không quy toàn bộ mức giảm cho riêng
việc bỏ bước. Cache SQLite tăng từ 32 lên 256 MiB nên mức RAM đo được tăng.

Đã đối chiếu **25 bảng mô tả chung** của hai bản: số dòng, thứ tự cột và giá trị
tương đương, sai số so sánh số thực 1e-9. Các bảng/cột mới như kiểm tra cardinality
và râu boxplot được kiểm tra riêng bằng dữ liệu tổng hợp.

Đây là một phép đo trước/sau trên prefix; cache hệ điều hành và tải máy không
được cố định. Không suy rộng tỷ lệ giảm hoặc mức RAM này cho toàn bộ 105 triệu
sự kiện. Bảy bước vẫn phải quét đầy đủ dữ liệu nguồn khi bỏ giới hạn prefix.

## Dùng trên máy RAM 16 GB

Giữ cấu hình trong [configs/mbd_mini_eda.json](../configs/mbd_mini_eda.json),
đổi `data_root` cho đúng máy. Có thể đặt `work_dir` ngoài dữ liệu gốc trên SSD.
`sqlite_cache_mb` chỉ điều khiển cache SQLite, không giới hạn tổng RAM của Python.
Để ước lượng lần chạy thật, dùng `metadata.json`: `duration_seconds` toàn lần
phân tích, thời gian mỗi nguồn trong `sources`, thời gian mỗi bước trong `sections`.

Hướng dẫn cập nhật/cài lại package: [setup trên máy khác](setup_other_machine.md).
