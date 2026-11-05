# Khả năng sử dụng MBD-mini cho bài toán propensity

Ngày kiểm tra dữ liệu: 03/11/2026. Phạm vi: MBD-mini trong `data/MBD-mini/`. Đây là kiểm tra dữ liệu và thiết kế bài toán; chưa huấn luyện mô hình.

## Bài toán và đầu ra

Dự đoán khả năng khách hàng mua từng sản phẩm trong tháng đích từ lịch sử trước khoảng nhãn. Nhãn có sẵn là `target_1`–`target_4`, mỗi cột 0/1. Đầu ra mô hình là bốn xác suất `p_target_1`–`p_target_4`; khách hàng có thể mua nhiều sản phẩm, nên tổng bốn xác suất không bắt buộc bằng 1. Tên nghiệp vụ của bốn sản phẩm chưa được xác định.

## Bằng chứng từ dữ liệu

### MBD-mini

- `client_split`: **100.224 khách hàng**, mỗi ID một fold; không có khách xung đột fold.
- `targets`: **1.202.688 dòng**, đủ **12 tháng × 100.224 khách hàng**, từ `2022-02-28` đến `2023-01-31`. Không trùng khóa khách hàng–tháng, không thiếu nhãn; toàn bộ khách và fold khớp `client_split`.
- `detail/trx`: **38.764.930 sự kiện**, thời gian lưu trong file từ **2020-12-31 đến 2022-12-31**.
- `detail/geo`: **66.295.724 sự kiện**, từ **2021-12-11 đến 2022-12-31**.
- `detail/dialog`: **217.205 sự kiện**, từ **2021-02-10 đến 2022-12-31**. Timestamp của ba bảng không có null; chưa xác nhận timezone của timestamp không kèm múi giờ.
- Các bảng `ptls/trx`, `ptls/geo`, `ptls/dialog` lần lượt có **98.721**, **72.573**, **46.320** dòng chuỗi. Nguồn mô tả `ptls` là biểu diễn chuỗi của dữ liệu; không cộng số dòng này với `detail` thành số sự kiện mới. Chưa đối chiếu tương đương từng sự kiện hoặc từ điển mã hóa giữa hai dạng.

| Nhãn | Số dòng dương | Tỷ lệ trên khách hàng–tháng |
|---|---:|---:|
| target_1 | 5.362 | 0,4458% |
| target_2 | 577 | 0,0480% |
| target_3 | 4.530 | 0,3767% |
| target_4 | 3.060 | 0,2544% |

Điều kiện sử dụng:

1. Nhãn mất cân bằng mạnh. Riêng `target_2` có rất ít mẫu dương theo tháng và theo fold, nên kết quả có thể dao động. Không dùng accuracy làm tiêu chí chính.
2. Đã đối chiếu ý nghĩa `mon` với bài báo MBD và EventGPT: quy ước triển khai là **tháng lịch chứa `mon` là tháng nhãn**, cutoff là ngày đầu tháng đó và chỉ lấy sự kiện có `event_time < cutoff`. Ví dụ `mon=2022-02-28` tương ứng nhãn tháng 02/2022, cutoff `2022-02-01 00:00:00`; không dịch `mon` thêm một tháng. Xem [bằng chứng và giới hạn xác minh](mon_research.md). Múi giờ của timestamp nguồn và quy trình tạo nhãn chưa được xác nhận trực tiếp từ nhà phát hành.
3. Chưa xác minh `trans_count` và `diff_trans_date` trong bảng targets được tính đến thời điểm nào; ban đầu chỉ lấy ID, tháng, fold và nhãn từ bảng này, tự tạo feature từ lịch sử sự kiện. `diff_trans_date` có **120.786 giá trị thiếu**.
4. Timeline chung đủ để thử lịch sử 84 ngày, nhưng không bảo đảm từng khách có đủ lịch sử hoặc đủ cả ba nguồn. Điều kiện cohort phải được kiểm tra tại từng cutoff, chỉ từ phần quá khứ.

Mô tả nhãn được đối chiếu với [dataset card MBD-mini](https://huggingface.co/datasets/ai-lab/MBD-mini) và [paper MBD, mục 2.1 và 3.2.1](https://arxiv.org/html/2409.17587v1). Các số đếm trong báo cáo này lấy từ bản mini đang có trên máy, không lấy quy mô bộ MBD đầy đủ trong paper.

## Phạm vi kiểm chứng và cách chạy lại

Kiểm tra footer toàn bộ 3.820 file Parquet; đọc toàn bộ timestamp của ba bảng detail vì footer INT96 không chứa min/max; đọc đầy đủ targets và client_split để kiểm tra nhãn, khóa và fold. Chưa quét toàn bộ giá trị thuộc tính sự kiện, trùng log hay lịch sử từng người. Không thay đổi dữ liệu nguồn.

```powershell
.\.venv\Scripts\python.exe -X utf8 src/audit_propensity_feasibility.py mbd
```

Kết quả: [mbd_summary.json](mbd_summary.json), [mbd_monthly.csv](mbd_monthly.csv).
