# Rà soát EDA toàn lịch sử MBD-mini

Kết quả chính: [report.md](mbd_mini_eda/report.md). Bản này thay cách trình bày
trước đó: không cắt lịch sử về năm 2022/khoảng giao nguồn, không dùng log cho
nhịp hành vi hoặc Amount, dùng bảng matrix cho tỷ lệ hành vi, zoom Y riêng
cho từng target và kiểm tra outlier theo currency/khách/ngày.

## Chạy lại

Python >=3.11, dùng venv hiện có, cài extras cần thiết:

```powershell
.\.venv\Scripts\python.exe -m pip install ".[plots,supplement]"
.\.venv\Scripts\python.exe -u -X utf8 src/review_mbd_eda.py
```

Tùy chọn đường dẫn và kích thước mẫu:

```powershell
.\.venv\Scripts\python.exe -u -X utf8 src/review_mbd_eda.py --data-root D:/Datasets/MBD-mini --cache-dir outputs/mbd_review_cache --sample-size 1000 --seed 42 --memory-limit 3GB --threads 2
```

Chạy cùng lệnh để vẽ lại. Không cần chạy lại pipeline cơ bản hoặc bổ sung cũ.
Script vẫn đọc footer Parquet để xác nhận inventory và đọc các dòng mẫu khi
xuất sampledata, nhưng dùng cache cho duplicate, tổng hợp khách–tháng/ngày,
null và phân vị đã tính. Không quảng cáo chế độ hoàn toàn offline với raw:
raw vẫn phải tồn tại. Thay seed/size chỉ thay mẫu; thống kê chính vẫn trên
toàn bộ 105.277.859 log.

## Cache được giữ

| Đường dẫn | Nội dung |
| --- | --- |
| `outputs/mbd_review_cache/review.duckdb` | Kết quả SQL hoàn tất, gồm duplicate chính xác, tổng hợp khách–tháng và các audit |
| `outputs/mbd_review_cache/source_inventory.json` | Danh sách toàn bộ shard, kích thước, mtime_ns, số dòng, schema và fingerprint |
| `outputs/mbd_review_cache/source_month_counts.npz` | `counts[customer,month,source]`, ID, tháng, fingerprint, mask ô chưa quan sát |
| `outputs/mbd_review_cache/last_run.json` | Các truy vấn dùng lại/đã chạy mới và thời gian lần gần nhất |
| `outputs/behavior_coverage/matrix.npz` | Cache cũ tổng ba nguồn/ngày hoạt động, được đối chiếu chính xác với tổng từ raw mới |
| `outputs/behavior_coverage/canva_requested_revision/daily_and_customer_2022.npz` | Cache cũ theo ngày/khách/amount riêng năm 2022, giữ để tham khảo |

`unknown_source_mask` dùng bit 2=geo, 4=dialog; bật khi số đếm bằng 0 trước
01/2022. Khôi phục ô null từ mask; `counts` vẫn là số log nguyên gốc. Không
điền 0 cho thiếu thuộc tính raw. Các phép tính dùng tất cả khách, kể cả 468
khách không có log. Với ngày ngoài mốc đầu/cuối nguồn, daily_behavior ghi null.

Cache xác nhận inventory bằng path/size/mtime/row count/schema, không hash
toàn bộ nội dung raw. Nếu raw thay đổi, chọn `--cache-dir` mới; script từ chối
ghi đè cache cũ. Mỗi truy vấn hoàn tất được commit riêng nên chạy bị ngắt có
thể tiếp tục bằng cùng lệnh. Câu SQL thay đổi sẽ chỉ tính lại kết quả liên quan.

## Mẫu và các giới hạn đã kiểm tra

[sampledata](../sampledata/README.md) được version trên Git: 1.000 khách phân
tầng, đủ lịch sử detail, nhãn và ptls liên kết. Các tầng hiếm có thể không
được lấy mẫu, và cực trị không được bảo toàn chính xác. Đây là mẫu EDA;
không dùng các tầng dựa trên toàn lịch sử để tạo split mô hình quá khứ.

`ptls/trx` ít hơn detail **2.020.020 log ở 13 khách**, chiều dài tối đa 10.000.
Do đó detail là nguồn chuẩn để giữ toàn thời gian. Geo/dialog có số phần tử
khớp theo khách. Việc không có null phần tử ptls không chứng minh detail
không có null mã thuộc tính.

## Kiểm chứng và file trung gian

```powershell
.\.venv\Scripts\python.exe -u -X utf8 -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -X utf8 src/validate_mbd_review.py
```

Kiểm thử dùng dữ liệu tổng hợp có duplicate thật, sự kiện cùng timestamp
khác thuộc tính, amount âm/đuôi lớn, null và lịch sử trước 2022. Bước xác nhận
kiểm tra manifest, tổng log, tỷ lệ, fingerprint/cache, sample và ô null; không
thay thế việc xem trực quan các hình.

Các hình log và biểu đồ tỷ lệ dạng cột cũ được bỏ khỏi export cuối. Preview,
contact sheet và bản ZIP phát sinh có thể tạo lại không cần giữ trong Git.
Raw, venv, cache và tài liệu đầu vào độc bản được giữ trên máy; chỉ dọn các
file trung gian có tên/nguồn gốc xác định, không xóa theo đuôi trên toàn repo.
