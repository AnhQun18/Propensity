# Ma trận độ phủ hành vi khách hàng × tháng

Ma trận nhóm **nhiều nhất / gần mean / gần median / ít nhất có log / không có log**
và ma trận tỷ lệ theo tháng nằm trong [báo cáo mới](mbd_mini_eda/report.md).
Tái lập bằng [quy trình rà soát có cache](mbd_eda_review.md). Công cụ bên dưới
giữ vai trò xem chi tiết cache hợp ba nguồn; không thay thế mask null theo từng
nguồn trước 01/2022 trong bản rà soát.

## Cách tính

Mỗi ô `counts[u, m]` là **tổng số log giao dịch + vị trí + hội thoại** của khách `u` trong tháng `m`. Chỉ cộng `detail`, không cộng thêm `ptls`. Không gộp các sự kiện khác nguồn thành một giao dịch nghiệp vụ.

- Có hành vi: `counts > 0` — chỉ cần có log ở **ít nhất một** nguồn.
- Có đủ ba hành vi trong tháng: `source_mask == 7` — mỗi nguồn có ít nhất một log, không yêu cầu cùng ngày.
- `active_days` (tùy chọn): số **ngày lịch khác nhau có log ở bất kỳ nguồn nào**. Cùng một ngày xuất hiện trong cả ba nguồn vẫn chỉ đếm một ngày. Tỷ lệ theo tháng là `active_days / số ngày lịch trong tháng`.

Chính sách mặc định chỉ cần `client_id` thuộc `client_split` và timestamp hợp lệ; vẫn giữ sự kiện có thuộc tính khác null. Vì vậy đây là độ phủ **log**, không phải cohort đã lọc đầy đủ thuộc tính của audit chặt. Không có log không chứng minh nguồn bị thất thoát hoặc khách thực tế không hoạt động. Dùng lịch naive như lưu trong file, không tự đổi timezone. Các tháng biên có thể chỉ có một phần ngày/tháng.

## Chạy

Từ thư mục gốc workspace, dùng môi trường đã có `numpy`, `pandas`, `pyarrow`, `matplotlib`:

```powershell
# Toàn bộ khoảng có log, cả 3 nguồn; tùy chọn tính số ngày có log hợp nhất.
.\.venv\Scripts\python.exe -X utf8 src/plot_behavior_coverage.py --active-days

# Chỉ xem lịch sử trước cutoff 01/12/2022, không đưa log tháng 12 vào input.
.\.venv\Scripts\python.exe -X utf8 src/plot_behavior_coverage.py --start 2021-12 --end 2022-12 --output-dir outputs/behavior_coverage/before_december

# Vẽ lại từ cache, không quét raw: --start/--end không đổi được phạm vi cache.
.\.venv\Scripts\python.exe -X utf8 src/plot_behavior_coverage.py --plot-only --max-display-rows 800

# Giữ thứ tự ID đã sắp xếp thay vì nhóm khách theo mức hoạt động.
.\.venv\Scripts\python.exe -X utf8 src/plot_behavior_coverage.py --plot-only --row-order id
```

Đầu tháng `--start` được lấy vào; đầu tháng `--end` bị loại. Chẳng hạn `--end 2023-01` lấy đến hết 12/2022. Phạm vi mặc định là `[2020-12-01, 2023-01-01)` để giữ cả log ngày 31/12/2020 trong dữ liệu gốc. Muốn đổi phạm vi hoặc thêm `active_days`, chạy lại không có `--plot-only`.

## Đầu ra

### Heatmap có số trong ô, giống ảnh tham chiếu

```powershell
.\.venv\Scripts\python.exe -X utf8 src/plot_behavior_coverage.py --plot-only --annotated --sample-size 25 --sample-seed 42 --heatmap-start 2021-01 --heatmap-end 2023-01

# Xem 25 khách tiếp theo theo thứ tự ID (offset đếm từ 0).
.\.venv\Scripts\python.exe -X utf8 src/plot_behavior_coverage.py --plot-only --annotated --sample-method page --row-order id --sample-size 25 --page-start 25 --heatmap-start 2021-01
```

Xuất `heatmap_annotated.png`, `heatmap_annotated.pdf`, `heatmap_sample.csv` và
`heatmap_sample_metadata.json` vào cùng thư mục output. PNG/PDF là ma trận màu
`Greens`, thang tuyến tính từ 0 đến giá trị lớn nhất của phần đang xem, có số
nguyên trong từng ô. Không cắt các giá trị lớn hoặc đổi thành phần trăm. Khi
so sánh các ảnh khác mẫu, chú ý thang màu của từng ảnh có thể khác nhau.

Mặc định chọn ngẫu nhiên 25 khách không lặp, giữ cả khách có 0 log. Đây là mẫu
để đọc từng khách, không phải biểu đồ tổng hợp đủ 100.224 khách. Có thể chọn
1–100 khách, khuyến nghị 20–40 để số trong ô còn dễ đọc. `--sample-method page`
lấy một đoạn từ thứ tự `--row-order`, không lấy ngẫu nhiên. Mã `KH-xxxxxx` là
vị trí trong danh sách ID chuẩn đã sắp xếp, ánh xạ sang ID thật trong CSV.
Hai tham số `--heatmap-start`, `--heatmap-end` chỉ cắt phần tháng hiển thị của
heatmap từ cache, không làm đổi cache hay phạm vi overview/viewer. Tháng cuối
loại trừ; `2023-01` nghĩa là hiển thị đến hết 12/2022.

Trong `outputs/behavior_coverage/`:

| Tệp | Nội dung |
|---|---|
| `matrix.npz` | Ma trận gốc không mất khách: `counts:uint32`, `source_mask:uint8`, `active_days:uint8` nếu bật, `client_ids`, `months` |
| `overview.png` | Biểu đồ raster tổng quan; tối đa 1.000 nhóm hàng; màu là tỷ lệ khách có log trong nhóm |
| `explorer.html` | Viewer Canvas chạy offline, tổng quan và xem chính xác từng khách theo trang |
| `monthly_summary.csv` | Tổng sự kiện, số khách có log, số khách đủ ba nguồn mỗi tháng |
| `summary.json` | Phạm vi, định nghĩa, dung lượng, số dòng đọc/giữ/bỏ theo nguồn |

HTML dùng hạng khách sau sắp xếp thay vì nhúng client ID. Cache giữ ánh xạ ID để tra lại đúng khách. Hạng chi tiết và hàng tổng quan dùng cùng thứ tự; nhấn một ô tổng quan để mở nhóm tương ứng. Viewer hỗ trợ bốn lớp màu: có log, tổng log, số nguồn, tỷ lệ ngày có log (nếu đã tính). Màu của tổng log dùng thang tuyến tính; đọc số ô hoặc chọn trang/nhóm để xem khách có quy mô nhỏ.

## Tối ưu theo hai bước riêng

**Tổng hợp dữ liệu:** đọc Parquet theo batch 100.000 dòng, chỉ hai cột `client_id`, `event_time`; không load embedding 768 chiều hoặc ghép hơn 105 triệu log vào một DataFrame. Cộng ngay vào ma trận bằng `np.add.at` để các cặp khách–tháng lặp không bị mất số đếm. Bit mask ba nguồn giữ trong một byte/ô. Tính ngày hoạt động chỉ bật khi cần vì dùng thêm bitmap khách × ngày. Mỗi batch giải phóng sau khi cập nhật ma trận.

**Hiển thị:** không dùng seaborn annotation hoặc tạo hàng triệu DOM/SVG rect. PNG dùng `imshow` và HTML dùng một Canvas; tổng quan gom hàng bằng tỷ lệ có log **trên tất cả khách trong nhóm**, không lấy mẫu một số khách. Khách có ít hoạt động vẫn được giữ. Chi tiết chia trang, một hàng một khách; ma trận cache không bị gom mất thông tin.

100.224 khách × 25 tháng = 2.505.600 ô. `counts:uint32` dùng 10,02 MB; bit mask dùng 2,51 MB. Mảng đếm tích lũy dùng `uint64` (20,04 MB), kiểm tra giới hạn trước khi ép sang `uint32`. Bitmap ngày tùy chọn khoảng 76 MB ở phạm vi mặc định; ngoài ra còn bộ nhớ batch, chỉ mục ID, nén và render. Đây là dung lượng các mảng, không phải đo peak RAM toàn tiến trình.

Với hình cao 1.000 pixel, không thể phân biệt 100.224 khách riêng rẽ. Bản tổng quan nên dùng ~500–1.000 nhóm; dùng chế độ chi tiết để đọc từng khách. Sau lần tổng hợp đầu, chỉnh cách vẽ bằng cache để không đọc raw lần nữa.

## Kiểm tra

```powershell
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -p test_behavior_coverage.py -v
```

Test kiểm tra cập nhật chỉ số lặp, cùng ngày qua nhiều nguồn, cutoff cuối loại trừ, ID lạ, timestamp null và bảo toàn tất cả khách khi gom hàng. Script tự đối chiếu tổng số sự kiện giữa ma trận và audit từng nguồn. `monthly_summary.csv` có thể đối chiếu tiếp với export `descriptive.temporal/monthly_activity.csv` trên cùng chính sách ID/timestamp.
