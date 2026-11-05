# MBD-mini: lệch thời gian, dữ liệu thiếu và số khách hàng còn sử dụng được

Ngày kiểm tra: **05/11/2026**. Phạm vi: bản `data/MBD-mini/` trên máy. Quét toàn bộ **1.275 file detail**, tương ứng **105.277.859 sự kiện**; đọc toàn bộ khách hàng, nhãn và timestamp của **1.110 file ptls**. Các file nguồn không bị sửa hoặc xóa.

**Kết quả chính:** ban đầu có **100.224 client ID**. Nếu yêu cầu mỗi khách có ít nhất một sự kiện đầy đủ thuộc tính ở cả ba nguồn trong khoảng thời gian chung thì còn **27.411 ID**. Nếu thêm điều kiện chỉ sử dụng ngày lịch trọn vẹn thì còn **27.342 ID**, tức **27,28%** ban đầu, loại **72.882 ID** theo bộ điều kiện này. Khoảng lịch sử chung trọn ngày là **12/12/2021–30/12/2022**, gồm **384 ngày** theo timestamp lưu trong file.

Đây là kết quả của một chính sách lọc chặt đã nêu rõ, không phải khẳng định những khách bị loại đều có dữ liệu lỗi. Không phát sinh giao dịch/hội thoại và thiếu quá trình thu thập là hai tình huống mà các file hiện tại không cho phép phân biệt chắc chắn.

## 1. Các nguồn lệch nhau bao lâu?

| Nguồn | Timestamp nhỏ nhất | Timestamp lớn nhất | Client ID trước kiểm tra thuộc tính |
|---|---|---|---:|
| `detail/trx` | 31/12/2020 21:00:02 | 31/12/2022 20:59:58 | 98.721 |
| `detail/geo` | 11/12/2021 19:27:07 | 31/12/2022 20:59:58 | 72.573 |
| `detail/dialog` | 10/02/2021 09:36:07 | 31/12/2022 20:25:11 | 46.320 |
| `targets` | `mon=2022-02-28` | `mon=2023-01-31` | 100.224 |

Khoảng giao chính xác của ba nguồn sự kiện:

```text
[2021-12-11 19:27:07.507106, 2022-12-31 20:25:11.730797]
```

Thời lượng chênh giữa hai đầu là **385,0403 ngày**, khoảng 385 ngày và 58 phút. Nếu bỏ cả ngày đầu và ngày cuối vì chỉ được ghi nhận một phần ngày, khoảng dùng theo ngày là:

```text
[2021-12-12 00:00:00, 2022-12-31 00:00:00)
= từ ngày 12/12/2021 đến hết ngày 30/12/2022
= 384 ngày
```

Không có timezone trong schema timestamp; các mốc trên dùng đúng lịch được lưu, không tự quy đổi sang giờ Việt Nam. Việc bỏ ngày biên là quy tắc thận trọng cho bài toán theo ngày, không phải bằng chứng file bị hỏng trong các ngày đó.

Theo [nghiên cứu ý nghĩa `mon`](../propensity_feasibility/mon_research.md), quy ước triển khai là tháng chứa `mon` là tháng nhãn, với cutoff ở ngày đầu tháng và lịch sử `event_time < cutoff`. Vì vậy `mon=2023-01-31` tương ứng nhãn tháng 01/2023, cutoff `2023-01-01`; có thể dùng nhãn chính thức này với lịch sử kết thúc trong tháng 12/2022. Múi giờ nguồn vẫn chưa được xác nhận. Các cửa sổ và số đếm lịch sử trong báo cáo này giữ nguyên phạm vi đã quét.

Các file Parquet trong cùng một nguồn là các shard theo khách hàng/fold. Không lấy giao thời gian của từng shard rồi yêu cầu một khách có mặt trong mọi file. Phép giao ở đây thực hiện giữa **ba nguồn**, sau đó kiểm tra theo **client ID**.

## 2. “Đủ thông tin” được tính như thế nào?

Một sự kiện được giữ theo chính sách chặt nếu:

1. Có `client_id` thuộc bảng khách hàng và `event_time` không thiếu.
2. Mọi cột vật lý của sự kiện đều không null.
3. Với giao dịch, `amount` hữu hạn.
4. Với hội thoại, embedding có đúng 768 phần tử và mọi phần tử là số hữu hạn, không null.

Không yêu cầu có sự kiện mỗi ngày. Các mã phân loại không được tự diễn giải; giá trị mã 0 không bị coi là thiếu.

| Bước lọc liên tiếp | Client ID còn lại | Số giảm so với bước trước |
|---|---:|---:|
| Bảng khách hàng ban đầu | 100.224 | — |
| Có sự kiện với ID/timestamp ở cả ba nguồn, bất kỳ thời điểm | 35.424 | 64.800 |
| Sau kiểm tra đầy đủ thuộc tính sự kiện | 35.371 | 53 |
| Cả ba nguồn còn sự kiện trong khoảng giao chính xác | 27.411 | 7.960 |
| Chỉ giữ các ngày lịch trọn vẹn trong khoảng chung | **27.342** | **69** |

Nếu chỉ kiểm tra ID và timestamp, chưa loại giao dịch thiếu thuộc tính, thì có **27.453 ID** đủ ba nguồn trong khoảng giao. Bổ sung kiểm tra thuộc tính làm giảm 42 ID ở cùng khoảng đó.

Nếu bài toán chỉ bắt buộc **giao dịch + vị trí**, số còn lại trong khoảng giao là **71.102 ID**. Việc bắt buộc có hội thoại là nguyên nhân làm giảm nhiều khách nhất; đây là lựa chọn cohort cần cân nhắc khi xây mô hình.

## 3. Dòng nào thiếu thuộc tính, dòng nào nằm ngoài thời gian chung?

| Nguồn | Số dòng gốc | Loại do thiếu/không hợp lệ | Dòng hợp lệ ngoài khoảng chung | Dòng hợp lệ trong khoảng chung |
|---|---:|---:|---:|---:|
| Giao dịch | 38.764.930 | 342.181 | 18.109.239 | 20.313.510 |
| Vị trí | 66.295.724 | 0 | 1.708 | 66.294.016 |
| Hội thoại | 217.205 | 0 | 91.334 | 125.871 |

Ba nguồn đều không thiếu ID hoặc timestamp và không có ID nằm ngoài `client_split`. Không có số tiền NaN/vô hạn hoặc embedding sai chiều/không hữu hạn trong phần đã quét. **342.181 giao dịch bị loại là do thiếu một hoặc nhiều thuộc tính phân loại**, gồm:

| Cột | Số ô null |
|---|---:|
| `currency` | 118 |
| `src_type11`, `src_type12` | 217.361 mỗi cột |
| `dst_type11`, `dst_type12` | 243.824 mỗi cột |
| `src_type21`, `src_type22` | 42.293 mỗi cột |
| `src_type31`, `src_type32` | 78.996 mỗi cột |

Các ô null có thể nằm cùng dòng nên không cộng các số ô thành số giao dịch. Nếu mô hình xử lý được categorical missing, có thể giữ những giao dịch này; báo cáo dùng chính sách đầy đủ tất cả cột để đưa ra một phép lọc cụ thể và tái lập được.

Sau khi lọc tiếp theo **27.411 khách hàng đủ ba nguồn trong khoảng giao chính xác**, dữ liệu còn:

| Nguồn | Số dòng giữ lại |
|---|---:|
| Giao dịch | 9.267.260 |
| Vị trí | 26.708.183 |
| Hội thoại | 99.118 |
| Tổng | **36.074.561** |

Các số dòng này ứng với cohort 27.411 và khoảng giao chính xác, chưa áp dụng thêm bước bỏ hai ngày biên.

Bảng nhãn có **1.202.688 dòng** hợp lệ theo ID, tháng và bốn nhãn nhị phân; không trùng khóa khách hàng–tháng. Cột phụ `diff_trans_date` thiếu **120.786 ô**. Nếu `dropna()` tất cả cột của bảng nhãn thì chỉ còn **1.081.902 dòng**, nhưng **vẫn có đủ 100.224 client ID** xuất hiện ở ít nhất một dòng. Vì vậy phải phân biệt số dòng bị loại với số khách hàng mất hoàn toàn. Không cần loại nhãn chỉ vì cột phụ trợ này null.

## 4. Có bao nhiêu khách có lịch sử 84 ngày?

Không có một con số duy nhất nếu chưa xác định “lịch sử đủ 84 ngày”. Báo cáo tính riêng ba cách hiểu.

### 4.1. Biên sự kiện chung của từng khách dài ít nhất 84 ngày

Với mỗi client, tính:

```text
start_client = max(sự kiện đầu tiên hợp lệ của trx, geo, dialog)
end_client   = min(sự kiện cuối cùng hợp lệ của trx, geo, dialog)
span_client  = end_client - start_client
```

| Điều kiện độc lập trên biên sự kiện chung | Client ID |
|---|---:|
| Có giao thời gian không âm | 23.372 |
| Giao thời gian dương | 19.684 |
| Span ít nhất 7 ngày | 18.958 |
| Span ít nhất 30 ngày | 16.755 |
| Span ít nhất **84 ngày** | **14.193** |
| Span ít nhất 180 ngày | 10.631 |
| Span ít nhất 365 ngày | 807 |

Đây là các phép lọc theo span, không phải bước nối tiếp sau dòng 27.342 ở bảng trước. Các khách có sự kiện trong cùng khoảng chung nhưng sự kiện của các nguồn không nằm chồng lên nhau có thể không đạt điều kiện span. Một hội thoại duy nhất có thể làm span bằng 0.

**Span chỉ mô tả hai đầu thời gian của sự kiện**, không chứng minh thu thập liên tục hoặc có hoạt động trong mọi ngày ở giữa. Việc lấy sự kiện cuối tương lai để chọn cohort chỉ phù hợp mô tả dữ liệu; không áp dụng máy móc để chọn khách tại thời điểm dự đoán.

### 4.2. Có ít nhất một sự kiện mỗi nguồn trong cùng một cửa sổ 84 ngày

Chỉ dùng cửa sổ lịch sử nằm hoàn toàn trong 384 ngày chung, với quy ước `[cutoff - 84 ngày, cutoff)`:

- Cutoff sớm nhất: **06/03/2022**.
- Cutoff muộn nhất: **31/12/2022**, lấy lịch sử đến hết 30/12/2022.
- Có **301** cutoff theo ngày; đây là các cửa sổ chồng lấn, không phải 301 tập kiểm thử độc lập.
- Có **25.442 khách hàng khác nhau** đạt điều kiện ở ít nhất một cửa sổ theo ngày.
- Cửa sổ có nhiều khách nhất: **11/09–03/12/2022**, cutoff **04/12/2022**, có **11.188 khách**.
- Trong chính cửa sổ đó, nếu yêu cầu cả ba nguồn đã có sự kiện đầu tiên từ trước hoặc trong ngày bắt đầu lịch sử, còn **8.211 khách**. Điều kiện này đo lịch sử đã hình thành, vẫn không chứng minh thu thập liên tục.

Số 25.442 là hợp ID qua nhiều cutoff, không phải số khách cùng đủ điều kiện tại một cutoff. Nó có thể lớn hơn 14.193 vì tiêu chí “có sự kiện trong cửa sổ” không yêu cầu biên sự kiện từng nguồn dài 84 ngày.

Nếu thêm yêu cầu có **7 ngày tương lai** nằm trong phạm vi lịch chung để tự tạo nhãn hoạt động, cutoff cuối phải lùi về **24/12/2022**; còn **294** cutoff ngày khả thi về lịch. Đây là thiết kế nhãn hoạt động minh họa, không phải định nghĩa sẵn của bốn target sản phẩm.

### 4.3. Các mốc đầu tháng với lịch sử 84 ngày

| Cutoff | Lịch sử bắt đầu | Khách có sự kiện ở đủ ba nguồn | Khách có đủ ba nguồn đã hình thành từ đầu lịch sử |
|---|---|---:|---:|
| 01/04/2022 | 07/01/2022 | 4.673 | 2.444 |
| 01/05/2022 | 06/02/2022 | 4.811 | 2.688 |
| 01/06/2022 | 09/03/2022 | 5.351 | 3.193 |
| 01/07/2022 | 08/04/2022 | 7.296 | 4.559 |
| 01/08/2022 | 09/05/2022 | 9.389 | 6.043 |
| 01/09/2022 | 09/06/2022 | 10.880 | 7.318 |
| 01/10/2022 | 09/07/2022 | 10.967 | 7.745 |
| 01/11/2022 | 09/08/2022 | 10.907 | 7.861 |
| 01/12/2022 | 08/09/2022 | 11.130 | 8.157 |

Hợp ID của chín mốc là **22.506 khách**; giao ID, yêu cầu có sự kiện đủ ba nguồn trong cả chín cửa sổ, chỉ còn **204 khách**. Cùng một khách có thể tạo nhiều mẫu khách hàng–cutoff, nên không cộng cột số khách để suy ra số ID duy nhất.

Các mốc trên mô tả feature. Các cột đếm nhãn trong `monthly_windows.csv` ghép theo tháng lịch của `mon`, được đặt tên `same_month_marker`; phép ghép này phù hợp quy ước tháng nhãn đã nghiên cứu. Đây vẫn là thống kê cửa sổ, chưa phải tập train/test đã được tạo và kiểm tra đầy đủ.

## 5. `detail` và `ptls` có tương đương về thời gian không?

Không coi `ptls` là một nguồn độc lập để cộng với detail. Kiểm tra theo từng ID cho thấy:

| Nguồn | Sự kiện trong detail | Timestamp phần tử trong ptls | Client khác số sự kiện | Client khác timestamp đầu |
|---|---:|---:|---:|---:|
| Giao dịch | 38.764.930 | 36.744.910 | **13** | **13** |
| Vị trí | 66.295.724 | 66.295.724 | 0 | 0 |
| Hội thoại | 217.205 | 217.205 | 0 | 0 |

Timestamp đầu/cuối được so ở độ chính xác giây vì `ptls.event_time` là số nguyên giây còn detail giữ nano giây. Timestamp cuối của mọi ID khớp; tập ID giữa hai dạng cũng khớp. Riêng `ptls/trx` có tổng cộng ít hơn **2.020.020 sự kiện** so với detail, tập trung ở **13 khách hàng**. Chưa xác định nguyên nhân, nên không kết luận các ID đó bị lỗi hoặc tự xóa chúng khỏi detail.

Báo cáo lấy `detail` làm nguồn chuẩn cho toàn bộ số đếm chất lượng, khoảng chung và cửa sổ. So sánh ptls ở đây chỉ kiểm tra số sự kiện và timestamp đầu/cuối mỗi khách; chưa đối chiếu toàn bộ giá trị, thứ tự sự kiện hoặc chiều dài từng mảng feature khác.

## 6. Cách dùng kết quả

- Nếu cần một tập khách có cả ba nguồn trong cùng phạm vi ngày để tiếp tục EDA, cohort theo chính sách chặt là **27.342 ID**, trong **384 ngày** chung.
- Nếu cần lịch sử 84 ngày tại một mốc dự đoán, lọc theo **khách hàng–cutoff**. Số khách tại một mốc thấp hơn nhiều so với số ID của toàn kỳ; dùng bảng cửa sổ để chọn mốc và kiểm tra số mẫu dương theo quy ước tháng nhãn chứa `mon`, cutoff ở ngày đầu tháng.
- Nếu mục tiêu chỉ cần giao dịch + vị trí, có thể giữ cohort lớn hơn và dùng cờ thiếu hội thoại. Không có bằng chứng rằng bỏ tất cả khách không có hội thoại sẽ giúp mô hình tốt hơn.
- Không điền 0 cho những ngày ngoài phạm vi nguồn để giả lập lịch sử. Bên trong phạm vi, việc gán 0 cho “không ghi nhận sự kiện” cần được hiểu đúng là trạng thái của log, không phải bằng chứng chắc chắn về mọi hoạt động thực tế.

## 7. Tệp kết quả và chạy lại

- `summary.json`: toàn bộ số đếm, điều kiện lọc, kiểm tra ptls và thống kê cửa sổ.
- `file_time_ranges.csv`: số dòng và thời gian đầu/cuối của từng file detail.
- `monthly_windows.csv`: cửa sổ đầu tháng cho lịch sử 7, 30, 84 và 180 ngày.
- `daily_windows_7d.csv`, `daily_windows_30d.csv`, `daily_windows_84d.csv`, `daily_windows_180d.csv`: số khách tại từng cutoff ngày.

Các đầu ra chỉ chứa số liệu tổng hợp; không xuất danh sách client ID. Chưa huấn luyện mô hình hoặc tạo bản dữ liệu đã lọc mới.

```powershell
.\.venv\Scripts\python.exe -X utf8 src/audit_mbd_temporal_coverage.py
```

Đã kiểm tra bảo toàn tổng dòng qua các bước lọc, số dòng khớp audit cũ của toàn bộ ba nguồn, tổng các nhóm khách theo tổ hợp modality bằng 100.224, và độ dài/số lượng của toàn bộ cửa sổ xuất ra.
