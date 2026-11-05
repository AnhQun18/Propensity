# Kiểm kê output MBD-mini sau bổ sung — 06/11/2026

Đã đưa export từ `docs.zip` vào đúng `docs/mbd_mini_eda`, đối chiếu với raw cục bộ,
chạy bổ sung, sửa biểu đồ và xuất lại kết quả cuối. Chỉ lấy subtree export MBD-mini
trong ZIP, không ghi đè các tài liệu nghiên cứu khác đi kèm.

Kết quả cuối: **34 bảng CSV, 18 biểu đồ PNG**, report, metadata và manifest.
So với export gốc: thêm **8 bảng và 5 biểu đồ**, đồng thời thay những bảng/hình
cần sửa. Xem [báo cáo kết quả](mbd_mini_eda/report.md),
[chi tiết 263 kiểm tra](mbd_mini_eda_validation.json) và
[hướng dẫn chạy đầy đủ trên máy khác](run_complete_other_machine.md).

## Đối chiếu với khung 7 bước

Tên CSV trong bảng dưới nằm tại `mbd_mini_eda/tables/descriptive.<mục>/`.
Các bảng đếm/phân phối là bảng tổng hợp, không xuất danh sách client_id từng khách.

| Mục trong khung | Bằng chứng đầu ra cuối | Kết luận và phạm vi |
|---|---|---|
| 1. Cấu trúc và quy mô | `structure/table_inventory.csv`, `column_roles.csv`, `relationship_summary.csv`; sơ đồ quan hệ trong report; quy mô khách–tháng trong bảng targets | Đủ danh sách 5 bảng, 36 cột logic, kiểu, vai trò, grain, khóa nối và số dòng. Ý nghĩa mã nghiệp vụ cần từ điển bên ngoài. |
| 2. Chất lượng dữ liệu | `quality/missing_values.csv`, `invalid_values.csv`, `duplicates.csv`, `key_coverage.csv`; `targets/label_coverage.csv` | Đủ null theo cột, lỗi thời gian/amount/ID/fold, embedding, khóa tham chiếu, trùng toàn bộ. Các phép kiểm tra miền giá trị nghiệp vụ chỉ làm được khi biết miền hợp lệ. |
| 3. Phạm vi thời gian | `temporal/source_time_ranges.csv`, `monthly_activity.csv`, `history_length_summary.csv`; `monthly_activity.png`, `monthly_customers.png`, `source_timeline.png` | Đủ đầu/cuối nguồn, sự kiện/khách theo tháng, span và ngày hoạt động. Timeline có mốc nhãn, trục tháng đã sửa sang datetime. |
| 4. Giao dịch | `transactions/amount_summary.csv`, `amount_histogram.csv`, `amount_log_histogram.csv`, `transaction_outliers.csv`, `amount_by_currency.csv`, `transaction_categories.csv`, `transaction_categories_full.csv`; histogram, boxplot, quantiles và tần suất event_type | Mean/std/min/max và phân vị đều tính trên 38.764.930 amount hữu hạn; không còn dùng mẫu 5.000 cho kết quả cuối. Tần suất mọi mã được giữ trong bảng full; bảng trình bày top 20 vẫn có remaining. |
| 5. Vị trí và hội thoại | `modalities/geo_summary.csv`, `dialog_summary.csv`, `embedding_quality.csv`, `location_categories.csv`, `location_categories_full.csv`; hình mã vị trí và kích thước embedding | Đủ số sự kiện/ngày có log/vị trí khác nhau/hội thoại mỗi khách. Có kiểm tra kích thước và giá trị hữu hạn của embedding. Vị trí khác nhau dùng geohash_6 trên toàn lịch sử. |
| 6. Khách hàng | `customers/customer_activity.csv`, `modality_coverage.csv`, `customer_aggregation_checks.csv`, `first_last_event_month.csv`, `first_last_event_coverage.csv`, `source_count_distribution.csv`; hình tháng đầu/cuối và độ phủ | Đủ số sự kiện, ngày hoạt động, tháng đầu/cuối, span, khoảng cách trung bình mỗi khách, số nguồn có dữ liệu; giữ cả khách không có log. Không xuất hồ sơ cá nhân. |
| 7. Nhãn | `targets/target_prevalence.csv`, `target_by_month_fold.csv`, `customer_positive_history.csv`, `positive_labels_per_row.csv`, `target_cooccurrence.csv`, `label_coverage.csv`; hình tỷ lệ dương và theo tháng | Đủ dương/âm/tỷ lệ từng target, 12 tháng × 5 fold × 4 target, khách từng dương và đồng xuất hiện trên cùng khách–tháng. |

**Không còn thiếu bảng hoặc biểu đồ bắt buộc trong phạm vi tính toán trên.**
Các giới hạn diễn giải và độ chi tiết được nêu ở cuối trang; chúng không được
ẩn bằng kết luận “dữ liệu hoàn toàn hợp lệ”.

## Những gì đã sửa và bổ sung

- Sửa trong package cách đọc số nguyên nullable từ Arrow: null ở một batch
  không làm mã `11` đổi thành `11.0`, đồng thời giữ chính xác int64 lớn.
- Sửa trục thời gian theo datetime, không đưa tháng cũ về cuối trục categorical.
- Tính lại tần suất đầy đủ của mọi cột categorical giao dịch/vị trí từ raw.
  Currency thực tế có **13 mã khác null**, không phải 18 biểu diễn; mã 11 có
  **38.724.240** giao dịch sau gộp đúng. Bảng full còn một dòng nhóm missing.
- Tính lại phân vị liên tục Type 7 chính xác trên toàn bộ amount hữu hạn,
  râu boxplot, outlier, histogram tuyến tính và log. Median cuối là
  **18.035,8369140625**, mean khoảng **1.304.901,2467444781**, theo đơn vị lưu trong raw.
- Bổ sung amount theo currency, phân phối tháng đầu/cuối và độ phủ tương ứng,
  số nguồn có log, độ phủ nhãn theo tháng/fold.
- Thay kiểm tra trùng trên mẫu bằng GROUP BY toàn bộ nội dung dòng, xuyên
  shard/batch, bao gồm fold; không dùng hash rút gọn thay cho nội dung.
- Tạo lại report có nhận xét số liệu, phạm vi và liên kết; manifest khớp đúng
  file cuối. Metadata giữ xuất xứ lần chạy cơ bản và ghi riêng lần bổ sung raw.

## Đối soát định lượng

| Nguồn | Số dòng | Kiểm tra trùng nguyên dòng |
|---|---:|---:|
| clients | 100.224 | 0 |
| targets | 1.202.688 | 0 |
| transactions | 38.764.930 | 0 |
| geo | 66.295.724 | 0 |
| dialog | 217.205 | 0 |

- Tổng sự kiện: **105.277.859**. Tổng theo tháng khớp từng nguồn; mọi nguồn
  của export gốc đều đã quét đầy đủ, không bị cắt max_batches.
- Không có ID sự kiện ngoài cohort, ID thiếu hoặc fold không khớp theo bảng
  chất lượng. Khóa khách và khóa khách–tháng không null/trùng.
- 98.721 khách có giao dịch, 72.573 có vị trí, 46.320 có hội thoại với ID/time
  hợp lệ. **468 khách không có log ở cả ba nguồn** vẫn nằm trong mẫu số 100.224.
- Số khách có 0/1/2/3 nguồn lần lượt là **468 / 17.322 / 47.010 / 35.424**.
- Histogram tuyến tính và log đều cộng đủ 38.764.930 amount hữu hạn. Tổng dòng
  theo currency khớp tổng giao dịch; mean có trọng số theo currency khớp mean toàn bộ.
- Tần suất full cộng đủ số dòng ở từng cột, distinct bằng số mã thật,
  không còn mã nguyên mang biểu diễn thập phân. Bảng location full có 752.805 dòng
  tổng hợp và khoảng 59 MB, dưới giới hạn 100 MB/file của GitHub.
- Tổng nhãn dương theo tháng/fold khớp toàn bộ: target_1=5.362,
  target_2=577, target_3=4.530, target_4=3.060. Không thiếu giá trị của 4 target;
  60 nhóm tháng/fold có đủ 4 nhãn.
- 217.205 embedding có chiều 768 và norm hữu hạn. `targets.diff_trans_date`
  thiếu 120.786 dòng (10,043%); thiếu biến phụ trợ không phải thiếu nhãn target.

263 kiểm tra tự động xác minh manifest, shape/cột, file thừa/thiếu, liên kết report,
PNG đọc được, tổng đếm/phân phối và các đối soát trên. Đã xem trực quan cả 18 hình.
Bộ kiểm thử hồi quy dùng dữ liệu tổng hợp kiểm tra thêm bất biến qua batch, timestamp
nanosecond, null/NaN trong embedding, phân vị và histogram, input không bị thay đổi,
đường dẫn output được bảo vệ và lỗi mã int/float.

## Thời gian và cách tái lập

Export gốc ghi thời gian phân tích **2.145,355 giây** (~35 phút 45 giây), không
gồm mọi thao tác xuất báo cáo. Lần bổ sung này mất **474,713 giây** (~7 phút 55 giây),
DuckDB 2.0.0, memory-limit 3GB, 2 threads, temp-limit 65GB, trên máy Windows
i3-8145U, RAM khoảng 8 GB. Đây là giới hạn cấu hình, không phải đo peak RAM/SSD.
Hai lần đo trên hai lần thực thi khác nhau; không dùng để khẳng định tốc độ máy khác.

Lệnh đã chạy sau khi giải nén export, từ gốc repo:

```powershell
.\.venv\Scripts\python.exe -u -X utf8 deliverables/mbd_mini_eda_fixes/complete_mbd_statistics.py --export docs/mbd_mini_eda --data-root data/MBD-mini --output-dir build/mbd_mini_eda_completed --work-dir build/mbd_supp_work --memory-limit 3GB --threads 2 --temp-limit 65GB
```

Sau khi kiểm tra, thay export cơ bản tại `docs/mbd_mini_eda` bằng bản hoàn chỉnh;
bản cơ bản giữ cục bộ ở `build/mbd_mini_eda_original_export`. Không đẩy raw,
thư mục tạm, venv hoặc bản export trung gian lên Git.

## Những điểm còn cần lưu ý

1. **Ý nghĩa nghiệp vụ và miền hợp lệ:** raw chứa mã đã encode; thống kê đủ
   không tự cung cấp từ điển tên giao dịch/currency/vị trí, đơn vị amount hoặc
   tên nghiệp vụ bốn sản phẩm. Không có bảng giải mã thì chưa xác nhận được mọi mã hợp lệ
   về nghiệp vụ. Client_split chỉ có ID/fold, không có hồ sơ nhân khẩu học để
   kiểm kê độ phủ các thuộc tính hồ sơ khác.
2. **Quy ước thời gian:** timezone=null, giữ timestamp naive như raw. Quy ước
   tháng nhãn/cutoff được trình bày riêng trong [nghiên cứu mon](propensity_feasibility/mon_research.md).
   Pipeline 7 bước chỉ mô tả marker `mon`, không áp dụng cửa sổ hay cutoff.
   Không suy diễn riêng từ chênh lệch ngày/tháng thành lỗi dữ liệu.
3. **Độ chi tiết khách hàng:** tháng đầu/cuối là phân phối theo tháng; khoảng
   cách là trung bình mỗi khách `(last-first)/(event_count-1)` cho khách có
   ít nhất hai sự kiện. Chưa có median/P95 của *từng khoảng cách liên tiếp*
   hoặc xuất từng timestamp/khách; nếu cần mức đó phải thêm phép tính riêng.
4. **Định nghĩa trùng:** một dòng trùng là dư thừa so với toàn bộ cột đang có;
   NaN và null được chuẩn hóa là thiếu. Không có event_id nghiệp vụ để xác nhận
   hai sự kiện khác nội dung có thực chất cùng một giao dịch hay không.
5. **Phạm vi xác minh raw:** đã đối chiếu schema, số dòng, khóa tham chiếu và
   tổng nhãn; không có hash toàn bộ raw trên máy tạo ZIP để chứng minh đồng nhất
   từng byte giữa hai máy. Các bảng gốc được kế thừa từ lần full scan đã export.
   Khi dùng một bản raw khác, tạo lại export cơ bản trước khi bổ sung.
