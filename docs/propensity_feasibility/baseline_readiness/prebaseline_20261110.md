# Rà soát trước baseline sau thảo luận cửa sổ lịch sử

> Snapshot trước triển khai. Sau rà soát đã bổ sung mã calendar baseline,
> trainer/evaluator/calibrator, cấu hình và notebook 05; xem
> [thiết kế và trạng thái mới](../../propensity_baseline_design.md).

Ngày rà soát: 10/11/2026. Phạm vi: code hiện tại, cấu hình, hợp đồng feature,
protocol và output `outputs/propensity_processing_v3/`. Chỉ review và kiểm chứng;
không đổi feature, raw, lịch đánh giá, cài thư viện hoặc fit mô hình.

## Kết luận

Output hiện tại đạt kiểm tra dữ liệu cho baseline thống kê dùng 30 ngày, 84 ngày
và toàn lịch sử. Tuy nhiên nó **chưa triển khai đúng phương án vừa thảo luận**:
tháng lịch M−1 cộng lịch sử cũ trước tháng M−1. Trainer/evaluator/calibrator cũng
chưa được triển khai. Cần xử lý khoảng cách thiết kế này trước khi tuyên bố đã
đánh giá phương án tháng gần nhất + tích lũy cũ.

## 1. Các việc cần hoàn tất trước khi fit

### F01 — Feature chưa khớp phương án tháng lịch và tích lũy cũ

- `configs/propensity_processing.json` dùng `windows_days=[30,84]`.
- `features.py::window_features` dùng `[cutoff-days, cutoff)` hoặc mọi sự kiện
  trước cutoff. Cửa sổ `all` bao gồm cả tháng gần nhất.
- Schema có 221 predictor; không có nhóm riêng cho tháng lịch M−1, phần lịch sử
  cũ kết thúc tại đầu M−1, hoặc biến thay đổi so với nền.

Ví dụ target tháng 12/2022, cutoff 01/12/2022: tháng gần nhất cần là
`[01/11/2022,01/12/2022)`, lịch sử cũ cần kết thúc trước 01/11/2022. Cửa sổ
30 ngày tình cờ trùng tháng 11; các tháng dài 28 hoặc 31 ngày sẽ khác.

Nếu thực hiện phương án mới, thêm hai khoảng thời gian rõ ràng, cập nhật
schema/notebook/validation rồi tạo output mới. Thống kê median, nunique, max,
top-share và recency của lịch sử cũ phải tính lại theo đúng khoảng; không lấy
thống kê `all` trừ thống kê tháng gần nhất. Không ghi đè output đã kiểm chứng.

Đây là khoảng cách giữa thiết kế và code, không phải bằng chứng output 30/84
ngày hiện tại bị sai. Output cũ vẫn dùng được làm đối chứng riêng.

### F02 — Thiếu thành phần huấn luyện và đánh giá

Trong `src/`, `scripts/`, `tests/` chưa có trainer/evaluator/calibrator cho
baseline. `protocol.py` mới mô tả lịch và tên metric, chưa thực thi mô hình.

Cần có cấu hình xác định danh sách predictor, seed, loss, số cây hoặc quy tắc
early stopping, cách xử lý NaN, trọng số lớp, phương pháp calibration, giới hạn
RAM/threads và cách lưu model/prediction/metric. Chỉ đọc predictor đã duyệt;
giữ `client_id`, cutoff và các nhãn ở vai trò khóa/đối soát.

### F03 — Môi trường chưa có thư viện mô hình

Metadata trong `.venv` hiện không có CatBoost, scikit-learn, LightGBM, XGBoost.
`pyproject.toml` chưa có nhóm dependency cho baseline. Trước khi fit cần cài
thư viện thực sự sử dụng, kiểm tra import/fit nhỏ và lưu phiên bản cụ thể.
Không cần cài đồng thời cả ba thư viện boosting cho baseline đầu tiên.

### F04 — Chuyển protocol thành quy tắc thực thi

Protocol đã có validation cuốn chiếu 09–11/2022, final train 02–11/2022,
calibration 12/2022 và test 01/2023. Cần bảo đảm trainer thực thi đúng:

- Không dùng tháng 12/2022 để chọn feature, số cây, class weight hay loại
  calibrator; chọn các quy tắc bằng validation trước đó.
- Giữ nguyên base model sau khi fit calibrator; khóa tháng test trước khi chọn
  mô hình. Nếu tiếp tục phát triển nhiều mô hình, dùng cùng validation để so
  sánh và chỉ mở test sau khi chọn mô hình cuối.
- Bắt đầu với phân bố lớp tự nhiên; ghi rõ mọi thay đổi class weight/sampling.
- Metric top-K tính theo từng tháng, `K=ceil(0.01*N)` (1.003 với N=100.224),
  có tie-break cố định không dựa vào nhãn; không xếp hạng chung nhiều tháng.
- Lưu prediction có khóa, metric theo tháng/độ phủ và quy tắc tổng hợp metric.
  Bootstrap theo khách khi gộp nhiều tháng.
- Giữ giả định độ trễ nhãn 0 và UTC được ghi rõ. Protocol hiện từ chối lịch
  calibration không khả thi khi thêm độ trễ; đây chưa phải SLA của nguồn.

Các nguyên tắc này đã có phần lớn trong tài liệu; thiếu phần triển khai và
kiểm tra thực thi, không phải thiếu toàn bộ thiết kế đánh giá.

## 2. Phạm vi baseline đề xuất

Sau khi feature khớp phương án cần kiểm tra, so sánh trên cùng cohort/split:

1. Xác suất nền từ nhãn train để đối chiếu log loss/Brier; random ranking là
   tham chiếu top-K, với seed hoặc kỳ vọng giải tích được ghi rõ.
2. CatBoost chỉ dùng feature tháng gần nhất.
3. CatBoost thêm feature tích lũy cũ và độ phủ.

Có thể bổ sung đối chứng chỉ giao dịch rồi thêm geo/dialog. Khi chỉ dùng trx,
không đưa `cold_start` toàn cục vào vì cờ này dùng thông tin cả ba nguồn.
Embedding, encoder chuỗi, ensemble, tối ưu tham số rộng và giải mã tên sản
phẩm không phải điều kiện để bắt đầu baseline đầu tiên.

## 3. Bằng chứng kiểm chứng mới

- Chạy lại `scripts/validate_prepared_data.py outputs/propensity_processing_v3`
  với code hiện tại: **12/12 tháng đạt, schema v2, 221 predictor**; không nâng
  contract, không tạo lại feature.
- Manifest hiện có tổng **1.202.688 dòng, 12.854 nhãn dương, 0 nhãn unknown**.
  Bộ kiểm tra mới đối chiếu schema/dtype, khóa/quần thể, nhãn tham chiếu,
  finite/null, các bất biến feature và số dòng các tập xuất ra.
- Có 12 file feature tháng và 24 file model-input; phần feature nén khoảng
  360 MB. Đây không phải ước lượng RAM khi fit.
- Hash `schema.py` và `validation.py` hiện khớp bằng chứng `validation_v2.json`.
- 2/2 kiểm tra hợp đồng notebook đạt; `sync_preparation_notebook.py --check`
  xác nhận cell nguồn notebook 02/03/04 đồng bộ.
- 13/13 kiểm thử trong `test_propensity_data.py` đạt (77,859 giây), gồm biên
  cutoff/cửa sổ, thay đổi sự kiện tương lai không đổi feature quá khứ, nhãn
  thiếu/sai/trùng, currency unknown, tràn tổng, schema và workflow notebook.
- Tổng cộng 15/15 kiểm thử được chạy trong đợt này đạt. Không chạy lại bộ test
  merge vì đợt này không thay đổi merge hoặc dùng một file merge mới để fit.

Kiểm tra đầu ra không tái tính toàn bộ aggregate từ raw. Kiểm tra chống dùng
tương lai ở mức fixture tổng hợp có trong bộ tests chuẩn bị dữ liệu.
