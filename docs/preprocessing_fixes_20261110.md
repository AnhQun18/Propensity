# Xử lí ba vấn đề tiền xử lí — 10/11/2026

**Đã xử lí cả ba vấn đề của đợt review.** Dữ liệu hiện tại giữ nguyên
1.202.688 dòng, 12.854 nhãn dương, 221 predictor mặc định. Chưa fit mô hình.

## 1. Khóa schema

`src/propensity_data/schema.py` định nghĩa danh sách feature cho phép từ cấu hình
cửa sổ và embedding. Danh sách không được suy ra từ các cột có sẵn.
Schema v2 lưu tên, thứ tự và DuckDB dtype cho mọi cột; cột lạ như `fold` hoặc
`trans_count`, thiếu cột, đổi thứ tự và sai dtype đều bị từ chối.
Embedding bật/tắt, chiều embedding và cửa sổ có hợp đồng riêng theo cấu hình.

## 2. Kiểm tra dùng chung trước ready

`src/propensity_data/validation.py::validate_month` đọc Parquet đã xuất và kiểm tra:

- Cột/thứ tự/dtype khớp schema; predictor chỉ hữu hạn hoặc null.
- Khóa khách–tháng, cutoff và quần thể khách khớp tham chiếu.
- Nhãn 0/1/null, OR bốn target, eligibility và nhãn khớp bảng target tham chiếu.
- Count giữa các cửa sổ, active days, recency, phạm vi nguồn, currency,
  tỷ trọng và các cờ có log/cold-start.
- Tổng supervised + unlabelled bằng scoring, số khách bằng universe.

CLI và hai notebook dùng cùng hàm này trước khi ghi manifest ready. Ca hồi quy
hai amount hữu hạn `1e308` cộng thành `inf` giờ làm pipeline dừng với
`manifest.status=failed`. Không cap hoặc âm thầm sửa giá trị để vượt kiểm tra.
Reader cả hai notebook kiểm tra ready, inventory fingerprint, schema và protocol.

## 3. Cập nhật trạng thái thực thi

README, hướng dẫn chuẩn bị dữ liệu, danh mục notebook/script/source và nội dung
notebook đã được cập nhật. Báo cáo baseline readiness ngày 09/11 được giữ như
snapshot lịch sử, có ghi chú dẫn sang trạng thái mới này.

Hai notebook đã chạy raw ngày 10/11 trước đợt sửa, tạo đủ 12 tháng. Bản thực thi
và log nằm tại `outputs/notebook_execution_20261110_01/`. Notebook nguồn giữ output
trống; workflow của bản code mới đã chạy trên fixture tổng hợp.

## Kiểm chứng sau sửa

- **13/13 kiểm thử tiền xử lí và 2/2 kiểm tra hợp đồng notebook đạt.**
- Bộ kiểm thử tiền xử lí có ca schema sai/cột lạ, tràn tổng amount, thực thi cả
  hai workflow notebook, reader từ chối output failed và nâng cấp schema cũ.
- Kiểm tra hợp đồng notebook và đồng bộ helper đều đạt.
- Dùng quy tắc mới kiểm tra trực tiếp đủ 12 tháng của cả
  `outputs/propensity_data_v1/` và `outputs/propensity_processing_v3/`: đạt.
- Schema/contract/manifest của hai bộ output cục bộ đã nâng lên v2 sau khi tất
  cả tháng đạt. `contract_before_schema_v2/` lưu metadata cũ;
  `validation_v2.json` lưu kết quả và hash mã kiểm tra.
- Không tái tạo feature từ raw. File Parquet và bằng chứng thực thi cũ được giữ
  nguyên; script chỉ ghi metadata schema/contract và kết quả kiểm tra.

`scripts/validate_prepared_data.py` hỗ trợ kiểm tra output đã chuẩn bị; xem
[cách dùng](propensity_data_preparation.md). Output lỗi vẫn không được dùng để fit.
