# Mã nguồn (`src/`)

Thư mục `src/` gồm **2 nhóm tách biệt rõ ràng**:
1. **5 package độc lập chuyên cho Đọc dữ liệu, Thống kê mô tả & EDA (cơ bản đến nâng cao)** — sẵn sàng đưa lên repo công khai, chỉ cần giữ thư mục `src/` trong thư mục làm việc là dùng ngay không cần `pip install`.
2. **Mã nguồn riêng của bài toán MBD-mini** — giữ nguyên ổn định cho báo cáo và bộ test hiện hành.

---

## A. 5 Package độc lập cho Đọc, Thống kê mô tả & EDA (Public-Ready)

Mỗi package đặt tên trực tiếp theo nhiệm vụ, hỗ trợ đa định dạng đầu vào tối đa (Parquet/Hive, CSV/TSV/nén, JSON/JSONL, Feather/IPC, ORC, Excel, SQLite, DuckDB, DataFrame/Arrow/Polars) và hỗ trợ cả 2 cách import:
- **Giữ `src/` trong thư mục làm việc (không cần cài đặt):** `from src.<package> import ...` hoặc `python -m src.<package> ...`
- **Qua `PYTHONPATH=src` hoặc `pip install -e .`:** `import <package>`

| Bước | Package | Nhiệm vụ | Tài liệu |
| :-: | :--- | :--- | :--- |
| **1. Đọc & Kiểm kê** | `data_io/` | Đọc/ghi đa định dạng (10+ format), streaming batch, kiểm kê song song, băm fingerprint bất biến, `RunManifest` & `DuckDBSession` | [`data_io/README.md`](data_io/README.md) |
| **2. Thống kê mô tả** | `data_profiler/` | `describe_table` (tóm tắt nhanh mọi cột) + engine quét 1-pass out-of-core bằng SQLite (`OnlineMoments`, `UniformSample`, `skewness/kurtosis/zeros/quantiles`, `PrimaryCodeGrouper`, bảng đơn & đa bảng) | [`data_profiler/README.md`](data_profiler/README.md) |
| **3. EDA Chất lượng** | `data_audit/` | Kiểm toán tổ hợp null (`joint co-missingness`), ngoại lai đơn biến (IQR/MAD/tails) & ngoại lai trong nhóm (`log-MAD` hash-bucket), độ phủ thời gian đa nguồn `>50%`, kiểm tra bản ghi lỗi & khóa nối | [`data_audit/README.md`](data_audit/README.md) |
| **4. EDA Nâng cao** | `eda_advanced/` | Tương quan Pearson/Spearman/Kendall, Cramér's V, VIF đa cộng tuyến, quét tín hiệu target (IV/WoE/AUC/KS), phân tích Drift (PSI/KS/Wasserstein/JS), Wilson & cluster bootstrap CI, kiểm định 2 mẫu, entropy/novelty theo kỳ, bipartite IDF/SVD, event study có phân tầng cố định, Shift-Share 6 nhóm & lift | [`eda_advanced/README.md`](eda_advanced/README.md) |
| **5. Biểu đồ & Báo cáo EDA** | `eda_report/` | Xuất báo cáo EDA Markdown/CSV/JSON + Mermaid ERD có **chặn rò rỉ cột định danh**, biểu đồ EDA headless (correlation heatmap, missingness, distribution, rate matrix, timeline, 4-file bundle) & `SlideBuilder` PDF 16:9 | [`eda_report/README.md`](eda_report/README.md) |

---

## B. Mã nguồn riêng cho bài toán MBD-mini

| Vị trí | Vai trò |
| --- | --- |
| `propensity_data/` | Thư viện chuẩn bị dữ liệu trước cutoff, kiểm tra schema, hợp nhất bảng và CLI cho MBD-mini |
| `propensity_model/` | Pipeline huấn luyện & đánh giá mô hình cho MBD-mini |
| `mbd_mini_eda/` | Package thống kê mô tả nền tảng cho MBD-mini |
| `audit_mbd_inventory.py`, `audit_mbd_join_questions.py` | Kiểm kê dữ liệu, ERD và khóa nối (Báo cáo 1 · Slide 2) |
| `audit_mbd_temporal_coverage.py`, `audit_mbd_full_history_missingness.py` | Độ phủ thời gian các nguồn & thiếu log toàn lịch sử (Báo cáo 1 · Slides 3, 8, 13, 17) |
| `audit_any_product_cohorts.py` | Kiểm tra nhãn `y_any` và cohort (Báo cáo 1 · Slides 4–7) |
| [`scripts/`](scripts/README.md) | Script tạo dữ liệu nền, biểu đồ, đối soát số liệu và xuất PDF cho cả 2 báo cáo |
