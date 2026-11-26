# Đặc Tả Đóng Băng Baseline v2 — Giao Thức Rolling Theo Tháng × Tách Khách (`Protocol B`)

> **Trạng thái:** `FROZEN` — Tách biệt hoàn toàn **giao thức đánh giá** khỏi **mô hình**. Mọi mô hình mới về sau chỉ cần chạy qua cùng khung giao thức này.
> **File cấu hình máy đọc (kiểm tra tự động SHA-256):** [`configs/baseline_v2_evaluation_spec.json`](../configs/baseline_v2_evaluation_spec.json)
> **Ghi chú kế thừa:** Toàn bộ hệ thống Baseline v1 (`BASELINE.md`, `outputs/baseline_moscow_v1/`, `outputs/fe_v1_ablation/`) được **giữ nguyên hoàn toàn**, không ghi đè.

---

## 1. Nguyên tắc thiết kế & Giao thức B vs A

1. **Giao thức chính — `Protocol B` (`Out-of-Client 5-Fold × Expanding Rolling theo tháng`):**
   - Phản ánh đúng cách chiến dịch vận hành thật: **mỗi tháng huấn luyện lại trên toàn bộ lịch sử hợp lệ tính đến trước tháng dự báo, chấm điểm khách hàng mới của tháng kế tiếp** với ngân sách gọi cố định hàng tháng **`K = ceil(1% × 100.224) = 1.003` cuộc gọi/tháng**.
   - Bất biến tách khách trên mọi tháng đánh giá $m$ và mọi fold $k \in \{0..4\}$:
     - **Train:** các tháng `2022-04 .. m - 1` trên **`fold != k`** (~`80.179` khách/tháng).
     - **Eval:** tháng `m` trên **`fold == k`** (~`20.045` khách chưa từng xuất hiện trong train -> **`0%` trùng `client_id`**).
     - Gộp 5 fold lại thành dự báo đủ `100.224` khách cho tháng $m$, tính chỉ số **từng tháng trước (kèm 95% CI)**, rồi mới gộp **Macro-Monthly Average (kèm 95% CI)** và **Pooled OOF**.
   - **Lịch Validation (`Val` — chỉ để chẩn đoán, tune siêu tham số & chạy ablation):**
     - `2022-09`: Train `2022-04 .. 2022-08` (5 tháng)
     - `2022-10`: Train `2022-04 .. 2022-09` (6 tháng)
   - **Lịch Locked Test (`Test` — chỉ mở 1 lần sau khi đóng băng siêu tham số trên Val):**
     - `2022-11`: Train `2022-04 .. 2022-10` (7 tháng)
     - `2022-12`: Train `2022-04 .. 2022-11` (8 tháng)
     - `2023-01`: Train `2022-04 .. 2022-12` (9 tháng)

2. **Giao thức đối chiếu — `Protocol A` (`Flat OOC`):**
   - Fit 1 lần cho cả giai đoạn (`2022-04..08` cho Val `09..10`; `2022-04..10` cho Test `11..01`), giữ lại để đối chiếu trực tiếp xem việc **huấn luyện lại hàng tháng** trong `Protocol B` mang lại khác biệt bao nhiêu so với baseline v1 cũ.

3. **Chỉ số & Tính theo tháng:**
   - **Hai chỉ số chính:** `Lift@1%` (ngân sách tháng `K = 1.003`) và `PR-AUC` (`Average Precision`).
   - **Thứ tự báo cáo bắt buộc:** Từng tháng (`mean ± seed SD` và `95% CI` client-clustered bootstrap `1.000` lần) $\rightarrow$ Tổng hợp Macro-Monthly (`95% CI` bootstrap theo cụm `client_id` giữ ngân sách `K` theo từng tháng) $\rightarrow$ Pooled OOF (đối chiếu với Protocol A).

---

## 2. Thang Baseline v2 (6 Mốc Tinh Gọn)

| Exp ID | Vai trò | Số biến | Câu hỏi nghiên cứu trả lời |
|---|---|---:|---|
| `L0_base_rate` | Sàn ngẫu nhiên theo tháng | `0` | Chọn ngẫu nhiên `1.003` khách/tháng đạt mức nào? |
| `L2_rule_trx_rfm` | Luật RFM thuần `TRX` | `6` | Luật đơn giản có đủ chưa, và snapshot có sạch không? |
| `B_logreg_T` | `LogReg L2` trên `TRX` | `8` | Mốc tuyến tính chỉ dùng dữ liệu giao dịch đạt tới đâu? |
| `B_logreg_TGD` | `LogReg L2` trên `TRX+GEO+DIALOG` thô | `14` | Ngưỡng tuyến tính đa nguồn mạnh nhất cần vượt là bao nhiêu? |
| `B_catboost_TGD_tuned` | `CatBoost` 14 biến thô (tune nhẹ) | `14` | Mốc họ cây công bằng (`learning_rate`, `depth`, `l2_leaf_reg`, early stopping) đạt bao nhiêu? |
| `L1_rule_target_lag` | **Tham chiếu — dùng nhãn quá khứ** | `3` | Lịch sử mua `1/3/6m` đứng đâu? *(Xem cảnh báo độ trễ nhãn bên dưới)* |

### Thay đổi so với Baseline v1:
- **Bỏ `CatBoost` mặc định trên 8 biến (`B-raw-T`):** không còn cần thiết khi đã có `B_catboost_TGD_tuned` công bằng trên 14 biến.
- **Bỏ hoàn toàn resample nhãn:** giữ phân phối tự nhiên (~`99:1`), tập trung xếp hạng (nếu cần xác suất chuẩn có thể hiệu chỉnh hậu kỳ sau).
- **Cảnh báo bắt buộc về `L1_rule_target_lag` (Độ trễ nhãn):** Theo kết quả kiểm toán tại [`docs/propensity_feasibility/remaining_questions/assessment.md`](propensity_feasibility/remaining_questions/assessment.md#L135-L186), dữ liệu `targets` không có timestamp công bố nhãn (`available_at`). `L1` dùng `prior_1m_bought_any` giả định nhãn tháng `m-1` sẵn có ngay tại `00:00` ngày đầu tháng `m` (`label_delay = 0`). Nếu thực tế ngân hàng có độ trễ chốt nhãn $\ge 1$ ngày, `L1` là mốc tham chiếu **"thiên đường" (upper bound)** chứ không triển khai được ngay.

---

## 3. Thiết kế Snapshot, Biến & Các Ablation Chỉ Chạy Trên Val

1. **Snapshot & Burn-in:**
   - Giữ nguyên chuẩn `Europe/Moscow (UTC+3)` và lịch sử tích lũy `TRX` từ `01/01/2021` trong `panel_moscow.parquet` (không cắt giai đoạn 2021 chưa có nhãn).
   - Burn-in tối thiểu `2` tháng (`2022-02`, `2022-03`) để nguồn `GEO` (bắt đầu `11/12/2021`) đủ cửa sổ `90d` từ `2022-04`.
2. **Tuning nhẹ `CatBoost` (chỉ trên Val):**
   - Lưới gọn `9` cấu hình `(learning_rate ∈ {0.03, 0.05, 0.08}, depth ∈ {4, 6}, l2_leaf_reg ∈ {3, 10, 25})` với `max_iterations = 1500, early_stopping_rounds = 100`.
   - Cơ chế early stopping không rò rỉ trong rolling: với mỗi bước `(m, fold k)`, tách tháng `m-1` (`fold != k`) làm `eval_set` nội bộ để tìm `best_iteration_`, sau đó refit trên toàn bộ `2022-04..m-1` (`fold != k`) đúng `best_iteration_` cây. Chọn 1 cấu hình tốt nhất theo **Macro-Val `Lift@1%` (ưu tiên) / `PR-AUC`**, khóa cố định trước khi mở Test.
3. **Ablation & Chẩn đoán chỉ chạy trên Val (`2022-09`, `2022-10`):**
   - **Tách nguồn `T+G` và `T+D`** cho cả hai họ `LogReg` và `CatBoost` (so với `T` và `TGD`).
   - **Xử lý `trx_all_days_since_first_event`** (biến bị cắt trái tại `01/2021` và tăng đều theo lịch):
     - Biến thể 1 (`no_tenure`): loại bỏ biến này (`13` biến).
     - Biến thể 2 (`cap365`): chặn trên `min(trx_all_days_since_first_event, 365.0)` (`14` biến, tối đa 12 tháng).
   - **Trộn tuyến tính + cây (`LogReg + CatBoost` ensemble đơn giản trên 14 biến):** thử trộn xác suất (`w = 0.5/0.7/0.3`), trộn phân vị (`rank_pct 50/50`) và trộn logit (`50/50`).
   - **Permutation Importance (5 repeats), Phân tích Lát cắt (`cold_start`, `trx_90d`, `geo_90d`, `dialog_90d/all`) và Shuffled-Label Sanity Check.**
