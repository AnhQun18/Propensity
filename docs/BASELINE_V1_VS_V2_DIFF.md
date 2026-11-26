# Báo Cáo Đối Chiếu Chi Tiết: Baseline v1 vs Baseline v2

> **Mục đích tài liệu:** Ghi lại đầy đủ mọi khác biệt về **kiến trúc mã nguồn**, **giao thức đánh giá**, **thang mốc baseline**, **thiết kế biến & ablations**, và **kết quả định lượng thực nghiệm** giữa **Baseline v1** ([`BASELINE.md`](../BASELINE.md)) và **Baseline v2** ([`BASELINE_V2.md`](../BASELINE_V2.md)).
> **Nguyên tắc bảo toàn:** Toàn bộ mã nguồn, spec và kết quả của v1 được giữ nguyên vẹn; v2 chạy hoàn toàn độc lập trong [`src/baseline_v2/`](../src/baseline_v2/) và xuất kết quả ra [`outputs/baseline_v2/`](../outputs/baseline_v2/).

---

## 1. Tóm Tắt Các Khác Biệt Cốt Lõi

| Phương diện | Baseline v1 (`fe_v1_eval_spec_v1`) | Baseline v2 (`baseline_v2_monthly_rolling_ooc_v1`, `v2.0.0`) | Tác động chính |
|---|---|---|---|
| **Triết lý thiết kế** | Các mô hình & phép đo gắn liền với pipeline ablation FE-v1. | **Tách giao thức khỏi mô hình**: đóng băng `spec.json` + khóa `SHA-256` trước khi chạy; mọi mô hình triển khai chung giao diện `BasePropensityModel`. | Đổi mô hình về sau không phải bàn lại cách đo. |
| **Giao thức chính** | **Flat OOC (trải phẳng theo giai đoạn)**: fit `1` lần cho cả Val (`09..10`) và `1` lần cho cả Test (`11..01`). | **`Protocol B` (Monthly Rolling × 5-fold OOC)**: mỗi tháng `m` huấn luyện lại trên toàn bộ `2022-04..m-1` (`fold != k`), chấm `fold == k` ở tháng `m`. Giữ **`Protocol A` (Flat OOC)** làm đối chiếu. | Mô phỏng đúng vận hành chiến dịch thật; `LogLoss` giảm có ý nghĩa (`p=0.0`). |
| **Ngân sách `K` & Cách gộp** | Báo cáo chính theo **Pooled OOF** (`K_val = 2.005`, `K_test = 3.007`), `monthly_metrics_test.csv` chỉ lưu điểm trung bình không có CI riêng từng tháng. | **Tính theo từng tháng trước (`K = 1.003` cuộc gọi/tháng) kèm 95% CI**, sau đó mới gộp **Macro-Monthly Average (kèm 95% CI)**; `Pooled OOF` chỉ giữ làm đối chiếu phụ. | Loại bỏ hiệu ứng lệch base rate theo tháng (chứng minh qua `L0` Pooled ROC-AUC `0.5280` vs từng tháng `~0.50`). |
| **Kỷ luật Val vs Test** | Chạy chung quy trình so sánh 2x2 và Holm trên Val & Test. | **Val-only gate nghiêm ngặt**: tuning CatBoost, `T+G`/`T+D`, `tenure` ablations, blend ensemble, Permutation Importance, slice & sanity check chỉ chạy trên Val; Test mở `1` lần. | Không rò rỉ quyết định siêu tham số từ Test. |
| **Thang Baseline** | **7 mốc** (6 chính + `B-raw-TGD_logreg` phụ = thực chất 7 cấu hình đánh số từ `L0` tới `FE-v1` khi phân tích FE, trong bảng baseline thuần có `L0, L1, L2, L3, B-raw-T, B-raw-TGD, B-raw-TGD_logreg`). | **6 mốc tinh gọn** có 1 câu hỏi duy nhất mỗi mốc: `L0`, `L2`, `B_logreg_T`, `B_logreg_TGD`, `B_catboost_TGD_tuned`, `L1` *(tham chiếu)*. Bỏ `B-raw-T` (CatBoost mặc định trên 8 biến). | Bảng kết quả gọn, loại bỏ CatBoost yếu kém do mặc định không regularize. |
| **Cấu hình CatBoost (14 biến)** | **Mặc định (`depth=6, l2=3, ~500 trees`)**, không early stopping -> bị overfit và **thua LogReg nặng**. | **Tune nhẹ trên Val + Temporal Early Stopping (`lr=0.03, depth=6, l2=25.0`, median `432` cây)** -> **Đảo chiều kết quả, thắng LogReg trên mọi chỉ số (`+21.2%` Lift@1%, `+26.4%` PR-AUC)**. | Sửa hoàn toàn kết luận "cây thua tuyến tính trên 14 biến thô" của v1. |
| **Vị thế của `L1_rule_target_lag`** | Đứng ngang hàng trong bảng chính (`uses_target_lag=True`). | Đánh dấu bắt buộc **`is_reference_only=true`** ("dùng nhãn quá khứ") kèm cảnh báo độ trễ nhãn vì `targets` không có `available_at` (`label_delay=0` chưa xác minh). | Ngăn nhầm lẫn `L1` thành baseline triển khai được nếu ngân hàng chốt nhãn trễ. |
| **Ablation mới trên Val** | Tập trung chuỗi FE (`8 -> 63 -> 100 -> 112 -> 139` biến). | Tách nguồn `T+G`/`T+D` cho cả 2 họ; kiểm tra biến `trx_all_days_since_first_event` (`no_tenure`, `cap365`); thử blend `LogReg + CatBoost`. | Hiểu rõ đóng góp riêng của `DIALOG` vs `GEO` và rủi ro cắt trái của biến tenure. |
| **Hỗ trợ Colab** | Chỉ script Python local. | Bổ sung package [`src/baseline_v2/`](../src/baseline_v2/) và notebook Google Colab [`notebooks/11_baseline_v2_colab.ipynb`](../notebooks/11_baseline_v2_colab.ipynb). | Sẵn sàng chạy GPU/High-RAM trên cloud. |

---

## 2. Khác Biệt Về Kiến Trúc & Giao Thức Đánh Giá

### 2.1 Tách giao thức khỏi mô hình & Khóa SHA-256
- **Baseline v1:** Cấu hình lưu ở [`configs/fe_v1_evaluation_spec.json`](../configs/fe_v1_evaluation_spec.json) và [`docs/FE_V1_FROZEN_SPEC.md`](../docs/FE_V1_FROZEN_SPEC.md). Logic chạy nằm tập trung trong script phục vụ đồng thời cả baseline và chu trình FE (`L4`, `M_deriv`, `L5c`, `FE-v1`).
- **Baseline v2:** Tổ chức lại thành module chuẩn tách lớp trong [`src/baseline_v2/`](../src/baseline_v2/):
  - [`spec.py`](../src/baseline_v2/spec.py): đọc [`configs/baseline_v2_evaluation_spec.json`](../configs/baseline_v2_evaluation_spec.json), kiểm tra tự động toàn vẹn `SHA-256` trước mọi lần chạy, đảm bảo danh sách biến hợp lệ và không chứa cột cấm (`client_id, fold, month, cutoff_msk, y_any, target_1..4, prior_months_observed`).
  - [`models.py`](../src/baseline_v2/models.py): trừu tượng hóa `BasePropensityModel` (`fit(X_train, y_train, X_es, y_es)`, `predict_scores(X)`). Các mô hình mới sau v2 chỉ cần kế thừa lớp này để cắm thẳng vào khung đánh giá.
  - [`protocol.py`](../src/baseline_v2/protocol.py): hiện thực hóa chung cả hai lịch trình `B_rolling_ooc` và `A_flat_ooc`, đảm bảo bất biến `0%` trùng `client_id` ở mọi `(month, fold k)`.
  - [`metrics.py`](../src/baseline_v2/metrics.py): tính chỉ số từng tháng (95% CI bootstrap), `Macro-Monthly`, `Pooled OOF`, và kiểm định ghép cặp kèm hiệu chỉnh Holm-Bonferroni.
  - [`diagnostics.py`](../src/baseline_v2/diagnostics.py): gom toàn bộ khâu **chỉ chạy trên Val** (tuning CatBoost, ablation `T+G`/`T+D`, `tenure`, `blend`, Permutation Importance, Slice, Shuffled Sanity).
  - [`runner.py`](../src/baseline_v2/runner.py) & [`src/scripts/run_baseline_v2.py`](../src/scripts/run_baseline_v2.py): hỗ trợ `--phase val`, `--phase test`, `--phase all`.

### 2.2 `Protocol A (Flat)` vs `Protocol B (Monthly Rolling)`
- **Điểm giống nhau được giữ nguyên:**
  - Cùng file [`outputs/baseline_moscow_v1/panel_moscow.parquet`](../outputs/baseline_moscow_v1/panel_moscow.parquet), cùng múi giờ `Europe/Moscow (UTC+3)`, cùng quy tắc `cutoff < YYYY-MM-01 00:00:00 MSK`.
  - Giữ lịch sử `TRX` tích lũy từ `01/2021` và 2 tháng burn-in (`2022-02`, `2022-03`) cho cửa sổ `90d` của `GEO` (bắt đầu `11/12/2021`).
  - Cùng không resample nhãn (`resample_labels = false`).
- **Điểm khác biệt tại lịch huấn luyện:**
  - **Trong v1 (`A_flat_ooc`):**
    - Giai đoạn Val (`2022-09`, `2022-10`): học **1 lần** trên `2022-04..2022-08` (5 tháng), dùng model đó chấm cả tháng `09` và tháng `10`.
    - Giai đoạn Test (`2022-11`, `2022-12`, `2023-01`): học **1 lần** trên `2022-04..2022-10` (7 tháng), dùng model đó chấm cả `11`, `12`, `01`.
  - **Trong v2 (`B_rolling_ooc`):**
    - Với tháng `m = 2022-09`: học `04..08` (5 tháng) $\rightarrow$ chấm `2022-09`.
    - Với tháng `m = 2022-10`: **huấn luyện lại** trên `04..09` (6 tháng) $\rightarrow$ chấm `2022-10`.
    - Với tháng `m = 2022-11`: học `04..10` (7 tháng) $\rightarrow$ chấm `2022-11` *(trùng bước đầu của `A_flat_ooc`)*.
    - Với tháng `m = 2022-12`: **huấn luyện lại** trên `04..11` (8 tháng) $\rightarrow$ chấm `2022-12`.
    - Với tháng `m = 2023-01`: **huấn luyện lại** trên `04..12` (9 tháng) $\rightarrow$ chấm `2023-01`.

### 2.3 Thực chứng lý do v2 ưu tiên `Macro-Monthly` trước `Pooled OOF`
Trong **v1** (`BASELINE.md`, [`model_summary_multiseed.csv`](../outputs/fe_v1_ablation/metrics/model_summary_multiseed.csv)), báo cáo chính lấy **Pooled OOF** trên toàn bộ `300.672` dòng Test (`K = 3.007` cuộc gọi gộp 3 tháng):
- Trên thực tế vận hành, call center gọi `K = 1.003` khách **mỗi tháng**, không thể dồn `3.007` cuộc gọi sang tháng có tỷ lệ mua cao hơn.
- Khi gộp `Pooled OOF`, tỷ lệ nền (`base_rate`) của các tháng Test giảm đơn điệu (`0.8651% -> 0.7573% -> 0.6735%`), khiến ngay cả mô hình sàn ngẫu nhiên `L0_base_rate` trong **v2 `B_rolling_ooc` Pooled OOF Test** có **`ROC-AUC = 0.5280`** (và `Lift@5% = 1.165×`) chỉ nhờ `p_train` các tháng sau thấp hơn tháng trước! Trong khi đó, khi chấm **từng tháng trước**, `ROC-AUC` của `L0` luôn nằm chuẩn quanh `0.50` (`0.5042`, `0.4908`, `0.5047`; Macro-Monthly = **`0.4999`**).

---

## 3. Khác Biệt Về Thang Baseline, Thiết Kế Biến & Mô Hình

1. **Rút 7 mốc xuống 6 mốc rõ câu hỏi:**
   - **Loại bỏ `B-raw-T` (CatBoost mặc định trên 8 biến TRX):** Trong v1, `B-raw-T` chỉ phục vụ cặp đối chứng `C1` so với `L3_logreg_rfm`. Sang v2, mô hình cây chỉ cần 1 đại diện công bằng là `B_catboost_TGD_tuned`.
   - **Đổi tên để thống nhất không gian giao thức:** `L3_logreg_rfm` $\rightarrow$ `B_logreg_T`; `B-raw-TGD_logreg` (từ mô hình phụ) $\rightarrow$ `B_logreg_TGD` (mốc đối thủ chính thức); `B-raw-TGD` $\rightarrow$ `B_catboost_TGD_tuned`.
2. **Thay đổi cách huấn luyện `CatBoost`:**
   - **v1:** Cố định `iterations=500, depth=6, l2_leaf_reg=3` (default), không có `eval_set` early stopping.
   - **v2:** Tune lưới `9` cấu hình hoàn toàn trên Val `Protocol B` với `max_iterations=1500, early_stopping_rounds=100`, `eval_metric=Logloss`, chia `eval_set` nội bộ là tháng `m-1` (`fold != k`) rồi refit trên toàn bộ `2022-04..m-1` (`fold != k`) đúng `best_iteration_` cây. Cấu hình thắng: **`lr=0.03, depth=6, l2_leaf_reg=25.0`** (`median_best_iter=432`).
3. **Bổ sung 3 nhóm Val-only Ablation mới:**
   - **Tách nguồn trên cùng 2 họ mô hình:** `Abl_logreg_TG`, `Abl_logreg_TD`, `Abl_catboost_TG`, `Abl_catboost_TD`. (v1 chỉ tách `TRX` vs `TGD` cho cả 2 họ và FE từ `63 -> 100 -> 112 -> 139`).
   - **Kiểm tra biến bị cắt trái `trx_all_days_since_first_event`:** `no_tenure` (13 biến) và `cap365` (chặn trên 365 ngày).
   - **Blend tuyến tính + cây (`LogReg + CatBoost`):** 5 biến thể (`prob 50/50, 70/30, 30/70`, `rank 50/50`, `logit 50/50`).
4. **Cơ chế Sanity Check (`val_shuffled_sanity.csv` vs `shuffled_label_sanity.csv`):**
   - **v1:** Chạy `10` lần lặp cho cả `shuffle_y_train` và `shuffle_y_eval` trên `FE-v1`.
   - **v2:** Hiện tại trong `diagnostics.py` chạy `1` lần lặp `shuffle_y_train` trên Val (`2022-09`, `2022-10`) cho `B_logreg_TGD` và `B_catboost_TGD_tuned` (chưa xáo trộn `y_eval`), dẫn đến `B_logreg_TGD` tháng `2022-10` có gap `0.00564` (> `0.005`).

---

## 4. So Sánh Định Lượng Thực Nghiệm: v1 vs v2 (`Protocol A` & `Protocol B`)

### 4.1 Khớp nối chuẩn hóa: v1 (`BASELINE.md`) vs v2 `Protocol A (Pooled OOF)`
Bảng dưới đối chiếu **Test Pooled OOF** (`300.672` dòng, `2.301` pos) giữa [`model_summary_multiseed.csv`](../outputs/fe_v1_ablation/metrics/model_summary_multiseed.csv) của v1 và [`pooled_oof_summary.csv`](../outputs/baseline_v2/metrics/pooled_oof_summary.csv) của v2 `A_flat_ooc`:

| Mốc v1 $\leftrightarrow$ Mốc v2 | v1 Pooled Test `Lift@1%` | v2 `A_flat_ooc` Pooled `Lift@1%` | v1 Pooled `PR-AUC` | v2 `A_flat_ooc` `PR-AUC` | v1 `ROC-AUC` | v2 `A_flat_ooc` `ROC-AUC` | Ghi chú khớp nối |
|---|---:|---:|---:|---:|---:|---:|---|
| `L0_base_rate` | `0.9126×` | `0.9126×` | `0.00768` | `0.00768` | `0.5022` | `0.5022` | Khớp tuyệt đối |
| `L2_rule_trx_rfm` | `4.4324×` | `4.4324×` | `0.01759` | `0.01759` | `0.6694` | `0.6694` | Khớp tuyệt đối |
| `L3_logreg_rfm` $\leftrightarrow$ `B_logreg_T` | `4.7366×` | `4.7366×` | `0.02074` | `0.02074` | `0.7394` | `0.7394` | Khớp tuyệt đối |
| `B-raw-TGD_logreg` $\leftrightarrow$ `B_logreg_TGD` | `7.1267×` | `7.1267×` | `0.02700` | `0.02700` | `0.7640` | `0.7640` | Khớp tuyệt đối |
| `L1_rule_target_lag` *(ref)* | `7.3874×` | `7.3874×` | `0.01682` | `0.01682` | `0.5914` | `0.5914` | Khớp tuyệt đối |
| `B-raw-TGD` *(v1 default CB)* $\leftrightarrow$ `B_catboost_TGD_tuned` *(v2)* | `6.3010×` | **`8.8649×`** | `0.02230` | **`0.03452`** | `0.7213` | **`0.7728`** | **Tăng `+40.7%` Lift@1%, `+54.8%` PR-AUC nhờ tune nhẹ + ES** |
| *(Tham chiếu FE-v1 139 biến)* | `7.2281×` | — | `0.02729` | — | `0.7483` | — | *`B_catboost_TGD_tuned` (chỉ 14 biến thô) đã vượt cả `FE-v1` mặc định (139 biến)!* |

> [!IMPORTANT]
> **Phát hiện mang tính bước ngoặt khi đối chiếu v1 vs v2:** Mô hình **`B_catboost_TGD_tuned` trên đúng 14 biến thô của v2** (`Lift@1% Pooled = 8.8649×` ở `A_flat_ooc`, `8.8939×` ở `B_rolling_ooc`; `PR-AUC = 0.0344–0.0345`; `ROC-AUC = 0.773`) đã **vượt qua cả mốc chuẩn `FE-v1` (139 biến, CatBoost mặc định)** của vòng trước (`Lift@1% = 7.2281×`, `PR-AUC = 0.02729`, `ROC-AUC = 0.7483`). Điều này cho thấy ở vòng FE-v1, phần lớn giới hạn của `CatBoost` đến từ việc thiếu regularization (`l2_leaf_reg=3`) và thiếu early stopping hơn là bản thân họ cây GBDT.

### 4.2 Định lượng tác động của Monthly Rolling (`Protocol B` minus `Protocol A`)
Trích từ [`paired_protocol_b_vs_a.csv`](../outputs/baseline_v2/metrics/paired_protocol_b_vs_a.csv) trên chỉ số **Macro-Monthly** (`1.000` bootstrap replicates):

| Giai đoạn | `exp_id` | `B` `Lift@1%` | `A` `Lift@1%` | $\Delta\text{Lift@1\%}$ (`p`) | `B` `PR-AUC` | `A` `PR-AUC` | $\Delta\text{PR-AUC}$ (`p`) | $\Delta\text{LogLoss}$ (`p`) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Val** (`09..10`) | `B_logreg_T` | `3.7984×` | `3.7408×` | `+0.0576×` (`0.780`) | `0.02272` | `0.02275` | `-0.00003` (`0.456`) | `-0.000002` (`0.798`) |
| **Val** (`09..10`) | `B_logreg_TGD` | `6.0264×` | `6.0264×` | `0.0000×` (`0.962`) | `0.03141` | `0.03164` | `-0.00023` (`0.036`) | `-0.000001` (`0.948`) |
| **Val** (`09..10`) | `B_catboost_TGD_tuned` | `8.6764×` | `8.5421×` | `+0.1343×` (`0.662`) | `0.03736` | `0.03695` | `+0.00041` (`0.184`) | **`-0.000060`** (**`0.000`**) |
| **Test** (`11..01`) | `B_logreg_T` | `4.8767×` | `4.7342×` | `+0.1426×` (`0.284`) | `0.02092` | `0.02079` | **`+0.00012`** (**`0.000`**) | **`-0.000031`** (**`0.000`**) |
| **Test** (`11..01`) | `B_logreg_TGD` | `7.1809×` | `7.0492×` | `+0.1317×` (`0.254`) | `0.02730` | `0.02731` | `-0.00001` (`0.888`) | **`-0.000037`** (**`0.000`**) |
| **Test** (`11..01`) | **`B_catboost_TGD_tuned`** | **`8.7023×`** | `8.4171×` | **`+0.2851×`** (`+3.39%`, `0.208`) | `0.03450` | `0.03464` | `-0.00014` (`0.620`) | **`-0.000087`** (**`0.000`**) |

**Chi tiết theo từng tháng Test (`monthly_metrics_all.csv`):**
- **`2022-11`:** `Protocol B` và `Protocol A` dùng chung tập train `2022-04..10` nên điểm giống hệt nhau (`B_catboost_TGD_tuned = 9.0666×` / `0.03969`).
- **`2022-12` (thêm tháng train `2022-11`):** Rolling giúp `B_logreg_TGD` tăng `Lift@1%` từ `8.2941× -> 8.6891×` (`+0.395×`) và `B_catboost_TGD_tuned` tăng `Lift@1%` từ `10.3128× -> 10.5761×` (`+0.263×`), đồng thời giảm `LogLoss` (`0.04053 -> 0.04046`).
- **`2023-01` (thêm 2 tháng train `2022-11, 2022-12`):** Rolling giúp `B_catboost_TGD_tuned` giữ `Lift@1%` tốt hơn rõ rệt (`6.4642×` vs `5.8721×`, `+0.592×`) và `B_logreg_T` tăng `4.4411× -> 4.7371×`, dù biến động bootstrap theo tháng vẫn rộng nên độ chênh Macro chưa đạt `p < 0.05` trên `Lift@1%` (nhưng **cải thiện `LogLoss` có ý nghĩa thống kê `p = 0.0` cho cả 3 họ mô hình ML** nhờ mô hình cập nhật tỷ lệ nền gần nhất trước tháng chấm điểm).

### 4.3 Kết luận về việc sử dụng cho vòng Feature Engineering tiếp theo (FE-v2)
1. **Không quay lại dùng `Protocol A` làm giao thức chính:** Mọi vòng FE/mô hình mới kể từ bây giờ chạy qua khung **`Protocol B` (`configs/baseline_v2_evaluation_spec.json`)**, báo cáo **từng tháng trước (`K=1.003`)** rồi mới gộp `Macro-Monthly`.
2. **Ngưỡng baseline mới cần vượt trên Locked Test (`Protocol B` Macro-Monthly):**
   - Ngưỡng tuyến tính đa nguồn (`B_logreg_TGD`): `Lift@1% = 7.1809×`, `PR-AUC = 0.02730`.
   - **Ngưỡng họ cây chuẩn hóa công bằng (`B_catboost_TGD_tuned`, 14 biến thô):** **`Lift@1% = 8.7023×`**, **`PR-AUC = 0.03450`**, **`ROC-AUC = 0.7723`** — đây là mốc chuẩn mới cần vượt khi thiết kế FE-v2 hoặc mô hình mới.
