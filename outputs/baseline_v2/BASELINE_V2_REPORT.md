# Báo cáo Hoàn chỉnh Hệ thống Baseline v2 & Kiểm chứng Sau Baseline (`MBD-mini`)

- **Mã đặc tả:** `baseline_v2_monthly_rolling_ooc_v1` (`v2.0.0`, SHA-256 locked trong [spec_freeze_lock.json](spec_freeze_lock.json))
- **Múi giờ & Cutoff chuẩn:** `Europe/Moscow (UTC+3)`, cắt nghiêm ngặt `event_time_msk < YYYY-MM-01 00:00:00 MSK`
- **Giao thức chính:** `Protocol B (Monthly Expanding Rolling × 5-Fold Out-of-Client)` với ngân sách cố định hàng tháng **`K = ceil(1% × 100.224) = 1.003` khách/tháng**; giữ `Protocol A (Flat OOC)` làm cầu nối đối chiếu với hệ Baseline v1 / `FE-v1`.
- **Phạm vi báo cáo:** Tổng hợp toàn bộ kết quả định lượng của hệ **Baseline v2** (`6` mốc từ `L0` đến `B_catboost_TGD_tuned`) kèm toàn bộ kết quả **4 Nhóm kiểm chứng sau Baseline** (`Nhóm 1: Dialog`, `Nhóm 2: Thiết kế dữ liệu & Rò rỉ`, `Nhóm 3: Độ tin cậy thống kê`, `Nhóm 4: Diễn giải kinh doanh & Đối chiếu bài báo MBD`).

---

## 1. Dữ liệu, Mốc Thời gian & Giao thức Huấn luyện

### 1.1. Quy mô và dòng thời gian thống nhất (`MBD-mini`)
- **Dữ liệu nguồn đã kiểm toán:**
  - `targets` / `client_split`: **`100.224` khách hàng** × `12` mốc tháng (`2022-02-28` → `2023-01-31`) = **`1.202.688` dòng khách hàng–tháng**, `5` fold cố định.
  - `detail/trx`: **`38.764.930` giao dịch** (`2020-12-31 21:00:02` → `2022-12-31 20:59:58`, tương ứng `2021-01-01 00:00:02 MSK`).
  - `detail/geo`: **`66.295.724` sự kiện vị trí** (`2021-12-11` → `2022-12-31`).
  - `detail/dialog`: **`217.205` sự kiện hội thoại** (`2021-02-10` → `2022-12-31`).
- *(Đã rà soát toàn bộ tài liệu và slide trước đây: không còn nội dung sai ghi "target trễ 6 tháng".)*
- **Lịch Modeling Panel (`10` tháng × `100.224` = `1.002.240` dòng):** Bỏ 2 tháng `2022-02..03` làm burn-in để cửa sổ GEO 90 ngày phủ đủ từ `2022-04`.
  - **Train nền:** `2022-04 .. 2022-08` (`501.120` dòng, `6.064` ca mua).
  - **Validation (chọn biến, tuning, ablation):** `2022-09` (`1.013` ca mua, tỷ lệ nền `1.011%`) và `2022-10` (`868` ca mua, `0.866%`).
  - **Locked Test (mở 1 lần sau khi chốt cấu hình):** `2022-11` (`867` ca mua, `0.865%`), `2022-12` (`759`, `0.757%`), `2023-01` (`675`, `0.673%`).
- **Tách Out-of-Client 5-Fold (`0%` trùng `client_id`):** Ở mỗi tháng `m` và mỗi fold `k ∈ {0..4}`, cả Core Train (`2022-04..m-2`) lẫn Early Stopping (`m-1`) chỉ lấy `fold != k` (~80% khách), dự báo hoàn toàn Out-of-Client trên `fold == k` (~20% khách).

### 1.2. Ý nghĩa `trans_count`, `diff_trans_date` và quan hệ với Golden Feature Set
- Đối chiếu trực tiếp với bảng `targets.tar.gz` của MBD gốc xác nhận **`trans_count` và `diff_trans_date` khớp 100% giá trị và vị trí null** với bản full. Tuy nhiên:
  - `trans_count` không giảm theo tháng và không tái tạo được từ log `detail/trx`; `diff_trans_date` chưa xác minh được công thức, đơn vị và điểm cắt thời gian.
  - **Quyết định:** Tuyệt đối **cấm dùng `trans_count`, `diff_trans_date` và nhóm `prior_*`** trong tập biến hành vi dự báo. Hai cột này cũng **không nằm trong** bất kỳ bộ feature chuẩn nào của Baseline v2 (`raw_T_8`, `raw_TGD_14`) hay `FE-v1` (`139` biến).
- **Quan hệ giữa bộ biến cốt lõi (`raw_TGD_14`) và `B-raw` / `FE-v1`:**
  - `B-raw-T` (`8` biến TRX) → `raw_TGD_14` (`14` biến thô đa nguồn: `8` TRX + `3` GEO 90d + `3` Dialog 90d) là bộ **core minimal benchmark** của Baseline v2 (`B_catboost_TGD_tuned`).
  - `L5c` (`112` biến bảng trong `panel_moscow.parquet`: `63` TRX + `30` GEO + `18` Dialog meta + `1` `cold_start`) và `FE-v1` (`139` biến = `112` bảng + `27` biến nén Dialog Embedding) là các mốc mở rộng ở Protocol A.

---

## 2. Kết quả Hệ Baseline v2 (`Protocol B` vs `Protocol A`)

### 2.1. Tuning CatBoost trên Validation (`val_catboost_tuning_grid.csv`)
Lưới `9` cấu hình gọn (`lr ∈ {0.03, 0.05, 0.08}`, `depth ∈ {4, 6}`, `l2_leaf_reg ∈ {3, 10, 25}`, `max_iterations=1500`, `es_rounds=100` trên tháng `m-1`) chỉ chạy trên Validation (`2022-09..10`). Cấu hình thắng cuộc được khóa trước khi mở Test:
- **`TuneCB_4_lr0.03_d6_l2_25.0`:** `learning_rate=0.03`, `depth=6`, `l2_leaf_reg=25.0`, `median_best_iter=432` (`362..580`).
- **Hiệu năng Validation:** Macro Lift@1% = **`8.9368x`** (`95% CI: 7.77x–10.06x`), Macro PR-AUC = **`0.03738`**, ROC-AUC = **`0.7705`**, LogLoss = **`0.04847`**.

### 2.2. Kết quả Locked Test theo từng tháng (`Protocol B`)
| Giao thức | Tháng | Mô hình (`exp_id`) | Số biến | Lift@1% (`95% CI`) | PR-AUC (`95% CI`) | ROC-AUC |
|---|---|---|---:|---:|---:|---:|
| `B_rolling_ooc` | `2022-11` | `L0_base_rate` | 0 | `1.1525x` (`0.46–1.89`) | `0.0087` (`0.0081–0.0094`) | `0.5042` |
| `B_rolling_ooc` | `2022-12` | `L0_base_rate` | 0 | `0.7899x` (`0.26–1.49`) | `0.0075` (`0.0068–0.0081`) | `0.4908` |
| `B_rolling_ooc` | `2023-01` | `L0_base_rate` | 0 | `0.7402x` (`0.15–1.44`) | `0.0068` (`0.0062–0.0075`) | `0.5047` |
| `B_rolling_ooc` | `2022-11` | `L2_rule_trx_rfm` | 6 | `3.8033x` (`2.60–5.07`) | `0.0181` (`0.0159–0.0217`) | `0.6480` |
| `B_rolling_ooc` | `2022-12` | `L2_rule_trx_rfm` | 6 | `3.8179x` (`2.58–5.31`) | `0.0158` (`0.0138–0.0187`) | `0.6404` |
| `B_rolling_ooc` | `2023-01` | `L2_rule_trx_rfm` | 6 | `5.9214x` (`4.19–7.56`) | `0.0203` (`0.0171–0.0255`) | `0.7303` |
| `B_rolling_ooc` | `2022-11` | `B_logreg_T` | 8 | `5.4169x` (`4.04–6.94`) | `0.0241` (`0.0214–0.0281`) | `0.7493` |
| `B_rolling_ooc` | `2022-12` | `B_logreg_T` | 8 | `4.4762x` (`3.07–6.00`) | `0.0205` (`0.0180–0.0240`) | `0.7291` |
| `B_rolling_ooc` | `2023-01` | `B_logreg_T` | 8 | `4.7371x` (`3.17–6.41`) | `0.0181` (`0.0156–0.0221`) | `0.7379` |
| `B_rolling_ooc` | `2022-11` | `B_logreg_TGD` | 14 | `7.3762x` (`5.62–9.06`) | `0.0309` (`0.0272–0.0360`) | `0.7736` |
| `B_rolling_ooc` | `2022-12` | `B_logreg_TGD` | 14 | `8.6891x` (`6.65–10.90`) | `0.0297` (`0.0252–0.0367`) | `0.7599` |
| `B_rolling_ooc` | `2023-01` | `B_logreg_TGD` | 14 | `5.4773x` (`3.82–7.09`) | `0.0213` (`0.0182–0.0262`) | `0.7546` |
| `B_rolling_ooc` | `2022-11` | **`B_catboost_TGD_tuned`** | **14** | **`9.0666x`** (`7.27–11.04`) | **`0.0397`** (`0.0335–0.0486`) | **`0.7765`** |
| `B_rolling_ooc` | `2022-12` | **`B_catboost_TGD_tuned`** | **14** | **`10.5761x`** (`8.49–12.66`) | **`0.0391`** (`0.0325–0.0502`) | **`0.7704`** |
| `B_rolling_ooc` | `2023-01` | **`B_catboost_TGD_tuned`** | **14** | **`6.4642x`** (`4.58–8.05`) | **`0.0247`** (`0.0211–0.0299`) | **`0.7701`** |
| `B_rolling_ooc` | `2022-11` | `L1_rule_target_lag` *(ref)* | 3 | `7.3762x` (`5.66–9.08`) | `0.0161` (`0.0137–0.0200`) | `0.5705` |
| `B_rolling_ooc` | `2022-12` | `L1_rule_target_lag` *(ref)* | 3 | `6.0560x` (`4.55–7.94`) | `0.0149` (`0.0124–0.0191`) | `0.5822` |
| `B_rolling_ooc` | `2023-01` | `L1_rule_target_lag` *(ref)* | 3 | `10.0664x` (`7.80–12.38`) | `0.0216` (`0.0174–0.0281`) | `0.6284` |

*(Cảnh báo `L1_rule_target_lag`: giả định `label_delay = 0` — nhãn tháng `m-1` sẵn có ngay tại `00:00` đầu tháng `m`; xem đây là mốc tham chiếu lý tưởng, không phải baseline hành vi triển khai mặc định).*

### 2.3. Tổng hợp Macro-Monthly (`Protocol B` vs `Protocol A`)
| Giao thức | Tập | Mô hình (`exp_id`) | Số biến | Lift@1% (`95% CI`) | PR-AUC (`95% CI`) | ROC-AUC |
|---|---|---|---:|---:|---:|---:|
| `B_rolling_ooc` | `val` | `L0_base_rate` | 0 | `0.8057x` (`0.42–1.23`) | `0.0093` (`0.0088–0.0098`) | `0.4997` |
| `B_rolling_ooc` | `val` | `L2_rule_trx_rfm` | 6 | `2.7953x` (`2.12–3.66`) | `0.0186` (`0.0171–0.0208`) | `0.6477` |
| `B_rolling_ooc` | `val` | `B_logreg_T` | 8 | `3.7984x` (`2.97–4.75`) | `0.0227` (`0.0209–0.0255`) | `0.7257` |
| `B_rolling_ooc` | `val` | `B_logreg_TGD` | 14 | `6.0264x` (`5.04–7.08`) | `0.0314` (`0.0289–0.0356`) | `0.7631` |
| `B_rolling_ooc` | `val` | **`B_catboost_TGD_tuned`** | **14** | **`8.6764x`** (`7.58–9.81`) | **`0.0374`** (`0.0341–0.0426`) | **`0.7709`** |
| `B_rolling_ooc` | `val` | `L1_rule_target_lag` *(ref)* | 3 | `6.1332x` (`4.99–7.28`) | `0.0200` (`0.0172–0.0244`) | `0.5898` |
| `A_flat_ooc` | `val` | `B_logreg_TGD` | 14 | `6.0264x` (`4.97–7.07`) | `0.0316` (`0.0291–0.0360`) | `0.7635` |
| `A_flat_ooc` | `val` | `B_catboost_TGD_tuned` | 14 | `8.5421x` (`7.51–9.71`) | `0.0369` (`0.0338–0.0421`) | `0.7703` |
| `B_rolling_ooc` | `test` | `L0_base_rate` | 0 | `0.8942x` (`0.48–1.36`) | `0.0077` (`0.0073–0.0081`) | `0.4999` |
| `B_rolling_ooc` | `test` | `L2_rule_trx_rfm` | 6 | `4.5142x` (`3.66–5.35`) | `0.0181` (`0.0167–0.0206`) | `0.6729` |
| `B_rolling_ooc` | `test` | `B_logreg_T` | 8 | `4.8767x` (`3.93–5.87`) | `0.0209` (`0.0194–0.0235`) | `0.7388` |
| `B_rolling_ooc` | `test` | `B_logreg_TGD` | 14 | `7.1809x` (`6.06–8.28`) | `0.0273` (`0.0252–0.0308`) | `0.7627` |
| `B_rolling_ooc` | `test` | **`B_catboost_TGD_tuned`** | **14** | **`8.7023x`** (`7.53–9.75`) | **`0.0345`** (`0.0314–0.0397`) | **`0.7723`** |
| `B_rolling_ooc` | `test` | `L1_rule_target_lag` *(ref)* | 3 | `7.8329x` (`6.64–9.12`) | `0.0176` (`0.0154–0.0207`) | `0.5937` |
| `A_flat_ooc` | `test` | `B_logreg_TGD` | 14 | `7.0492x` (`5.89–8.06`) | `0.0273` (`0.0252–0.0308`) | `0.7625` |
| `A_flat_ooc` | `test` | `B_catboost_TGD_tuned` | 14 | `8.4171x` (`7.35–9.54`) | `0.0346` (`0.0314–0.0398`) | `0.7722` |

### 2.4. Ablations chỉ trên Validation (`val_ablation_macro.csv`)
| Nhóm kiểm tra | Mô hình | Số biến | Lift@1% (`95% CI`) | PR-AUC (`95% CI`) | ROC-AUC |
|---|---|---:|---:|---:|---:|
| Tham chiếu chính | `B_catboost_TGD_tuned` | 14 | `8.9368x` (`7.77–10.06`) | `0.0374` (`0.0341–0.0427`) | `0.7705` |
| Tách nguồn `T+G` | `Abl_catboost_TG` *(bỏ Dialog)* | 11 | `5.6812x` (`4.67–6.82`) | `0.0270` (`0.0241–0.0311`) | `0.7311` |
| Tách nguồn `T+D` | `Abl_catboost_TD` *(bỏ GEO)* | 11 | `8.8629x` (`7.67–9.98`) | `0.0379` (`0.0345–0.0432`) | `0.7700` |
| Bỏ tenure | `Abl_catboost_TGD_no_tenure` | 13 | `7.5228x` (`6.51–8.87`) | `0.0354` (`0.0323–0.0403`) | `0.7623` |
| Trần tenure 365d | `Abl_catboost_TGD_cap365` | 14 | `8.0735x` (`7.05–9.35`) | `0.0373` (`0.0341–0.0422`) | `0.7707` |
| Blend 30/70 | `Abl_blend_prob_30_70` *(LR+CB)* | 14 | `8.7230x` (`7.64–9.95`) | `0.0370` (`0.0337–0.0425`) | `0.7708` |

---

## 3. Kiểm chứng Nhóm 1 — Dialog (Ưu tiên cao nhất)

### 3.1. Định nghĩa Dialog & Giả định nghiệp vụ
Theo [DATASETS.md](../../docs/DATASETS.md) và [mbd_mini.md](../../docs/dict/mbd_mini.md), bảng `detail/dialog` chỉ cung cấp `(client_id, event_time, embedding)`. Dataset **không có văn bản gốc, không có nhãn hướng cuộc gọi (khách gọi vào hay ngân hàng gọi ra), và không có cờ loại cuộc gọi** (hỏi thông tin vs đặt mua sản phẩm). Mọi diễn giải theo hướng "sales call / gọi để chốt đơn" được ghi rõ là **giả định cần nghiệp vụ ngân hàng xác nhận**.

### 3.2. Cuộc gọi và hành vi mua cùng tháng vs tháng sau (`group1_dialog_eda_summary.json`)
- **Cùng tháng vs tháng sau:** Trên `69.750` quan sát khách–tháng có gọi trong tháng `m`:
  - **`P(mua trong tháng m | có gọi trong m) = 6.244%`** (`4.355` ca mua), cao gấp **`1.65x`** so với **`P(mua tháng m+1 | có gọi trong m) = 3.785%`** (`2.640` ca mua). Với nhóm `không gọi trong m` (`1.032.714` quan sát), tỷ lệ tương ứng chỉ là `0.758%` và `0.888%`.
  - Khi kiểm soát `y_m = 0` (khách chưa mua trong tháng `m`), `P(mua tháng m+1 | có gọi m, y_m=0) = 3.577%` — vẫn cao gấp ~4 lần nhóm không gọi (`0.840%`).
- **Theo khoảng cách từ cuộc gọi cuối tới mốc cutoff `T` (`dialog_recency_buckets_to_cutoff.csv`):**
  - `0–3 ngày`: **`6.191%`** (`635 / 10.256`)
  - `4–7 ngày`: **`4.909%`** (`562 / 11.448`)
  - `8–14 ngày`: **`3.779%`** (`610 / 16.140`)
  - `15–30 ngày`: **`2.252%`** (`636 / 28.244`)
  - `> 30 ngày`: **`1.257%`** (`4.111 / 326.933`)
  - `Chưa từng gọi`: **`0.606%`** (`3.692 / 609.219`)
- **Theo số cuộc gọi trong tháng `m`:** tăng đơn điệu rõ: `1` cuộc (`4.52%` cùng tháng / `3.37%` tháng sau) → `2` cuộc (`8.07%` / `4.01%`) → `3–5` cuộc (`11.42%` / `5.22%`) → `6+` cuộc (`17.24%` / `7.18%`).

### 3.3. Khoảng đệm `7/14/30d`, Dịch nhãn `T+2`, Cụm Embedding & Nội dung vs Hoạt động
*(Chạy trên LightGBM Protocol A — [group1_dialog_model_experiments_fast.csv](../verification_audit/group1_dialog_model_experiments_fast.csv)):*

| Phép thử (Test `2022-11..2023-01`) | Biến thể | PR-AUC | ROC-AUC | Lift@1% | Hits@1% | Nhận xét |
|---|---|---:|---:|---:|---:|---|
| Không có Dialog (`L5b`) | `L5b_lgbm_no_dialog` | `0.02221` | `0.7520` | `5.35x` | `123` | Nền bảng TRX+GEO thuần |
| Dialog chuẩn (`lag 0d`) | `L5c_lgbm_lag0` | **`0.02841`** | `0.7755` | **`6.78x`** | `156` | Tăng `+0.00620` PR-AUC |
| Lùi cutoff Dialog **`7 ngày`** | `L5c_lgbm_lag7` | `0.02305` | `0.7620` | `5.48x` | `126` | Rơi **~86.5%** phần tăng thêm |
| Lùi cutoff Dialog **`14 ngày`** | `L5c_lgbm_lag14` | `0.02146` | `0.7563` | `4.35x` | `100` | Không còn lợi thế PR-AUC |
| Lùi cutoff Dialog **`30 ngày`** | `L5c_lgbm_lag30` | `0.02106` | `0.7505` | `4.22x` | `97` | Thấp hơn cả bỏ Dialog |
| Dịch nhãn **`T+2`** (`L5b` vs `FE-v1`) | `L5b_T+2` → `FE-v1_T+2` | `0.01570 → 0.01734` | `0.7108 → 0.7302` | `2.17x → 2.56x` | `50 → 59` | Tín hiệu **ý định sớm vẫn tồn tại** ở `T+2` |
| Bỏ Norm & Cosine (`PCA-only`) | `FE-v1_pca_only_lgbm` | **`0.02804`** *(vs `0.02716`)* | `0.7752` | **`6.43x`** *(vs `6.04x`)* | `148` | Bỏ `norm`/`cosine` còn **tốt hơn** |
| **Placebo** xáo Embedding giữa khách có gọi | `FE-v1_placebo_shuffle` | `0.02755` *(vs `0.02716`)* | `0.7737` | `7.04x` | `162` | Không sụp khi xáo nội dung vector |

- **Kiểm tra theo 6 cụm Embedding (`group1_dialog_models_by_cluster.csv`):** Cả `6` cụm (`0..5`) đều mang cùng quy luật (tỷ lệ mua cùng tháng `6.63%–8.71%` gấp `1.71x–2.02x` tháng sau) — kết quả mang tính gợi ý do không có nhãn chủ đề.
- **Kiểm tra `norm` & `cosine` (`dialog_norm_cosine_by_call_count.csv`):** Tương quan giữa `norm_all` và `log1p(count)` thấp âm nhẹ (`-0.118`), `cos_90d_all = 1.0` ở `100%` khách có số gọi 90d bằng toàn kỳ (artifact cơ học).
- **Cách đọc & khuyến nghị thiết kế:** Dialog kết hợp **(1) thành phần cận chốt đơn** tập trung mạnh trong `0–7` ngày trước đầu tháng (chiếm ~86.5% phần tăng PR-AUC) và **(2) thành phần ý định sớm còn sót lại ở `T+2`**, trong đó **tần suất/thời điểm gọi chi phối áp đảo nội dung embedding**. Vì vậy:
  - Các mô hình mở rộng (`FE-v2`) bắt buộc loại bỏ `dlg_cos_90d_all` và phải luôn báo cáo đối chiếu kèm nhánh **`no_dialog`** (thuần TRX+GEO) để phục vụ bài toán chọn khách chưa tương tác.

---

## 4. Kiểm chứng Nhóm 2 — Thiết kế Dữ liệu, Múi giờ & Rò rỉ

- **Tác động rò rỉ múi giờ (`group2_utc_vs_msk_l4.csv`):** Chạy `L4` (TRX+GEO) bằng cutoff `UTC 00:00` (lấn `+3h` sang đầu tháng mới) gây lạm phát giả tạo PR-AUC **`+0.00050 .. +0.00084`** trên LightGBM (`Test: 0.01992 vs 0.01942`; `Val: 0.02186 vs 0.02102`) và **`+0.00017 .. +0.00051`** trên CatBoost (`Test: 0.02115 vs 0.02064`), Lift@1% Test bị thổi từ `4.17x` lên `4.78x`. Bắt buộc chuẩn hóa `MSK (UTC+3)`.
- **Shuffled-Label Sanity Check (`val_shuffled_sanity.csv`):** Khi xáo nhãn trên `B_catboost_TGD_tuned`, PR-AUC rớt về sát tỷ lệ nền (`0.01175` vs `0.01011` tháng `09`; `0.01029` vs `0.00866` tháng `10` — **PASSED**). *(Lưu ý `B_logreg_TGD` trên `2022-10` có mức `0.0143` do feature ngoại lai đơn lẻ ở hồi quy tuyến tính).*
- **Permutation Importance (không phát hiện biến rò rỉ nhãn):**
  - Trên `B_catboost_TGD_tuned` (`14` biến, Val): đứng đầu là `trx_all_days_since_first_event` (`-0.01415` PR-AUC / `-4.92x` Lift), `dialog_90d_has_event` (`-0.01288` / `-4.98x`), `dialog_90d_days_since_last_event` (`-0.01059`), `trx_30d_log1p_event_count` (`-0.00703`), `trx_90d_log1p_event_count` (`-0.00521`).
  - Trên `FE-v1` (`139` biến, chạy Colab GPU trên Test — [group2_fe_v1_permutation_importance_139.csv](../verification_audit/group2_fe_v1_permutation_importance_139.csv)): đứng đầu là `dialog_30d_active_days` (`-0.00925` PR-AUC / `-2.28x` Lift), `trx_all_days_since_first_event` (`-0.00549` / `-1.46x`), `trx_30d_dst_type11_nunique` (`-0.00358`), `dialog_all_has_event` (`-0.00308`), `dialog_all_days_since_first_event` (`-0.00258`).
- **Lát cắt điểm mù (`val_slice_metrics.csv` & `slice_metrics_with_missing_trx.csv`):**
  - **`cold_start == 1`** (`3.456` Val / `2.272` Test): tỷ lệ mua cao (`1.42%` Val / `2.11%` Test), nhưng mô hình hành vi không dùng nhãn mù hoàn toàn trên Test (`ROC-AUC 0.44–0.52`, `Hits@1% ~ 0–1`), chỉ `L1_rule_target_lag` bắt được `3` ca mua lặp lại (`6.17x`).
  - **`trx_all_has_event == 0` (~`1.5%–2.5%` thiếu lịch sử TRX — `7.493` Test):** tỷ lệ mua `2.38%`. Mô hình chỉ TRX (`B-raw-T`, `L4_cb`) đạt `0.37x` Lift@1%; ngược lại khi bổ sung Dialog (`FE-v1`, `L5c_lgbm`, `B_logreg_TGD`), mô hình vẫn bắt được `11–18` ca mua (`Lift 6.55x–10.10x`).
  - **`trx_90d_has_event == 0` (`42.019` Test, tỷ lệ mua `0.51%`):** Tín hiệu Dialog giúp nâng Lift@1% lên **`18.8x–20.7x`** (`40–45` hits).

---

## 5. Kiểm chứng Nhóm 3 — Độ tin cậy Thống kê & Bất thường `2023-01`

### 5.1. Kiểm định cặp đa điều chỉnh Holm-Bonferroni & Benjamini-Hochberg
- **Hệ Protocol B Test (`paired_models_protocol_b.csv`, cluster bootstrap `1.000` lần theo `client_id`):**
  - `B_catboost_TGD_tuned` − `B_logreg_TGD`: **`ΔPR-AUC = +0.00721`** (`p_holm = 0.0`), **`ΔLift@1% = +1.5214x`** (`p_holm = 0.008`).
  - `B_catboost_TGD_tuned` − `L1_rule_target_lag`: **`ΔPR-AUC = +0.01694`** (`p_holm = 0.0`), **`ΔROC-AUC = +0.1786`** (`p_holm = 0.0`).
  - `B_logreg_TGD` − `B_logreg_T` (đóng góp thêm GEO+Dialog thô): **`ΔPR-AUC = +0.00638`** (`p_holm = 0.0`), **`ΔLift@1% = +2.3041x`** (`p_holm = 0.0`).
- **Các cặp bổ sung Protocol A Test (`paired_full_with_missing_holm_bh.csv`):**
  - `L4_cb` − `L1_rule_target_lag`: `ΔPR = +0.00406` (`p_holm = 0.006`, `p_bh = 0.0025`).
  - `L4_cb` − `L2_rule_trx_rfm`: `ΔPR = +0.00329` (`p_holm = 0.012`, `p_bh = 0.0067`).
  - `FE-v1` (`CatBoost 139`) − `L5c_lgbm_dialog_meta` (`112`): `ΔPR = -0.00112` (`p_holm = 0.404`, `p_bh = 0.404` — chưa khác biệt thống kê về PR-AUC dù Lift@1% cao hơn nhẹ `7.23x vs 6.78x`).

### 5.2. Đa seed, Quy ước gộp 5 Fold & Calibration Slope/Intercept
- **Quy ước xếp hạng:** Hệ thống báo cáo **Pooled OOF** (gộp điểm dự báo từ `5` mô hình fold rời rạc để xếp hạng chung toàn bộ `100.224` khách mỗi tháng) song song với **Mean-Across-Folds** (`fold 0..4`). Chênh lệch giữ ổn định qua từng fold (`FE-v1` Test trên `fold 0..4`: `0.0305, 0.0281, 0.0311, 0.0257, 0.0256`, `Pooled = 0.02729 ± 0.00111`).
- **Độ ổn định đa seed:** `B_catboost_TGD_tuned` qua `3` seeds (`42, 142, 242`) trên Protocol B Test có độ lệch rất nhỏ: **`PR-AUC SD = 0.00011`**, **`Lift@1% SD = 0.22x`**; `LogisticRegression (lbfgs)` tất định cho cùng kết quả qua mọi seed (từ `FE-v2` trở đi chốt chạy 1 seed `42`).
- **Calibration Slope & Intercept (`calibration_slope_intercept_summary.csv`):**
  - **`B_catboost_TGD_tuned` (Protocol B Test):** đạt chất lượng hiệu chỉnh gần lý tưởng nhờ regularization (`l2=25`) + Early Stopping: **`Slope = 0.9856`** (`± 0.00005`), **`Intercept = -0.1862`**, `ECE_quantile = 0.00098`, `LogLoss = 0.04086`.
  - Ngược lại, `FE-v1` cũ (Protocol A, chưa ES/regularization) bị overconfident rõ rệt (`Slope = 0.6940`, `Intercept = -1.304`, `LogLoss = 0.04237`).

### 5.3. Giải thích tháng `2023-01` suy giảm & Thử nghiệm biến lịch (`group3_calendar_feature_test.csv`)
- **Phân rã thành phần tháng (`monthly_composition_04_to_01.csv`):** `2023-01` ghi nhận tỷ lệ mua nền thấp kỷ lục **`0.673%`** (`675` ca mua, giảm sâu từ `1.694%` ở `2022-04`), đồng thời **tỷ trọng người mua lặp lại tăng vọt lên `42.37%` (`286/675`)** so với `26.6%–27.3%` trong hai tháng trước đó — phản ánh sự co lại mạnh của nhóm mua mới (phù hợp với giả thuyết mùa vụ kỳ nghỉ lễ đầu năm tại Nga `01–08/01`, vẫn lưu ý đây là giả thuyết do thiếu nhãn ngành hàng/chiến dịch).
- **Thử nghiệm thêm 5 biến lịch trên `base_14`:** Chỉ tăng nhẹ trên `2023-01` (`PR-AUC 0.02457 → 0.02533`, `Lift 6.07x → 6.96x`) nhưng **làm sụt Lift@1% trên Val `2022-10` (`9.09x → 8.29x`) và làm xấu `LogLoss` ở tất cả `5` tháng** (`0.03716 → 0.03780` trên `2023-01`). **Kết luận:** Không đưa trực tiếp cụm biến lịch thô vào `FE-v2`.

---

## 6. Kiểm chứng Nhóm 4 — Diễn giải Kinh doanh & Đối chiếu Bài báo MBD

### 6.1. Propensity khác Uplift: Gọi lại top-K có tạo giá trị tăng thêm?
- Kết quả Nhóm 1 cho thấy mô hình chấm điểm cao nhất các khách vừa có giao dịch hội thoại sát cutoff (`0–7` ngày), với tỷ lệ mua cùng tháng (`6.24%`) cao hơn hẳn tháng sau (`3.78%`).
- Trong vận hành gọi ra (outbound sales/retention), **Propensity Score $\neq$ Causal Uplift**: nếu nhóm top-K là các khách vốn tự chủ động liên hệ để hỏi mua/chốt sản phẩm, việc gọi lại đúng nhóm này có thể mang lại Conversion thô cao nhưng **Incremental Lift (doanh số tăng thêm nhờ cuộc gọi) thấp**.
- **Khuyến nghị triển khai:**
  1. Thẩm định với nghiệp vụ cơ chế phát sinh cuộc gọi (`inbound` vs `outbound`) và chạy thiết kế **Holdout / Randomized Control** ở nhóm top-K để đo Uplift thực.
  2. Song song báo cáo phiên bản mô hình **không dùng Dialog (`no_dialog` hoặc `lag >= 7d`)** chuyên phục vụ tìm khách hàng tiềm năng *chưa chủ động liên hệ*.

### 6.2. Đối chiếu chi tiết với Bài báo MBD Gốc ([docs/mbd_comparison/assessment.md](../../docs/mbd_comparison/assessment.md))
| Tiêu chí | Bài báo MBD gốc (`arXiv:2409.17587v2`) | Hệ thống Baseline v2 (`MBD-mini`) | Ghi chú ảnh hưởng |
|---|---|---|---|
| **Quy mô** | `1.000.000` khách có nhãn (nguồn `2.186.230`) | `100.224` khách (`10.0224%` khớp ID nguồn) | Số mẫu dương nhỏ hơn ~10 lần (`target_2` chỉ `577` dòng) |
| **Tên & cấu trúc nhãn** | 4 nhãn riêng (`bcard`, `cred`, `zp`, `acquiring_target`) | Nhãn gộp `y_any = max(target_1..4)` | Chuyển từ multi-label sang binary propensity tổng hợp |
| **Cờ `is_balanced` / `MBD-small`** | Có `is_balanced` (`70.532` khách = 1) / `MBD-small` (`69.474`) | **Không có `is_balanced`** (`7.135/100.224` thuộc nhóm = 1) | Không áp dụng lấy mẫu cân bằng nhân tạo; giữ tỷ lệ nền tự nhiên |
| **Chia Fold (`client_split`)** | Fold gốc theo archive | **`80.09%` (`80.268`) khách đổi mã fold** | Không được dùng chéo split hay trộn model giữa MBD full và mini |
| **Giao thức thời gian & Múi giờ** | Split chuẩn theo bài toán multi-label tĩnh | **Protocol B Monthly Rolling × 5-Fold OOC, `MSK (UTC+3)`** | Loại bỏ `2` tháng burn-in `02..03`, đo từng tháng theo ngân sách gọi `K=1.003` |
