# MBD-mini: Báo cáo Xây dựng & Kiểm chứng Hệ thống Baseline Propensity Chuẩn Múi Giờ Moscow (`UTC+3`)

Tài liệu này tổng hợp đầy đủ quy trình thiết kế dữ liệu chống rò rỉ theo múi giờ **Europe/Moscow (`UTC+3`)**, giao thức đánh giá nghiêm ngặt 2 chiều (**Out-of-Time + 5-Fold Disjoint Clients — `0%` trùng khách hàng giữa train và eval**), và kết quả thực nghiệm trọn bộ bậc thang baseline (**`L0 → L5d`** thuần hành vi + nhánh phụ tham chiếu **`Target-Lag`**) trên bộ dữ liệu **MBD-mini** (`100.224` khách hàng doanh nghiệp × `10` tháng mô hình hóa `04/2022–01/2023`, tổng cộng `1.002.240` dòng).

- **Mã nguồn tạo đặc trưng & embeddings chuẩn Moscow:** [`src/scripts/build_moscow_features.py`](src/scripts/build_moscow_features.py)
- **Mã nguồn chạy bậc thang Baseline & kiểm định Bootstrap:** [`src/scripts/run_moscow_ladder.py`](src/scripts/run_moscow_ladder.py)
- **Thư mục đầu ra chính thức:** [`outputs/baseline_moscow_v1/`](outputs/baseline_moscow_v1/)
  - Dữ liệu đặc trưng: [`panel_moscow.parquet`](outputs/baseline_moscow_v1/panel_moscow.parquet) (`184,62 MB`, `1.002.240 × 135` cột)
  - Vector ngữ nghĩa hội thoại `768d`: [`dialog_emb_moscow.npz`](outputs/baseline_moscow_v1/dialog_emb_moscow.npz) (`1.513,63 MB`, mean-pool `all` & `90d`)
  - Kiểm toán ranh giới thời gian & manifest: [`moscow_audit.json`](outputs/baseline_moscow_v1/moscow_audit.json) · [`ladder_manifest.json`](outputs/baseline_moscow_v1/ladder_manifest.json)
  - Bảng chỉ số đầy đủ: [`metrics/`](outputs/baseline_moscow_v1/metrics/) (`bootstrap_summary.csv`, `paired_bootstrap_test.csv`, `monthly_metrics.csv`, `calibration_bins_test.csv`, `slice_metrics_test.csv`, `pca_variance_log.csv`, `timings.csv`)

> **Lưu ý về phiên bản báo cáo:** Bản báo cáo này thay thế bản baseline thử nghiệm trước đây (vốn cắt mốc `00:00 UTC` và đánh giá chủ yếu trên panel trùng khách `outputs/baseline_ladder_v1/`). Phiên bản **Moscow v1 (`outputs/baseline_moscow_v1/`)** khắc phục triệt để **88.976 sự kiện rò rỉ ranh giới 3 giờ**, tách bạch hoàn toàn nhánh **thuần hành vi (không dùng target)** khỏi nhánh **Target-Lag**, bổ sung **nội dung ngữ nghĩa `DIALOG` embedding `768d` (fit PCA riêng trên train từng fold)**, và áp dụng bắt buộc chuẩn **`0%` trùng `client_id`** xuyên suốt cả Validation lẫn Locked Test 3 tháng cuối.

---

## Tóm tắt Điều hành (Executive Summary)

Trên **tập Locked Test 3 tháng cuối (`2022-11`, `2022-12`, `2023-01` — `300.672` lượt chấm điểm, `2.301` ca mua hàng, base rate `0,765%`, `0%` trùng khách với tập train)** với ngân sách chiến dịch chính **Top 1% (`3.007` cuộc gọi / 3 tháng ≈ `1.003` cuộc gọi/tháng)**:

| Cấp bậc / Mô hình | Dùng Target? | Số biến | PR-AUC `[95% CI]` | ROC-AUC | Lift@1% `[95% CI]` | Số khách mua trúng @1% (`P@1%`) | Trúng @5% (`R@5%`) | Brier (`BSS`) |
| --- | :---: | ---: | --- | ---: | --- | ---: | ---: | --- |
| `L0_base_rate` (Mốc sàn ngẫu nhiên) | Không | `0` | `0,0077` `[0,0073; 0,0081]` | `0,5022` | `0,91×` `[0,49; 1,38]` | `21` (`0,70%`) | `103` (`4,48%`) | `0,007608` (`0,00%`) |
| `L1_rule_target_lag` (Luật 1/3/6m mua cũ) | **Có** | `3` | `0,0168` `[0,0147; 0,0194]` | `0,5914` | `7,39×` `[6,26; 8,70]` | `170` (`5,65%`) | `511` (`22,21%`) | — |
| `L2_rule_trx_rfm` (Luật thuần TRX) | Không | `6` | `0,0176` `[0,0161; 0,0195]` | `0,6694` | `4,43×` `[3,68; 5,30]` | `102` (`3,39%`) | `388` (`16,86%`) | — |
| `L3_logreg_rfm` (LogReg L2 RFM nhỏ) | Không | `8` | `0,0207` `[0,0191; 0,0229]` | `0,7394` | `4,74×` `[3,89; 5,82]` | `109` (`3,62%`) | `397` (`17,25%`) | **`0,007556`** (`+0,69%`) |
| `L4_lgbm_trx` (LightGBM thuần TRX) | Không | `63` | `0,0215` `[0,0198; 0,0236]` | `0,7476` | `4,78×` `[3,93; 5,74]` | `110` (`3,66%`) | `412` (`17,91%`) | `0,007811` (`-2,67%`) |
| `L4_cb_trx` (CatBoost thuần TRX) | Không | `63` | `0,0205` `[0,0188; 0,0225]` | `0,7181` | `5,65×` `[4,63; 6,57]` | `130` (`4,32%`) | `382` (`16,60%`) | `0,007645` (`-0,49%`) |
| `L5c_lgbm_dialog_meta` (LGBM + GEO + DIALOG meta) | Không | `112` | `0,0284` `[0,0259; 0,0318]` | **`0,7755`** | `6,78×` `[5,76; 7,76]` | `156` (`5,19%`) | `533` (`23,16%`) | `0,007749` (`-1,86%`) |
| **`L5d_cb_dialog_emb` (Chính — CatBoost Đa nguồn)** | **Không** | **`139`** | **`0,0284`** `[0,0254; 0,0327]` | `0,7456` | **`7,87×`** `[6,76; 9,01]` | **`181`** (`6,02%`) | `483` (`20,99%`) | `0,007606` (`+0,02%`) |
| **`L5d_cb_dialog_emb_cal` (Chính + Calibrated)** | **Không** | **`139`** | **`0,0284`** `[0,0254; 0,0327]` | `0,7456` | **`7,87×`** `[6,76; 9,01]` | **`181`** (`6,02%`) | `483` (`20,99%`) | **`0,007589`** (`+0,25%`) |
| *`Ref_lgbm_target_lag` (Nhánh phụ LGBM + Target)* | ***Có*** | *`152`* | ***`0,0328`*** `[0,0298; 0,0367]` | ***`0,7925`*** | *`7,87×`* `[6,86; 9,03]` | *`181`* (`6,02%`) | ***`623`*** (`27,08%`) | *`0,007714`* (`-1,40%`) |
| *`Ref_cb_target_lag` (Nhánh phụ CB + Target)* | ***Có*** | *`152`* | ***`0,0327`*** `[0,0291; 0,0375]` | *`0,7603`* | ***`9,30×`*** `[8,13; 10,52]` | ***`214`*** (`7,12%`) | *`548`* (`23,82%`) | ***`0,007579`*** (`+0,38%`) |

**5 kết luận cốt lõi đã được chứng minh bằng Paired Bootstrap 1.000 lần (theo cụm `client_id`):**
1. **Đa nguồn hành vi tạo giá trị vượt trội so với thuần giao dịch (`TRX`):** Trên CatBoost, mô hình chính đa nguồn `L5d_cb_dialog_emb` vượt `L4_cb_trx` tới **`+0,0079` PR-AUC (+38,7%, $p < 0,001$)**, **`+0,0275` ROC-AUC ($p < 0,001$)**, **`+2,22×` Lift@1% ($p < 0,001$)** (bắt thêm **`+51` khách mua** từ `130 → 181` trên cùng ngân sách `3.007` cuộc gọi) và cải thiện Brier Score có ý nghĩa ($p = 0,020$).
2. **`DIALOG` là nguồn tín hiệu mạnh nhất ngoài `TRX` — và nội dung ngữ nghĩa (`embedding`) mang lại giá trị gia tăng thực sự vượt lên trên metadata hoạt động:**
   - Thêm `DIALOG metadata` (`L5b → L5c`) tạo cú nhảy lớn nhất ở cả CatBoost (`+0,0040` AP, $p = 0,002$; `+0,0245` AUC, $p < 0,001$) lẫn LightGBM (`+0,0062` AP, $p < 0,001$; `+1,43×` Lift@1%, $p = 0,014$).
   - Thêm **`DIALOG` embedding PCA + norms + topic-shift (`L5c → L5d`)** tiếp tục giúp CatBoost tăng có ý nghĩa thống kê: **`+0,0029` AP ($p = 0,014$)**, **`+0,0086` AUC ($p = 0,028$)**, **`+0,96×` Lift@1%** (`159 → 181` khách mua, `Lift@0,5%` vọt lên **`9,90×`**). Ngược lại, LightGBM mặc định không hấp thụ được 27 chiều PCA dày đặc có ~55% `NaN` (`ΔAP = -0,0013`, $p = 0,244$) do thiếu regularization.
3. **`GEO` đóng góp yếu khi đứng một mình, chỉ có tín hiệu từ nội dung không gian (`geohash`):** `GEO metadata` hoạt động (`L5a`) không tạo cải thiện (thậm chí nhiễu nhẹ trên cả 2 họ mô hình); chỉ khi thêm `geohash_4/5/6` (`L5b`) mới ghi nhận cải thiện có ý nghĩa trên LightGBM (`+0,0015` AP, $p = 0,006$; `+1,04×` Lift@1%, $p = 0,028$) và tăng nhẹ trên CatBoost (`+0,0011` AP, $p = 0,266$).
4. **Mô hình thuần hành vi (`L5d`) thắng áp đảo luật Target-Lag (`L1`):** Dù **hoàn toàn không dùng lịch sử nhãn mua**, `L5d_cb_dialog_emb` vẫn vượt `L1_rule_target_lag` tới **`+0,0116` PR-AUC (+69%, $p < 0,001$)**, **`+0,1541` ROC-AUC ($p < 0,001$)** và bắt được nhiều khách mua hơn ngay ở Top 1% (`181` vs `170` người mua).
5. **Nhánh phụ `+ Target-Lag` mở thêm trần hiệu năng rõ rệt khi bài toán cho phép dùng lịch sử mua:** Bổ sung 13 biến `prior_*` (`1/3/6/all`, $m < T$) giúp cả hai họ mô hình tăng mạnh: `Ref_cb_target_lag` đạt **`Lift@1% = 9,30×`** (`214` khách mua, `Lift@0,5% = 11,29×`, `ΔAP = +0,0042`, $p = 0,010$), còn `Ref_lgbm_target_lag` đạt **`ROC-AUC = 0,7925`**, **`PR-AUC = 0,0328`** (`ΔAP = +0,0056`, $p < 0,001$) và bắt tới **`27,08%`** toàn bộ người mua ở Top 5%.

---

## Chương 1. Kiểm toán Thời gian (`UTC` vs `Moscow UTC+3`) & Thiết kế Dữ liệu Chống Rò Rỉ

### 1.1. Bằng chứng rò rỉ ranh giới 3 giờ khi dùng mốc `UTC`
Dữ liệu sự kiện gốc (`detail/trx`, `detail/geo`, `detail/dialog`) lưu `event_time` theo múi giờ **`UTC`**, trong khi nhãn mục tiêu `targets.mon` (tháng mua sản phẩm của ngân hàng Nga) chốt mốc tại **`00:00:00 Europe/Moscow (UTC+3)`**, tức **`21:00:00 UTC` ngày cuối tháng trước**.

Kết quả kiểm toán chính xác trên toàn bộ `105.277.859` sự kiện gốc ([`moscow_audit.json`](outputs/baseline_moscow_v1/moscow_audit.json)):

| Nguồn dữ liệu | Tổng số sự kiện | Khoảng thời gian `UTC` gốc | Khoảng thời gian `Moscow (UTC+3)` | Sự kiện bị rò rỉ nếu cắt `00:00 UTC` | Số cặp `(client, month)` bị nhiễm |
| --- | ---: | --- | --- | ---: | ---: |
| `detail/trx` | `38.764.930` | `2020-12-31 21:00:02` .. `2022-12-31 20:59:58` | **`2021-01-01 00:00:02` .. `2022-12-31 23:59:58`** | **`31.013`** | `27.277` |
| `detail/geo` | `66.295.724` | `2021-12-11 19:27:07` .. `2022-12-31 20:59:58` | **`2021-12-11 22:27:07` .. `2022-12-31 23:59:58`** | **`57.942`** | `45.729` |
| `detail/dialog` | `217.205` | `2021-02-10 09:36:07` .. `2022-12-31 20:25:11` | **`2021-02-10 12:36:07` .. `2022-12-31 23:25:11`** | **`21`** | `21` |
| **Tổng cộng** | **`105.277.859`** | Kết thúc đúng `20:59:58 UTC` ngày `31/12/2022` | **Tròn tuyệt đối `00:00:02` → `23:59:58 MSK`** | **`88.976`** | — |

Dấu hiệu `min_utc = 2020-12-31 21:00:02` và `max_utc = 2022-12-31 20:59:58` chứng minh tuyệt đối dữ liệu được trích xuất theo múi giờ **Moscow (`UTC+3`)**. Do đó, toàn bộ pipeline đặc trưng & embedding áp dụng quy tắc cứng:

$$\text{event\_time\_msk} = \text{event\_time\_utc} + 3\text{h}, \qquad \text{event\_time\_msk} < \text{YYYY-MM-01 00:00:00 MSK}$$

tương đương `event_time_utc < prev_month_last_day 21:00:00 UTC`, loại bỏ 100% rò rỉ ranh giới 3 giờ.

### 1.2. Kiến trúc 3 cửa sổ (`30d`, `90d`, `all`) & phân tách chặt các nhóm đặc trưng
Toàn bộ đặc trưng được tính đồng nhất trên 3 cửa sổ nhìn lại **`30d`**, **`90d`** và **`all`** (từ `2021-01-01 00:00:00 MSK` nghiêm ngặt trước `cutoff_msk`), lưu trong [`panel_moscow.parquet`](outputs/baseline_moscow_v1/panel_moscow.parquet) (`1.002.240` dòng × `135` cột) và [`dialog_emb_moscow.npz`](outputs/baseline_moscow_v1/dialog_emb_moscow.npz):

| Nhóm đặc trưng | Số cột | Thành phần chính | Quy tắc chống rò rỉ / Xử lý `NaN` |
| --- | ---: | --- | --- |
| **Định danh & Nhãn (Cấm đưa vào feature)** | `10` | `client_id`, `fold`, `month`, `cutoff_msk`, `y_any`, `target_1..4`, `prior_months_observed` | Khóa cứng trong `FORBIDDEN_COLS`, assert trước mọi thí nghiệm. |
| **1. `TRX` thuần giao dịch** | `63` | 3 cửa sổ (`30d/90d/all`) × 20 biến (tần suất, `active_days`, recency, tenure, số GD mã chính `currency=11`, ngoại tệ & tỷ lệ ngoại tệ, GD âm, `abs_sum`, `log1p`, `signed_log_mean`, `median`, `max`, `event_type/subtype_nunique`, `top_share`, `dst_type11/12_nunique`) + 3 biến liên cửa sổ (`trx_rate_per_30d_tenure`, `trx_share_30d_of_90d`, `trx_first_activated_30d`). | Cờ/đếm điền `0`; recency, median, share giữ `NaN` có ngữ nghĩa. |
| **2. `GEO metadata` (Hoạt động)** | `18` | 3 cửa sổ × 6 biến (`has_event`, `event_count`, `log1p_event_count`, `active_days`, `days_since_last_event`, `days_since_first_event`). | Đo riêng hành vi mở app/phát sinh ping, không gồm tọa độ. |
| **3. `GEO content` (Nội dung không gian)** | `12` | 3 cửa sổ × 4 biến (`geohash_4/5/6_nunique`, `top_geohash_6_share`). | Tách riêng để kiểm định nội dung không gian có thêm giá trị ngoài metadata không. |
| **4. `DIALOG metadata` + `cold_start`** | `19` | 3 cửa sổ × 6 biến hoạt động hội thoại + cờ `cold_start` (`1` nếu không có sự kiện nào ở cả 3 nguồn trước `cutoff_msk`). | Đo riêng tần suất/độ tươi tương tác tư vấn. |
| **5. `DIALOG embedding` (Nội dung ngữ nghĩa)** | `+27` | Mean-pool `768d` (`all`: `393.021` cặp, `90d`: `138.038` cặp) → **`PCA(16)` cho `all`** + **`PCA(8)` cho `90d`** + **L2 norm `all`**, **L2 norm `90d`**, **Cosine similarity `90d vs all`** (đo độ dịch chuyển chủ đề hội thoại gần đây). | `StandardScaler + PCA` fit **nghiêm ngặt chỉ trên tập Train của từng fold (`fold != k`)**; khách chưa có dialog giữ nguyên `NaN` (không impute 0 để không trộn lẫn với activity metadata). |
| **6. `Target-Lag` (CHỈ dùng ở `L1` & `Ref_*`)** | `13` | `prior_1m_bought_any`, `prior_3m_buy_months/total_products`, `prior_6m_buy_months/total_products`, `prior_all_bought_any/buy_months/distinct_products`, `prior_months_since_last_buy`, `prior_all_bought_p1..p4` ($m < T$). | Bị cấm tuyệt đối khỏi toàn bộ nhánh chính `L2..L5d` (assert tự động trong code). |

### 1.3. Phân hoạch Thời gian & Bất biến `0%` Trùng Khách (`5-Fold OOT + OOC`)
- **Burn-in (`2022-02`, `2022-03`):** Không dùng để train/eval; bảo đảm từ `2022-04` trở đi `GEO` đã đủ cửa sổ `≥ 90` ngày (vì `GEO` bắt đầu `11/12/2021`) và mọi tháng Val/Test đều có đầy đủ cửa sổ target-lag `1/3/6m`.
- **Phân bố `10` tháng mô hình hóa (`100.224` khách/tháng, tổng `1.002.240` dòng):**

| Giai đoạn | Các tháng | Số dòng | Số ca dương (`y_any = 1`) | Base rate | Vai trò |
| --- | --- | ---: | ---: | ---: | --- |
| **Train (cho Val)** | `2022-04` (`1.698`), `05` (`1.100`), `06` (`1.177`), `07` (`957`), `08` (`1.132`) | `501.120` | `6.064` | `1,210%` | Huấn luyện Stage 1 (`fold != k`) |
| **Validation (Stage 1)** | `2022-09` (`1.013`), `2022-10` (`868`) | `200.448` | `1.881` | `0,938%` | Đánh giá trung gian & fit hậu hiệu chỉnh (`fold == k`) |
| **Train (cho Test)** | `2022-04` .. `2022-10` (7 tháng) | `701.568` | `7.945` | `1,132%` | Huấn luyện Stage 2 (`fold != k`) |
| **Locked Test (Stage 2)** | `2022-11` (`867`), `2022-12` (`759`), `2023-01` (`675`) | **`300.672`** | **`2.301`** | **`0,765%`** | **Đánh giá cuối cùng (3 tháng liên tiếp, `fold == k`)** |

- **Bất biến `0%` trùng `client_id`:** Đã xác minh trên toàn bộ dữ liệu gốc rằng 100% `100.224` khách hàng phân bố tách rời hoàn toàn theo `client_split/fold=0..4` (`0` bản ghi chéo fold trên cả `trx`, `geo`, `dialog`). Với mỗi fold $k \in \{0..4\}$, mô hình chỉ huấn luyện trên `fold != k` (~`80.179` khách) và chấm điểm cho `fold == k` (~`20.045` khách hoàn toàn mới). Khi ghép đủ 5 fold, ta thu được **100% độ phủ (`100.224` khách/tháng)** nhưng **nghiêm ngặt `0%` trùng khách hàng giữa train và eval**.
- **Độ ổn định của PCA `train-only` across 5 folds ([`pca_variance_log.csv`](outputs/baseline_moscow_v1/metrics/pca_variance_log.csv)):**
  - `PCA(16)` trên `dialog all` giữ trung bình **`84,20%`** phương sai ở Val (`84,12%–84,27%`) và **`84,15%`** ở Test (`84,07%–84,22%`); độ phủ khách có dialog tăng dần theo lịch sử từ `34,96%` (Train Val) → `41,25%` (Val) → `44,95%` (Test).
  - `PCA(8)` trên `dialog 90d` giữ trung bình **`82,22%`** phương sai ở Val và **`82,17%`** ở Test.
- **Chính sách Imbalance & Thống kê:** Không dùng SMOTE/resample (giữ nguyên phân bố tự nhiên kèm `cold_start`); phá hòa tất định `SHA-256(42, client_id)`; mọi khoảng tin cậy 95% CI và $p$-value đều tính bằng **1.000 lần Client-Clustered Bootstrap** (lấy mẫu có hoàn lại theo `client_id` để phản ánh đúng tương quan lặp lại của cùng khách hàng qua 3 tháng Test).

---

## Chương 2. Kết quả Toàn bộ Bậc thang Baseline (`L0 → L5d + Ref`)

### 2.1. Kết quả trên Locked Test 3 tháng cuối (`2022-11`, `2022-12`, `2023-01` — `300.672` dòng, `2.301` ca dương, `0%` trùng KH)

#### A. Chỉ số gộp 3 tháng Test (kèm 95% CI Client-Clustered Bootstrap 1.000 lần)

| Exp ID | Bậc / Mô tả | Số biến | PR-AUC `[95% CI]` | ROC-AUC `[95% CI]` | Lift@0,5% | Lift@1% `[95% CI]` | Lift@5% (`R@5%`) | LogLoss (`Skill`) | Brier (`BSS`) | CalRatio | ECE |
| --- | --- | ---: | --- | --- | ---: | --- | --- | --- | --- | ---: | ---: |
| `L0_base_rate` | Bậc 0: Base rate | `0` | `0,0077` `[0,0073; 0,0081]` | `0,5022` `[0,4900; 0,5138]` | `1,13×` | `0,91×` `[0,49; 1,38]` | `0,90×` (`4,48%`) | `0,04559` (`0,00%`) | `0,007608` (`0,00%`) | `1,480` | `0,00367` |
| `L1_rule_target_lag` | Bậc 1: Rule Target-Lag *(có target)* | `3` | `0,0168` `[0,0147; 0,0194]` | `0,5914` `[0,5820; 0,6009]` | `7,82×` | `7,39×` `[6,26; 8,70]` | `4,44×` (`22,21%`) | — | — | — | — |
| `L2_rule_trx_rfm` | Bậc 2: Rule thuần TRX | `6` | `0,0176` `[0,0161; 0,0195]` | `0,6694` `[0,6580; 0,6815]` | `5,56×` | `4,43×` `[3,68; 5,30]` | `3,37×` (`16,86%`) | — | — | — | — |
| `L3_logreg_rfm` | Bậc 3: LogReg L2 RFM nhỏ | `8` | `0,0207` `[0,0191; 0,0229]` | `0,7394` `[0,7295; 0,7489]` | `5,39×` | `4,74×` `[3,89; 5,82]` | `3,45×` (`17,25%`) | `0,04221` (`+7,43%`) | **`0,007556`** (`+0,69%`) | `1,070` | **`0,00112`** |
| `L4_lgbm_trx` | Bậc 4: LGBM thuần TRX | `63` | `0,0215` `[0,0198; 0,0236]` | `0,7476` `[0,7378; 0,7569]` | `5,21×` | `4,78×` `[3,93; 5,74]` | `3,58×` (`17,91%`) | `0,04302` (`+5,65%`) | `0,007811` (`-2,67%`) | `1,206` | `0,00287` |
| `L5a_lgbm_geo_meta` | Bậc 5a: LGBM + GEO meta | `81` | `0,0207` `[0,0192; 0,0227]` | `0,7483` `[0,7386; 0,7574]` | `5,04×` | `4,30×` `[3,55; 5,13]` | `3,39×` (`16,95%`) | `0,04333` (`+4,97%`) | `0,007891` (`-3,72%`) | `1,300` | `0,00290` |
| `L5b_lgbm_geo_full` | Bậc 5b: LGBM + GEO full | `93` | `0,0222` `[0,0204; 0,0246]` | `0,7520` `[0,7425; 0,7612]` | `5,73×` | `5,35×` `[4,49; 6,35]` | `3,69×` (`18,43%`) | `0,04306` (`+5,55%`) | `0,007881` (`-3,59%`) | `1,290` | `0,00287` |
| `L5c_lgbm_dialog_meta` | Bậc 5c: LGBM + DIALOG meta | `112` | `0,0284` `[0,0259; 0,0318]` | **`0,7755`** `[0,7663; 0,7853]` | `8,78×` | `6,78×` `[5,76; 7,76]` | `4,63×` (`23,16%`) | **`0,04183`** (`+8,25%`) | `0,007749` (`-1,86%`) | `1,251` | `0,00225` |
| `L5d_lgbm_dialog_emb` | Bậc 5d: LGBM + DIALOG emb | `139` | `0,0272` `[0,0247; 0,0303]` | `0,7744` `[0,7646; 0,7839]` | `6,95×` | `6,04×` `[5,07; 7,17]` | `4,48×` (`22,38%`) | `0,04205` (`+7,77%`) | `0,007796` (`-2,47%`) | `1,244` | `0,00228` |
| `L5d_lgbm_dialog_emb_cal` | Bậc 5d: LGBM + Calibrated | `139` | `0,0272` `[0,0247; 0,0303]` | `0,7744` `[0,7646; 0,7839]` | `6,95×` | `6,04×` `[5,07; 7,17]` | `4,48×` (`22,38%`) | `0,04185` (`+8,21%`) | `0,007700` (`-1,22%`) | `0,947` | `0,00204` |
| `L4_cb_trx` | Bậc 4: CatBoost thuần TRX | `63` | `0,0205` `[0,0188; 0,0225]` | `0,7181` `[0,7066; 0,7282]` | `6,95×` | `5,65×` `[4,63; 6,57]` | `3,32×` (`16,60%`) | `0,04346` (`+4,68%`) | `0,007645` (`-0,49%`) | `1,032` | `0,00291` |
| `L5a_cb_geo_meta` | Bậc 5a: CatBoost + GEO meta | `81` | `0,0204` `[0,0185; 0,0229]` | `0,7116` `[0,7014; 0,7224]` | `6,78×` | `5,74×` `[4,81; 6,63]` | `3,26×` (`16,30%`) | `0,04391` (`+3,69%`) | `0,007680` (`-0,95%`) | `1,014` | `0,00341` |
| `L5b_cb_geo_full` | Bậc 5b: CatBoost + GEO full | `93` | `0,0215` `[0,0195; 0,0241]` | `0,7125` `[0,7012; 0,7230]` | `7,82×` | `5,78×` `[4,87; 6,71]` | `3,54×` (`17,69%`) | `0,04371` (`+4,12%`) | `0,007667` (`-0,78%`) | `1,043` | `0,00296` |
| `L5c_cb_dialog_meta` | Bậc 5c: CatBoost + DIALOG meta | `112` | `0,0255` `[0,0230; 0,0290]` | `0,7370` `[0,7272; 0,7472]` | `8,78×` | `6,91×` `[5,89; 8,04]` | `4,03×` (`20,17%`) | `0,04302` (`+5,63%`) | `0,007638` (`-0,40%`) | `0,980` | `0,00301` |
| **`L5d_cb_dialog_emb`** | **Bậc 5d: CatBoost + DIALOG emb** | **`139`** | **`0,0284`** `[0,0254; 0,0327]` | `0,7456` `[0,7352; 0,7551]` | **`9,90×`** | **`7,87×`** `[6,76; 9,01]` | `4,20×` (`20,99%`) | `0,04237` (`+7,06%`) | `0,007606` (`+0,02%`) | **`0,969`** | `0,00248` |
| **`L5d_cb_dialog_emb_cal`** | **Bậc 5d: CatBoost + Calibrated** | **`139`** | **`0,0284`** `[0,0254; 0,0327]` | `0,7456` `[0,7352; 0,7551]` | **`9,90×`** | **`7,87×`** `[6,76; 9,01]` | `4,20×` (`20,99%`) | `0,04243` (`+6,93%`) | **`0,007589`** (`+0,25%`) | `0,881` | `0,00250` |
| *`Ref_lgbm_target_lag`* | *Nhánh phụ: LGBM + Target-Lag* | *`152`* | ***`0,0328`*** `[0,0298; 0,0367]` | ***`0,7925`*** `[0,7830; 0,8016]` | *`9,12×`* | *`7,87×`* `[6,86; 9,03]` | ***`5,41×`*** (`27,08%`) | ***`0,04101`*** (`+10,06%`) | *`0,007714`* (`-1,40%`) | *`1,205`* | *`0,00172`* |
| *`Ref_cb_target_lag`* | *Nhánh phụ: CatBoost + Target-Lag* | *`152`* | ***`0,0327`*** `[0,0291; 0,0375]` | *`0,7603`* `[0,7501; 0,7705]` | ***`11,29×`*** | ***`9,30×`*** `[8,13; 10,52]` | *`4,76×`* (`23,82%`) | *`0,04185`* (`+8,21%`) | ***`0,007579`*** (`+0,38%`) | *`0,914`* | *`0,00235`* |

#### B. Ma trận nhầm lẫn tại ngân sách Top 1% (`K = 3.007` cuộc gọi / `300.672` lượt chấm — Locked Test)

| Mô hình | TP (Gọi & Mua) | FP (Gọi & Không mua) | FN (Bỏ sót mua) | TN (Không gọi & Không mua) | Precision@1% | Recall@1% | Hiệu quả kinh doanh (Số cuộc gọi / 1 khách mua) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `L0_base_rate` | `21` | `2.986` | `2.280` | `295.385` | `0,70%` | `0,91%` | ~`143` cuộc gọi / 1 khách mua |
| `L2_rule_trx_rfm` | `102` | `2.905` | `2.199` | `295.466` | `3,39%` | `4,43%` | ~`29` cuộc gọi / 1 khách mua |
| `L3_logreg_rfm` | `109` | `2.898` | `2.192` | `295.473` | `3,62%` | `4,74%` | ~`28` cuộc gọi / 1 khách mua |
| `L4_cb_trx` | `130` | `2.877` | `2.171` | `295.494` | `4,32%` | `5,65%` | ~`23` cuộc gọi / 1 khách mua |
| `L5c_cb_dialog_meta` | `159` | `2.848` | `2.142` | `295.523` | `5,29%` | `6,91%` | ~`19` cuộc gọi / 1 khách mua |
| `L1_rule_target_lag` *(có target)* | `170` | `2.837` | `2.131` | `295.534` | `5,65%` | `7,39%` | ~`18` cuộc gọi / 1 khách mua |
| **`L5d_cb_dialog_emb` (Chính)** | **`181`** | **`2.826`** | **`2.120`** | **`295.545`** | **`6,02%`** | **`7,87%`** | **~`17` cuộc gọi / 1 khách mua** |
| *`Ref_cb_target_lag` (Nhánh phụ)* | ***`214`*** | *`2.793`* | *`2.087`* | *`295.578`* | ***`7,12%`*** | ***`9,30%`*** | ***~`14` cuộc gọi / 1 khách mua*** |

#### C. Phân rã từng tháng trên Locked Test (`2022-11`, `2022-12`, `2023-01`)

| Mô hình | `2022-11` (`867` pos, `0,865%`) AP \| Lift@1% (`Hits`) | `2022-12` (`759` pos, `0,757%`) AP \| Lift@1% (`Hits`) | `2023-01` (`675` pos, `0,673%`) AP \| Lift@1% (`Hits`) | Trung bình Macro 3 tháng (AP \| AUC \| Lift@1%) |
| --- | --- | --- | --- | --- |
| `L0_base_rate` | `0,0087` \| `1,15×` (`10`) | `0,0075` \| `0,79×` (`6`) | `0,0068` \| `0,74×` (`5`) | `0,0077` \| `0,5021` \| `0,89×` |
| `L1_rule_target_lag` | `0,0161` \| `7,38×` (`64`) | `0,0149` \| `6,06×` (`46`) | `0,0216` \| **`10,07×`** (`68`) | `0,0176` \| `0,5937` \| `7,83×` |
| `L2_rule_trx_rfm` | `0,0181` \| `3,80×` (`33`) | `0,0158` \| `3,82×` (`29`) | `0,0202` \| `5,77×` (`39`) | `0,0181` \| `0,6728` \| `4,46×` |
| `L3_logreg_rfm` | `0,0241` \| `5,42×` (`47`) | `0,0203` \| `4,34×` (`33`) | `0,0179` \| `4,44×` (`30`) | `0,0208` \| `0,7379` \| `4,73×` |
| `L4_lgbm_trx` | `0,0254` \| `4,73×` (`41`) | `0,0224` \| `6,45×` (`49`) | `0,0177` \| `3,40×` (`23`) | `0,0219` \| `0,7472` \| `4,86×` |
| `L5c_lgbm_dialog_meta` | **`0,0365`** \| `7,38×` (`64`) | `0,0318` \| `7,50×` (`57`) | `0,0197` \| `4,29×` (`29`) | **`0,0293`** \| **`0,7749`** \| `6,39×` |
| `L5d_lgbm_dialog_emb` | `0,0319` \| `6,11×` (`53`) | `0,0304` \| `7,50×` (`57`) | `0,0203` \| `4,15×` (`28`) | `0,0275` \| `0,7742` \| `5,92×` |
| `L4_cb_trx` | `0,0241` \| `5,53×` (`48`) | `0,0196` \| `5,66×` (`43`) | `0,0183` \| `5,77×` (`39`) | `0,0207` \| `0,7167` \| `5,66×` |
| `L5b_cb_geo_full` | `0,0261` \| `5,76×` (`50`) | `0,0234` \| `6,98×` (`53`) | `0,0169` \| `4,74×` (`32`) | `0,0221` \| `0,7110` \| `5,83×` |
| `L5c_cb_dialog_meta` | `0,0299` \| `7,95×` (`69`) | `0,0287` \| `8,69×` (`66`) | `0,0195` \| `3,85×` (`26`) | `0,0261` \| `0,7361` \| `6,83×` |
| **`L5d_cb_dialog_emb`** | `0,0327` \| **`7,95×`** (`69`) | **`0,0339`** \| **`9,48×`** (`72`) | `0,0202` \| `5,03×` (`34`) | `0,0289` \| `0,7445` \| **`7,49×`** |
| *`Ref_lgbm_target_lag`* | *`0,0355`* \| *`8,18×`* (`71`) | *`0,0361`* \| *`9,48×`* (`72`) | ***`0,0287`*** \| *`6,81×`* (`46`) | *`0,0334`* \| ***`0,7927`*** \| *`8,16×`* |
| *`Ref_cb_target_lag`* | *`0,0347`* \| ***`8,87×`*** (`77`) | ***`0,0406`*** \| ***`10,01×`*** (`76`) | *`0,0268`* \| **`8,59×`** (`58`) | ***`0,0340`*** \| *`0,7600`* \| ***`9,16×`*** |

> **Nhận xét quan trọng về độ trôi mùa vụ (`2023-01`):** Tháng `2023-01` (nghỉ Tết Dương lịch kéo dài tại Nga, tỷ lệ mua tụt về đáy `0,673%`, giảm 22% so với `2022-11`) khiến các mô hình thuần hành vi suy giảm ở Top 1% (`L5d_cb` từ `7,95× / 9,48×` xuống `5,03×`), trong khi tín hiệu **khách tái mua có sẵn (`L1` đạt `10,07×`, `Ref_cb` đạt `8,59×`)** lại bền vững nhất ở tháng thấp điểm này. Điều này nhấn mạnh giá trị của việc đánh giá trên **3 tháng Test liên tiếp (`2.301` ca dương)** thay vì chỉ kết luận trên một tháng đơn lẻ.

### 2.2. Kết quả đối chứng trên Validation (`2022-09`, `2022-10` — `200.448` dòng, `1.881` ca dương, `0%` trùng KH)

Xu hướng trên Validation hoàn toàn nhất quán với Locked Test, xác nhận không có hiện tượng chọn mô hình theo may rủi (overfitting to test):

| Exp ID | Số biến | PR-AUC `[95% CI]` | ROC-AUC | Lift@1% `[95% CI]` | Hits@1% (`K=2.005`) | LogLoss (`Skill`) | Brier (`BSS`) |
| --- | ---: | --- | ---: | --- | ---: | --- | --- |
| `L0_base_rate` | `0` | `0,0093` `[0,0087; 0,0098]` | `0,4989` | `0,80×` `[0,41; 1,22]` | `15` | `0,05349` (`0,00%`) | `0,009303` (`0,00%`) |
| `L1_rule_target_lag` | `3` | `0,0194` `[0,0168; 0,0234]` | `0,5895` | `6,17×` `[5,07; 7,32]` | `116` | — | — |
| `L2_rule_trx_rfm` | `6` | `0,0184` `[0,0169; 0,0203]` | `0,6481` | `2,87×` `[2,12; 3,66]` | `54` | — | — |
| `L3_logreg_rfm` | `8` | `0,0226` `[0,0207; 0,0251]` | `0,7255` | `3,77×` `[2,89; 4,62]` | `71` | `0,05022` (`+6,10%`) | **`0,009249`** (`+0,58%`) |
| `L4_lgbm_trx` | `63` | `0,0235` `[0,0217; 0,0257]` | `0,7379` | `3,45×` `[2,77; 4,40]` | `65` | `0,05129` (`+4,12%`) | `0,009638` (`-3,60%`) |
| `L5b_lgbm_geo_full` | `93` | `0,0243` `[0,0223; 0,0269]` | `0,7362` | `4,52×` `[3,70; 5,51]` | `85` | `0,05182` (`+3,12%`) | `0,009736` (`-4,65%`) |
| `L5c_lgbm_dialog_meta` | `112` | `0,0291` `[0,0269; 0,0323]` | `0,7666` | `5,21×` `[4,24; 6,26]` | `98` | `0,05086` (`+4,90%`) | `0,009811` (`-5,46%`) |
| `L5d_lgbm_dialog_emb` | `139` | `0,0288` `[0,0265; 0,0317]` | **`0,7710`** | `4,41×` `[3,53; 5,32]` | `83` | `0,05049` (`+5,60%`) | `0,009748` (`-4,78%`) |
| `L4_cb_trx` | `63` | `0,0267` `[0,0240; 0,0304]` | `0,7224` | `6,11×` `[5,08; 7,14]` | `115` | `0,05073` (`+5,16%`) | `0,009300` (`+0,03%`) |
| `L5b_cb_geo_full` | `93` | `0,0229` `[0,0208; 0,0256]` | `0,7083` | `4,89×` `[3,97; 5,94]` | `92` | `0,05202` (`+2,74%`) | `0,009473` (`-1,82%`) |
| `L5c_cb_dialog_meta` | `112` | `0,0285` `[0,0257; 0,0323]` | `0,7212` | `6,27×` `[5,33; 7,46]` | `118` | `0,05139` (`+3,91%`) | `0,009437` (`-1,44%`) |
| **`L5d_cb_dialog_emb`** | **`139`** | **`0,0299`** `[0,0270; 0,0337]` | `0,7297` | **`6,75×`** `[5,69; 7,79]` | **`127`** | `0,05061` (`+5,38%`) | `0,009371` (`-0,73%`) |
| *`Ref_lgbm_target_lag`* | *`152`* | ***`0,0322`*** `[0,0295; 0,0359]` | ***`0,7777`*** | *`5,42×`* `[4,45; 6,48]` | *`102`* | ***`0,05000`*** (`+6,52%`) | *`0,009676`* (`-4,01%`) |
| *`Ref_cb_target_lag`* | *`152`* | *`0,0308`* `[0,0278; 0,0347]` | *`0,7391`* | ***`6,70×`*** `[5,59; 7,75]` | *`126`* | *`0,05035`* (`+5,87%`) | *`0,009388`* (`-0,91%`) |

---

## Chương 3. Kiểm định Thống kê Paired Bootstrap & Phân rã Đóng góp Nguồn Dữ liệu

Toàn bộ **21 cặp kiểm định Paired Client-Clustered Bootstrap (`1.000` lần lặp, theo cụm `client_id`)** trên Locked Test (`300.672` dòng, [`paired_bootstrap_test.csv`](outputs/baseline_moscow_v1/metrics/paired_bootstrap_test.csv)):

| Nhóm kiểm định | Cặp so sánh (`B − A`) | `ΔPR-AUC` (`p`) | `ΔROC-AUC` (`p`) | `ΔLift@1%` (`p`) | `ΔLogLoss` (`p`) | `ΔBrier` (`p`) | Kết luận thống kê |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Bậc 0 → 3** | `L1 (Rule Target-Lag) − L0` | **`+0,0091`** (`<0,001`) | **`+0,0892`** (`<0,001`) | **`+6,47×`** (`<0,001`) | — | — | Lịch sử mua cũ cực mạnh ở đỉnh Top 1% |
| | `L2 (Rule TRX) − L0` | **`+0,0099`** (`<0,001`) | **`+0,1672`** (`<0,001`) | **`+3,52×`** (`<0,001`) | — | — | Luật RFM thuần TRX vượt trội mốc sàn |
| | `L3 (LogReg RFM) − L2 (Rule TRX)` | **`+0,0032`** (`<0,001`) | **`+0,0700`** (`<0,001`) | `+0,30×` (`0,582`) | — | — | ML tuyến tính vượt hẳn rule về AP & AUC |
| **Bậc 3 → 4** | `L4 (LGBM TRX) − L3 (LogReg RFM)` | `+0,0008` (`0,342`) | **`+0,0082`** (`0,016`) | `+0,04×` (`1,000`) | `+0,00081` (`<0,001`) | `+0,000256` (`<0,001`) | LGBM mặc định không thắng AP & kém xác suất |
| *(Thuần TRX)* | `L4 (CB TRX) − L3 (LogReg RFM)` | `-0,0002` (`0,800`) | `-0,0213` (`<0,001`) | `+0,91×` (`0,226`) | `+0,00125` (`<0,001`) | `+0,000090` (`<0,001`) | CB thuần TRX nhỉnh Top 1% nhưng thua AUC |
| | `L4: CatBoost − LightGBM (TRX)` | `-0,0010` (`0,226`) | `-0,0295` (`<0,001`) | `+0,87×` (`0,124`) | `+0,00044` (`0,010`) | **`-0,000166`** (`<0,001`) | CB đỡ overfit Brier; LGBM trội AUC toàn cục |
| **Ablation [CB]** | `L5a (+GEO meta) − L4 (TRX)` | `-0,0001` (`0,960`) | `-0,0065` (`0,136`) | `+0,09×` (`0,866`) | `+0,00045` (`0,004`) | `+0,000035` (`0,014`) | GEO metadata không mang lại lợi ích |
| | `L5b (+GEO content) − L5a (meta)` | `+0,0011` (`0,266`) | `+0,0009` (`0,888`) | `+0,04×` (`0,938`) | `-0,00019` (`0,270`) | `-0,000013` (`0,434`) | `geohash` kéo lại nhẹ, chưa đạt $p < 0,05$ |
| | `L5b (+GEO full) − L4 (TRX)` | `+0,0010` (`0,264`) | `-0,0056` (`0,170`) | `+0,13×` (`0,784`) | `+0,00026` (`0,108`) | `+0,000022` (`0,116`) | Tổng GEO trên CB trung tính |
| | **`L5c (+DIALOG meta) − L5b`** | **`+0,0040`** (`0,002`) | **`+0,0245`** (`<0,001`) | `+1,13×` (`0,092`) | **`-0,00069`** (`<0,001`) | `-0,000029` (`0,064`) | **Cú nhảy lớn từ hoạt động `DIALOG`** |
| | **`L5d (+DIALOG emb) − L5c`** | **`+0,0029`** (`0,014`) | **`+0,0086`** (`0,028`) | `+0,96×` (`0,098`) | **`-0,00065`** (`<0,001`) | `-0,000032` (`0,064`) | **Nội dung ngữ nghĩa PCA tăng có ý nghĩa!** |
| | **`L5d (Full) − L4 (TRX)` [CB]** | **`+0,0079`** (`<0,001`) | **`+0,0275`** (`<0,001`) | **`+2,22×`** (`<0,001`) | **`-0,00108`** (`<0,001`) | **`-0,000039`** (`0,020`) | **Đa nguồn thắng tuyệt đối mọi chỉ số!** |
| **Ablation [LGBM]** | `L5a (+GEO meta) − L4 (TRX)` | `-0,0008` (`0,152`) | `+0,0007` (`0,748`) | `-0,48×` (`0,320`) | `+0,00031` (`0,020`) | `+0,000080` (`<0,001`) | GEO metadata gây nhiễu nhẹ xác suất |
| | **`L5b (+GEO content) − L5a`** | **`+0,0015`** (`0,006`) | `+0,0037` (`0,094`) | **`+1,04×`** (`0,028`) | **`-0,00026`** (`0,038`) | `-0,000010` (`0,746`) | **`geohash` tăng AP & Lift@1% có ý nghĩa** |
| | **`L5c (+DIALOG meta) − L5b`** | **`+0,0062`** (`<0,001`) | **`+0,0235`** (`<0,001`) | **`+1,43×`** (`0,014`) | **`-0,00123`** (`<0,001`) | **`-0,000132`** (`<0,001`) | **Cú nhảy mạnh nhất trên LightGBM** |
| | `L5d (+DIALOG emb) − L5c` | `-0,0013` (`0,244`) | `-0,0011` (`0,516`) | `-0,74×` (`0,168`) | `+0,00022` (`0,054`) | `+0,000047` (`0,064`) | LGBM mặc định bị overfit nhẹ với 27 chiều PCA |
| **Target-Lag & Cal** | **`L5d [CB] (0 target) − L1 (target)`** | **`+0,0116`** (`<0,001`) | **`+0,1541`** (`<0,001`) | `+0,48×` (`0,650`) | — | — | **Thuần hành vi thắng áp đảo Rule Target!** |
| | **`Ref [CB] (+Target) − L5d [CB]`** | **`+0,0042`** (`0,010`) | **`+0,0148`** (`<0,001`) | **`+1,43×`** (`0,014`) | **`-0,00053`** (`0,002`) | `-0,000027` (`0,120`) | Target-lag cộng thêm `+1,43×` Lift@1% |
| | **`Ref [LGBM] (+Target) − L5d [LGBM]`** | **`+0,0056`** (`<0,001`) | **`+0,0181`** (`<0,001`) | **`+1,83×`** (`0,002`) | **`-0,00104`** (`<0,001`) | **`-0,000082`** (`<0,001`) | Target-lag tăng mạnh toàn diện trên LGBM |
| | **`Ref [CB] − L1 (cùng có target)`** | **`+0,0159`** (`<0,001`) | **`+0,1689`** (`<0,001`) | **`+1,91×`** (`0,022`) | — | — | **ML đa nguồn vượt xa Rule trên cùng target** |
| | `Calibration [CB L5d]: Cal − Raw` | `0,0000` (`1,000`) | `0,0000` (`1,000`) | `0,00×` (`1,000`) | `+0,00006` (`<0,001`) | **`-0,000017`** (`<0,001`) | Bảo toàn 100% thứ hạng, giảm Brier có ý nghĩa |

### Giải mã 3 phát hiện chuyên sâu từ bảng Ablation
1. **Tại sao `L5d (+DIALOG embedding)` thắng rõ trên CatBoost (`p = 0,014`) nhưng giảm nhẹ trên LightGBM mặc định?**
   - 27 đặc trưng embedding PCA là các biến liên tục dày đặc nhưng có **~55% `NaN`** (do chỉ ~45% khách hàng có lịch sử dialog). LightGBM với cấu hình mặc định (`num_leaves=31`, `learning_rate=0,1`, `min_child_samples=20`, không `colsample_bytree` và `reg_lambda=0`) dễ tách sâu vào nhiễu trên các nhánh lá nhỏ chứa vector PCA.
   - Ngược lại, CatBoost dùng **cây đối xứng (*oblivious trees*, `depth=6`, `l2_leaf_reg=3`)** áp cùng một điều kiện tách trên toàn tầng cây, hoạt động như một cơ chế regularization tự nhiên giúp chiết xuất đúng tín hiệu ngữ nghĩa từ `PCA(16) + PCA(8) + cosine topic-shift`.
2. **Tại sao LightGBM lại có `ROC-AUC` cao hơn CatBoost ở mọi tầng (`0,775` vs `0,746` ở `L5`, `0,793` vs `0,760` ở `Ref`) trong khi CatBoost thắng ở đỉnh `Lift@0,5%` & `Lift@1%` (`7,87×` vs `6,04×`; `9,30×` vs `7,87×`)?**
   - Cây bất đối xứng (*leaf-wise*) của LightGBM phân tách tốt hơn phần thân giữa của phân phối (decile 1–9 → tối ưu `ROC-AUC` và `Recall@5%`, như `Ref_lgbm` đạt `27,08%` Recall@5%), nhưng bị **over-confidence nặng ở đuôi trên** (xem Chương 4: decile 10 của `L5d_lgbm` dự báo trung bình `4,76%` trong khi thực tế chỉ `2,86%`, làm trộn lẫn false positive điểm rất cao vào Top 1% và đẩy `BSS` xuống âm `-2,47%`).
   - CatBoost giữ xác suất đuôi trên chặt chẽ hơn (`3,72%` vs thực tế `2,60%`), nên ít bị nhiễu lá cực trị làm ô nhiễm Top 0,5%–1% — cực kỳ phù hợp với chiến dịch ngân sách hẹp (`K = 1%`).
3. **Vai trò thực sự của `GEO`:** Chỉ riêng hoạt động mở app (`GEO metadata`) gần như không thêm thông tin ngoài `TRX`. Tín hiệu thực của `GEO` nằm ở **cấu trúc địa bàn (`geohash_4/5/6` & `top_geohash_6_share`)** — thể hiện rõ nhất ở bước nhảy `L5a → L5b` trên LightGBM (`+0,0015` AP, $p = 0,006$; `+1,04×` Lift@1%, $p = 0,028$).

---

## Chương 4. Kiểm chứng Calibration Xác Suất & Phân tích Lát Cắt (Slice Diagnostics)

### 4.1. Chất lượng xác suất (`Calibration`) & Bảng Reliability 10 phân vị trên Locked Test

Base rate giảm liên tục qua thời gian (`Train 04..10: 1,132%` → `Val 09..10: 0,938%` → `Test 11..01: 0,765%`). Khi áp hậu hiệu chỉnh 1 tham số **`intercept_shift` ($\text{logit}(p) + b$)** fit hoàn toàn trên dự báo Validation (`2022-09..10`, [`ladder_manifest.json`](outputs/baseline_moscow_v1/ladder_manifest.json)):

- **`L5d_lgbm_dialog_emb` ($b = -0,2900$, `val_raw_pred_mean = 1,227%` vs `val_obs = 0,938%`):**
  - Cải thiện **đồng loạt cả 4 chỉ số xác suất**: `Calibration Ratio: 1,244 → 0,947`, `ECE: 0,00228 → 0,00204`, `LogLoss: 0,04205 → 0,04185`, `Brier: 0,007796 → 0,007700` (`BSS: -2,47% → -1,22%`), đồng thời giữ nguyên 100% thứ hạng.
- **`L5d_cb_dialog_emb` ($b = -0,1005$, `val_raw_pred_mean = 1,032%` vs `val_obs = 0,938%`):**
  - Cải thiện có ý nghĩa **Brier Score** (`0,007606 → 0,007589`, **`BSS: +0,02% → +0,25%`**, $p < 0,001$), kéo sai số decile cao nhất (Decile 10) từ `1,12%` xuống `0,79%` (`mean_pred: 3,72% → 3,39%` vs thực tế `2,60%`). Tuy nhiên vì CatBoost raw trên Test đã hơi bảo thủ (`CalRatio = 0,969`), dịch xuống thêm kéo `CalRatio` về `0,881` và tăng nhẹ `LogLoss` (`+0,000058`). Do đó với CatBoost nên dùng **rolling intercept-shift trên tháng gần nhất** (ví dụ `T-1`) hoặc **Platt scaling** thay vì cố định trung bình 2 tháng Val.

**Reliability Curve 10 phân vị (Decile) trên Locked Test (`300.672` dòng, mỗi bin `~30.067` dòng):**

| Decile | `L3_logreg_rfm` (Pred \| Obs) | `L5d_lgbm_cal` (Pred \| Obs) | `L5d_cb_raw` (Pred \| Obs) | `L5d_cb_cal` (Pred \| Obs) | `Ref_cb_target_lag` (Pred \| Obs) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 (Thấp nhất) | `0,09%` \| `0,12%` | `0,08%` \| `0,11%` | `0,05%` \| `0,17%` | `0,05%` \| `0,17%` | `0,04%` \| `0,15%` |
| 3 | `0,26%` \| `0,20%` | `0,14%` \| `0,14%` | `0,14%` \| `0,28%` | `0,12%` \| `0,28%` | `0,12%` \| `0,20%` |
| 5 | `0,53%` \| `0,47%` | `0,24%` \| `0,39%` | `0,26%` \| `0,34%` | `0,23%` \| `0,34%` | `0,22%` \| `0,36%` |
| 7 | `0,88%` \| `0,75%` | `0,50%` \| `0,72%` | `0,52%` \| `0,68%` | `0,47%` \| `0,68%` | `0,45%` \| `0,56%` |
| 9 | `1,46%` \| `1,69%` | `1,21%` \| `1,53%` | `1,28%` \| `1,50%` | `1,16%` \| `1,50%` | `1,19%` \| `1,51%` |
| **10 (Cao nhất)** | `2,63%` \| `2,32%` | `3,68%` \| `2,86%` | `3,72%` \| `2,60%` | **`3,39%` \| `2,60%`** | **`3,73%` \| `2,88%`** |

### 4.2. Phân tích Lát cắt (Slice Diagnostics trên Locked Test — [`slice_metrics_test.csv`](outputs/baseline_moscow_v1/metrics/slice_metrics_test.csv))

Để xác minh lợi ích từng nguồn thực sự xuất phát đúng từ phân khúc khách hàng sở hữu dữ liệu đó (không phải nhiễu ngẫu nhiên), bảng dưới đối chiếu 6 mốc CatBoost trên 3 lát cắt cốt lõi:

| Lát cắt (`n`, `positives`, `base rate`) | `L4_cb_trx` AP \| Lift@1% (`Hits`) | `L5b_cb_geo_full` AP \| Lift@1% (`Hits`) | `L5c_cb_dialog_meta` AP \| Lift@1% (`Hits`) | **`L5d_cb_dialog_emb`** AP \| Lift@1% (`Hits`) | *`Ref_cb_target_lag`* AP \| Lift@1% (`Hits`) |
| --- | --- | --- | --- | --- | --- |
| **`has_dialog_all = 1`** (`135.147`, `1.629`, `1,205%`) | `0,0233` \| `4,11×` (`67`) | `0,0254` \| `4,79×` (`78`) | `0,0301` \| `5,52×` (`90`) | **`0,0335`** \| **`6,38×`** (**`104`**) | *`0,0366`* \| *`7,12×`* (`116`) |
| **`has_dialog_all = 0`** (`165.525`, `672`, `0,406%`) | `0,0156` \| `6,40×` (`43`) | `0,0138` \| `5,95×` (`40`) | `0,0143` \| `5,50×` (`37`) | **`0,0158`** \| `6,10×` (`41`) | *`0,0232`* \| **`11,75×`** (`79`) |
| **`has_geo_all = 1`** (`215.883`, `1.829`, `0,847%`) | `0,0230` \| `5,96×` (`109`) | `0,0231` \| `5,80×` (`106`) | `0,0275` \| `6,56×` (`120`) | **`0,0310`** \| **`7,87×`** (**`144`**) | *`0,0350`* \| *`9,13×`* (`167`) |
| **`has_geo_all = 0`** (`84.789`, `472`, `0,557%`) | `0,0136` \| `4,24×` (`20`) | `0,0169` \| `6,78×` (`32`) | `0,0205` \| **`7,41×`** (`35`) | `0,0209` \| `6,99×` (`33`) | *`0,0262`* \| *`8,69×`* (`41`) |
| **`cold_start = 0`** (`298.400`, `2.253`, `0,755%`) | `0,0204` \| `5,68×` (`128`) | `0,0215` \| `5,90×` (`133`) | `0,0255` \| `7,01×` (`158`) | **`0,0285`** \| **`7,99×`** (**`180`**) | *`0,0328`* \| *`9,41×`* (`212`) |
| **`cold_start = 1`** (`2.272`, `48`, **`2,113%`**) | `0,0196` \| `0,00×` (`0`) | `0,0202` \| `2,06×` (`1`) | `0,0192` \| `0,00×` (`0`) | `0,0192` \| `0,00×` (`0`) | ***`0,0510`*** \| ***`6,17×`*** (`3`) |

**Ba phát hiện cực rõ từ phân tích lát cắt:**
1. **Xác nhận cơ chế trực tiếp của `DIALOG embedding`:** Riêng trên nhóm khách có hội thoại (`has_dialog_all = 1`, chiếm tới `70,8%` tổng số ca mua hàng dù chỉ chiếm `44,9%` dân số!), `PR-AUC` tăng từng bậc mượt mà từ **`0,0233` (`L4`) → `0,0301` (`L5c` meta) → `0,0335` (`L5d` +embedding)**, còn số khách mua trúng Top 1% trong lát cắt tăng vọt **`67 → 90 → 104`**! Đồng thời trên nhóm không có dialog (`has_dialog_all = 0`), việc giữ nguyên `NaN` giúp `L5d` không bị suy giảm mà còn tăng `AUC` từ `0,7439 → 0,7587`.
2. **Đóng góp bổ trợ của `Target-Lag` mạnh nhất ở nhóm "im lặng hành vi":** Trên nhóm **`has_dialog_all = 0`**, `Ref_cb_target_lag` đẩy `Lift@1%` từ `6,10×` lên **`11,75×`** (`41 → 79` trúng). Trên nhóm **`cold_start = 1`** (`2.272` lượt, tỷ lệ mua rất cao `2,11%` nhưng không có bất kỳ log TRX/GEO/DIALOG nào trước `cutoff_msk`), mọi mô hình thuần hành vi đều mù hoàn toàn (`AUC ≈ 0,46`), chỉ có `Ref_cb_target_lag` tách được khách mua (`AP = 0,0510`, `Lift@0,5% = 11,83×`).
3. **Điểm mù kinh doanh của `cold_start = 1`:** Nhóm không có log hành vi chỉ chiếm `0,76%` dân số nhưng có tỷ lệ mua cao gấp **`2,8×`** trung bình (`2,11%` vs `0,76%`) và bị mô hình thuần hành vi đánh giá thấp xác suất (`CalRatio = 0,516`). Trong vận hành thực tế, cần bổ sung tín hiệu tĩnh cấp hồ sơ doanh nghiệp (CRM / ngành nghề / ngày mở tài khoản / kênh tiếp cận ban đầu) nếu có, hoặc áp dụng chính sách chăm sóc riêng cho tệp `cold_start`.

---

## Chương 5. Hiệu năng Tính toán, Khuyến nghị Chốt Baseline & Lộ trình Sau Baseline

### 5.1. Chi phí tính toán & Tối ưu hệ thống trên máy 8 GB RAM (`i3-8145U`, 2 nhân/4 luồng)
- **Xây dựng dữ liệu đặc trưng ([`build_moscow_features.py`](src/scripts/build_moscow_features.py)):** Sau khi tối ưu chạy tách rời từng phân vùng `fold=0..4` (`memory_limit='2.5GB'`, `preserve_insertion_order=false`, tái dùng `GROUP BY` của `event_type`/`geohash_6`), tốc độ mỗi tháng giảm từ **31–33 phút → 5,5–6,8 phút** (nhanh hơn **~5,5–6×**), loại bỏ hoàn toàn pagefile thrashing.
- **Huấn luyện & Đánh giá toàn bộ bậc thang ([`run_moscow_ladder.py`](src/scripts/run_moscow_ladder.py)):** Nhờ tách Stage 0 tiền tính PCA theo fold rồi giải phóng `DialogEmbeddingIndex` (~`1,8 GB`) khỏi RAM trước khi huấn luyện, toàn bộ pipeline chạy gọn trong RAM vật lý (`WS ≈ 1,4 GB`, còn trống `> 4 GB` RAM):
  - Tổng thời gian huấn luyện 16 mô hình × 5 fold trên Validation: **`2.599,4s` (~43,3 phút)** (riêng nhánh LightGBM 6 mô hình chỉ tốn `256s` ~4,3 phút; nhánh CatBoost 6 mô hình tốn `2.336s` ~39 phút).
  - Tổng thời gian huấn luyện 16 mô hình × 5 fold trên Locked Test: **`3.576,5s` (~59,6 phút)** (LightGBM 6 mô hình `386s` ~6,4 phút; CatBoost 6 mô hình `3.181s` ~53 phút).
  - Stage 0 (PCA 10 folds) + Stage 3 (Calibration + 34 Single Bootstrap + 21 Paired Bootstrap × 1.000 lần + Slice Diagnostics): **`~26 phút`**. Tổng cộng toàn bộ `run_moscow_ladder.py`: **`7.736,4s` (~2,15 giờ)**.

### 5.2. Chốt mốc Baseline chính thức
1. **Mốc Baseline chính (thuần hành vi — KHÔNG dùng target-lag):** **`L5d_cb_dialog_emb`** (kèm hiệu chỉnh hậu kỳ cho các bài toán cần xác suất). Đây là mốc chuẩn cho kịch bản dự báo nhu cầu thuần từ dấu vết hành vi đa kênh (`TRX + GEO + DIALOG meta + DIALOG 768d PCA`), đạt `PR-AUC = 0,0284`, `Lift@1% = 7,87×`, `Lift@0,5% = 9,90×` trên Locked Test 3 tháng `0%` trùng khách.
2. **Mốc Tham chiếu mở rộng (khi nghiệp vụ cho phép dùng lịch sử mua quá khứ $m < T$):** **`Ref_cb_target_lag`** (tối ưu Top-K ngân sách hẹp: `Lift@1% = 9,30×`, `Lift@0,5% = 11,29×`, `BSS = +0,38%`) và **`Ref_lgbm_target_lag`** (tối ưu phân tách toàn tệp & ngân sách rộng 5%: `ROC-AUC = 0,7925`, `PR-AUC = 0,0328`, `Recall@5% = 27,08%`).

### 5.3. Lộ trình 5 hướng nâng cấp ưu tiên ngay sau Baseline
1. **Regularization cho LightGBM (ưu tiên #1):** LightGBM mặc định đã đạt `ROC-AUC = 0,7755` (thuần hành vi) và `0,7925` (`Ref`) — vượt CatBoost `~0,03` AUC — nhưng bị tụt `Lift@1%` và âm `BSS` vì cây mặc định quá sâu/overfit lá hiếm (`lr=0,1`, `min_child_samples=20`). Chỉ cần thêm regularization (`learning_rate=0,03`, `min_child_samples=100–300`, `colsample_bytree=0,5–0,7`, `reg_lambda=10–50`, `max_depth=5–6`) có khả năng kết hợp được cả `ROC-AUC ~0,79+` lẫn `Lift@1%` cao và khai thác tốt 27 chiều `DIALOG` embedding.
2. **Tinh chỉnh biểu diễn `DIALOG` embedding:** Thay vì chỉ mean-pool đơn giản toàn lịch sử, thử nghiệm **exponentially time-decayed pooling** (trọng số cao hơn cho hội thoại gần cutoff), **last-dialog embedding**, khoảng cách cosine tới *prototype* người mua (fit `out-of-fold` trên tập train) và **SVD/Ridge stacking** để nén 768 chiều có giám sát mà không bị overfit.
3. **Feature Engineering có chọn lọc từ EDA (động lượng & tương tác đa kênh):** Bổ sung các biến tỷ số/tương tác có cơ sở từ EDA như `trx_momentum_30d_vs_prior`, `trx_30d_vs_all_etype_ratio`, `zero_trx_x_dialog_30d`, `active_sources_30d_count`, `trx_30d_etype_x_geo_g4_30d`, và kiểm định loại bỏ `GEO metadata` nhiễu bằng Ablation trên Validation.
4. **Ensemble đa họ (`CatBoost + Regularized LightGBM + Logistic/Neural`):** Vì CatBoost trội ở đuôi trên (`Lift@0,5%–1%`, `Brier`) trong khi LightGBM trội ở phân tách toàn cục (`ROC-AUC`, `Recall@5%`), blending theo xác suất đã hiệu chỉnh có tiềm năng vượt cả hai mô hình đơn lẻ.
5. **Mở rộng đa mục tiêu (`target_1..4`) & chân trời `H=3` trên dữ liệu Moscow `UTC+3`:** Huấn luyện 4 mô hình theo từng sản phẩm (để cá nhân hóa offer thay vì chỉ `y_any`) và đối chứng chân trời 3 tháng (`H=3` với embargo cứng 2 tháng) trên cùng nền dữ liệu chuẩn Moscow không rò rỉ.

---

## Phụ lục A. Danh mục Đầy đủ 135 Cột trong `panel_moscow.parquet` & 27 Đặc trưng `DIALOG` Embedding

| Nhóm | SL | Danh sách cột chi tiết |
| --- | ---: | --- |
| **Meta & Labels** *(cấm làm feature)* | `10` | `client_id`, `fold`, `month`, `cutoff_msk`, `y_any`, `target_1`, `target_2`, `target_3`, `target_4`, `prior_months_observed` |
| **`TRX` (3 cửa sổ `30d`, `90d`, `all`)** | `60` | Với mỗi `w ∈ {30d, 90d, all}`: `trx_{w}_has_event`, `trx_{w}_event_count`, `trx_{w}_log1p_event_count`, `trx_{w}_active_days`, `trx_{w}_days_since_last_event`, `trx_{w}_days_since_first_event`, `trx_{w}_main_count`, `trx_{w}_foreign_count`, `trx_{w}_foreign_fraction`, `trx_{w}_main_neg_count`, `trx_{w}_main_amount_abs_sum`, `trx_{w}_log1p_main_amount_abs_sum`, `trx_{w}_main_amount_signed_log_mean`, `trx_{w}_main_amount_median`, `trx_{w}_main_amount_max`, `trx_{w}_event_type_nunique`, `trx_{w}_event_subtype_nunique`, `trx_{w}_event_type_top_share`, `trx_{w}_dst_type11_nunique`, `trx_{w}_dst_type12_nunique` |
| **`TRX` liên cửa sổ** | `3` | `trx_rate_per_30d_tenure`, `trx_share_30d_of_90d`, `trx_first_activated_30d` |
| **`GEO metadata` (`30d`, `90d`, `all`)** | `18` | Với mỗi `w ∈ {30d, 90d, all}`: `geo_{w}_has_event`, `geo_{w}_event_count`, `geo_{w}_log1p_event_count`, `geo_{w}_active_days`, `geo_{w}_days_since_last_event`, `geo_{w}_days_since_first_event` |
| **`GEO content` (`30d`, `90d`, `all`)** | `12` | Với mỗi `w ∈ {30d, 90d, all}`: `geo_{w}_geohash_4_nunique`, `geo_{w}_geohash_5_nunique`, `geo_{w}_geohash_6_nunique`, `geo_{w}_top_geohash_6_share` |
| **`DIALOG metadata` (`30d`, `90d`, `all`) + Cold-start** | `19` | Với mỗi `w ∈ {30d, 90d, all}`: `dialog_{w}_has_event`, `dialog_{w}_event_count`, `dialog_{w}_log1p_event_count`, `dialog_{w}_active_days`, `dialog_{w}_days_since_last_event`, `dialog_{w}_days_since_first_event`, cộng `cold_start` |
| **`Target-Lag` (CHỈ dùng ở `L1` & `Ref_*`)** | `13` | `prior_1m_bought_any`, `prior_3m_buy_months`, `prior_3m_total_products`, `prior_6m_buy_months`, `prior_6m_total_products`, `prior_all_bought_any`, `prior_all_buy_months`, `prior_months_since_last_buy`, `prior_all_bought_p1`, `prior_all_bought_p2`, `prior_all_bought_p3`, `prior_all_bought_p4`, `prior_all_distinct_products` |
| **`DIALOG embedding` (tạo động theo fold, train-only)** | `+27` | `16` thành phần chính `PCA(all)` + `8` thành phần chính `PCA(90d)` + `norm_l2(all)` + `norm_l2(90d)` + `cosine_sim(90d, all)` (khách không có dialog giữ `NaN`) |

### Tập biến cho các bậc đơn giản (`L1`, `L2`, `L3`)
- **`L1_rule_target_lag` (`3` biến):** `score = 3*prior_1m_bought_any + 2*prior_3m_buy_months + 1*prior_6m_buy_months`.
- **`L2_rule_trx_rfm` (`6` biến):** Trung bình phân vị xếp hạng của `trx_30d/90d_event_count`, `trx_30d/90d_log1p_main_amount_abs_sum`, và nghịch đảo `trx_30d/90d_days_since_last_event`.
- **`L3_logreg_rfm` (`8` biến):** `trx_30d/90d_days_since_last_event`, `trx_30d/90d_log1p_event_count`, `trx_30d/90d_log1p_main_amount_abs_sum`, `trx_all_days_since_first_event`, `trx_all_has_event`.

---

## Phụ lục B. Cấu hình Thuật toán & Lệnh Tái lập Toàn bộ Pipeline

| Bậc / Họ mô hình | Tiền xử lý & Xử lý `NaN` | Siêu tham số cố định (`seed = 42`) |
| --- | --- | --- |
| `L0` `PrevalenceBaseline` | Không dùng biến | Trả về hằng số $p_{\text{train}}$ + phá hòa tất định `SHA-256(42, client_id)`. |
| `L1` / `L2` Rule Baselines | `NaN` điền `0` cho count/amount, bỏ khỏi xếp hạng recency | Công thức quy tắc phi-ML cố định (`is_probabilistic = False`). |
| `L3` `SmallRfmLogRegBaseline` | Fit riêng trên từng fold train: `SimpleImputer(median, add_indicator=True)` → `StandardScaler()` | `LogisticRegression(C=1.0, solver="lbfgs", max_iter=300, random_state=42)`. |
| `L4..L5d / Ref` `LgbmDefaultModel` | `float32`, giữ nguyên `NaN` có ngữ nghĩa | `LGBMClassifier(random_state=42, n_jobs=4, verbose=-1)` (100% tham số mặc định). |
| `L4..L5d / Ref` `CatBoostDefaultModel` | `float32`, giữ nguyên `NaN` có ngữ nghĩa | `CatBoostClassifier(iterations=500, random_seed=42, thread_count=4, border_count=128, used_ram_limit="1.5gb", loss_function="Logloss")` (depth=6, l2=3, auto lr mặc định). |
| `ProbabilityCalibrator` | Fit riêng trên dự báo Validation (`2022-09..10`), áp lên Locked Test | `intercept_shift` ($\text{logit}(p) + b$), bảo toàn 100% thứ hạng dự báo. |

```powershell
# 1. Tạo đặc trưng & dialog embeddings chuẩn múi giờ Moscow (UTC+3), phân vùng theo fold=0..4
.\.venv\Scripts\python.exe -B -X utf8 src\scripts\build_moscow_features.py

# 2. Chạy toàn bộ bậc thang Baseline Moscow (Val 09-10, Test 11-01, 5-fold 0% trùng KH, 1.000 Bootstrap)
.\.venv\Scripts\python.exe -B -X utf8 src\scripts\run_moscow_ladder.py
```
