# Đặc Tả Đóng Băng Trước Thực Nghiệm (Pre-Registration Spec — Bước 0)

> **Trạng thái:** `FROZEN` — Chốt và đóng băng hoàn toàn **trước khi chạy bất kỳ mô hình mới nào**. Tuyệt đối không chỉnh sửa nội dung, ngưỡng, danh sách biến hay cặp kiểm định sau khi quan sát kết quả.
> **File cấu hình máy đọc (kiểm tra tự động bằng mã băm SHA-256):** [`configs/fe_v1_evaluation_spec.json`](../configs/fe_v1_evaluation_spec.json)

---

## 1. Mục tiêu nghiên cứu & Minh bạch phương pháp luận

Thực nghiệm này tách bạch định lượng ba nguồn tạo hiệu năng trên bài toán dự báo xu hướng mua sản phẩm doanh nghiệp (**MBD-mini**, chuẩn múi giờ `Europe/Moscow UTC+3`):
1. **Đổi họ mô hình** trên cùng tập biến thô (`Logistic Regression L2` $\rightarrow$ `CatBoost mặc định`).
2. **Mở rộng nguồn dữ liệu ở mức thô** (`TRX` đơn nguồn $\rightarrow$ `TRX + GEO + DIALOG` chỉ với biến đếm/recency cơ bản, không biến phái sinh phức tạp).
3. **Feature Engineering vòng 1 (`FE-v1`)** gồm: biến phái sinh bảng đa cửa sổ (`30d/90d/all`, tỷ lệ, liên cửa sổ, `cold_start`), nội dung không gian (`geohash_4/5/6`), và nội dung ngữ nghĩa hội thoại (`DIALOG 768d` mean-pool $\rightarrow$ `PCA(16) + PCA(8) + L2 norms + cosine topic-shift`).

### Tuyên bố minh bạch về `FE-v1` và tập Locked Test (Bước 2 & Bước 5)
- `L5d_cb_dialog_emb` của vòng trước ([`BASELINE.md`](../BASELINE.md)) được đổi định danh thành **`FE-v1`**; ở `seed = 42` gốc, chỉ đổi `exp_id` và cập nhật manifest, **không huấn luyện lại**.
- `L4_cb` (`L4_cb_trx`) và `L5c_cb` (`L5c_cb_dialog_meta`) được giữ lại làm **các bậc trung gian**. Mọi cải tiến đặc trưng về sau sẽ mang mã **`FE-v2`** và đối sánh trực tiếp với `FE-v1`.
- **Ghi nhận bắt buộc:** `FE-v1` được chọn làm mốc chuẩn **sau khi kết quả trên tập Locked Test (`2022-11`, `2022-12`, `2023-01`) đã được xem ở vòng trước**. Do đó, mọi kết luận liên quan đến `FE-v1` trên tập Test này đều được báo cáo kèm lưu ý *post-selection optimism* và bắt buộc đối chứng song song với tập **Validation (`2022-09`, `2022-10`)**.

---

## 2. Các quyết định thiết kế đã chốt cố định (Bước 1)

Nguồn dữ liệu duy nhất: [`outputs/baseline_moscow_v1/panel_moscow.parquet`](../outputs/baseline_moscow_v1/panel_moscow.parquet) (`1.002.240 × 135`) và [`outputs/baseline_moscow_v1/dialog_emb_moscow.npz`](../outputs/baseline_moscow_v1/dialog_emb_moscow.npz) (chỉ dùng cho `FE-v1`). **Chỉ chọn cột có sẵn, không xây dựng lại đặc trưng.**

1. **Chọn cửa sổ `90d` đồng nhất cho phần bổ sung `GEO` và `DIALOG` của `B-raw-TGD`:**
   - Dùng bộ 3 biến cơ bản `(has_event, event_count, days_since_last_event)` ở cửa sổ **`90d`** cho cả `GEO` và `DIALOG`.
   - *Lý do:* Nhờ giai đoạn burn-in `2022-02..03` (sau thời điểm `GEO` bắt đầu `11/12/2021`), cửa sổ `90d` có độ dài cố định 90 ngày đồng nhất tuyệt đối trên toàn bộ 10 tháng `2022-04..2023-01`. Ngược lại, cửa sổ `all` có độ dài tích lũy tăng dần từ ~4 tháng lên ~13 tháng đối với `GEO` và ~14 lên ~23 tháng đối với `DIALOG`, gây trôi phân phối đếm (`event_count` phình to theo thời gian trong khi base rate giảm dần) mà baseline thô lại bị cấm dùng biến tỷ lệ chuẩn hóa theo tenure.
2. **Quyết định về `cold_start`: KHÔNG đưa `cold_start` vào `B-raw-TGD`:**
   - *Lý do:* `cold_start` là biến cờ **phái sinh liên nguồn trên cửa sổ `all`** (`trx_all_has_event == 0 & geo_all_has_event == 0 & dialog_all_has_event == 0`). Việc loại `cold_start` bảo đảm `B-raw-TGD` thuần túy chỉ ghép các biến thô độc lập của từng nguồn (`8 + 3 + 3 = 14` biến), giữ ranh giới sạch 100% giữa ghép nguồn thô và Feature Engineering liên nguồn.
3. **Quyết định về `LogReg` trên `B-raw-TGD`: CÓ chạy thêm `B-raw-TGD_logreg` làm mô hình đối chứng phụ:**
   - *Lý do:* Chi phí tính toán không đáng kể (~40 giây cho 3 seeds), nhưng giúp khép kín bảng đối chứng $2 \times 2$ (`LogReg` vs `CatBoost` $\times$ `TRX thô 8 biến` vs `3 nguồn thô 14 biến`). Để không làm giảm lực kiểm định (power) của các giả thuyết chính, `B-raw-TGD_logreg` được đặt ở bảng đối chứng phụ, ngoài họ Holm chính 7 cặp.

---

## 3. Danh mục mô hình, vai trò và danh sách biến cố định

Mọi mô hình thuần hành vi (`L2` đến `FE-v1`) bị **cấm tuyệt đối** dùng `FORBIDDEN_COLS` (`client_id`, `fold`, `month`, `cutoff_msk`, `y_any`, `target_1..4`, `prior_months_observed`) và toàn bộ 13 biến `Target-Lag` (`prior_*`). Các mô hình `CatBoost` giữ nguyên `NaN` tự nhiên (`float32`).

| Exp ID | Vai trò trong thang bậc | Họ mô hình | SL biến | Danh sách biến cố định |
| --- | --- | --- | ---: | --- |
| `L0_base_rate` | Mốc sàn ngẫu nhiên | `PrevalenceBaseline` | `0` | Không dùng biến (`p_train` + băm phá hòa `SHA-256`) |
| `L1_rule_target_lag` | Luật nghiệp vụ **có dùng nhãn quá khứ** | Rule phi-ML | `3` | `prior_1m_bought_any`, `prior_3m_buy_months`, `prior_6m_buy_months` |
| `L2_rule_trx_rfm` | Luật nghiệp vụ thuần `TRX` (không nhãn) | Rule phi-ML | `6` | `trx_30d_event_count`, `trx_90d_event_count`, `trx_30d_log1p_main_amount_abs_sum`, `trx_90d_log1p_main_amount_abs_sum`, `trx_30d_days_since_last_event`, `trx_90d_days_since_last_event` |
| `L3_logreg_rfm` | Baseline ML tuyến tính đơn nguồn `TRX` | `LogReg L2` (median impute + missing ind + scaler) | `8` | `trx_30d_days_since_last_event`, `trx_90d_days_since_last_event`, `trx_30d_log1p_event_count`, `trx_90d_log1p_event_count`, `trx_30d_log1p_main_amount_abs_sum`, `trx_90d_log1p_main_amount_abs_sum`, `trx_all_days_since_first_event`, `trx_all_has_event` |
| **`B-raw-T`** | **Baseline GBDT thô đơn nguồn `TRX`** | `CatBoost` mặc định | **`8`** | **Đúng 8 biến của `L3_logreg_rfm`** (giữ nguyên `NaN`) |
| **`B-raw-TGD`** | **Baseline GBDT thô 3 nguồn (`TRX+GEO+DIALOG`)** | `CatBoost` mặc định | **`14`** | **8 biến của `B-raw-T`** + `geo_90d_has_event`, `geo_90d_event_count`, `geo_90d_days_since_last_event` + `dialog_90d_has_event`, `dialog_90d_event_count`, `dialog_90d_days_since_last_event` (không tỷ lệ, không liên cửa sổ, không geohash, không PCA/norm/cosine, không `cold_start`, không `prior_*`, giữ nguyên `NaN`) |
| `B-raw-TGD_logreg` | *(Phụ)* Đối chứng tuyến tính 3 nguồn thô | `LogReg L2` | `14` | Đúng 14 biến của `B-raw-TGD` |
| `L4_cb` (`L4_cb_trx`) | Bậc trung gian: FE đầy đủ trên riêng `TRX` | `CatBoost` mặc định | `63` | 60 biến `trx_{30d,90d,all}_*` + 3 biến liên cửa sổ (`trx_rate_per_30d_tenure`, `trx_share_30d_of_90d`, `trx_first_activated_30d`) |
| `M_deriv_cb` | *(Phụ — phân rã bước 7)* Biến phái sinh bảng 3 nguồn | `CatBoost` mặc định | `100` | `63` biến `TRX` + `18` biến `geo_{30d,90d,all}` hoạt động + `18` biến `dialog_{30d,90d,all}` hoạt động + `cold_start` *(chưa có `geohash` và `embedding`)* |
| `L5c_cb` (`L5c_cb_dialog_meta`) | Bậc trung gian: Đa nguồn bảng + `geohash` | `CatBoost` mặc định | `112` | `100` biến của `M_deriv_cb` + `12` biến `geo_{30d,90d,all}_*geohash*` *(chưa có `embedding`)* |
| **`FE-v1`** (đổi từ `L5d_cb_dialog_emb`) | **Mốc chuẩn Feature Engineering vòng 1** | `CatBoost` mặc định | **`139`** | `112` biến của `L5c_cb` + `27` biến `DIALOG` embedding fit `train-only` theo từng fold (`PCA(16)` trên `all` + `PCA(8)` trên `90d` + `norm_l2(all)` + `norm_l2(90d)` + `cosine_sim(90d, all)`) |

---

## 4. Giao thức thực nghiệm, Ngân sách K, Metric chính & Ngưỡng ý nghĩa thực tế

### 4.1. Phân hoạch, Siêu tham số & Đa Seed
- **Giữ nguyên 100% split và nhãn của vòng trước:**
  - Stage 1 (**Validation**): Train `2022-04..2022-08` (`fold != k`) $\rightarrow$ Eval `2022-09..2022-10` (`fold == k`, `200.448` dòng, `1.881` ca dương, base rate `0,938%`).
  - Stage 2 (**Locked Test**): Train `2022-04..2022-10` (`fold != k`) $\rightarrow$ Eval `2022-11..2023-01` (`fold == k`, `300.672` dòng, `2.301` ca dương, base rate `0,765%`).
  - Bất biến: **`0%` trùng `client_id`** giữa tập huấn luyện và tập đánh giá trên cả 5 fold.
- **Cấu hình `CatBoost` mặc định giữ nguyên tuyệt đối:** `iterations=500`, `border_count=128`, `loss_function="Logloss"`, `thread_count=4`, `used_ram_limit="1.5gb"` (`depth=6`, `l2_leaf_reg=3`, auto learning rate).
- **Đa seed (`S = 3` seeds cố định: `42, 142, 242`):**
  - Chạy đủ 3 seeds (`42, 142, 242`) cho mọi mô hình ML trong bảng (`L3_logreg_rfm`, `B-raw-T`, `B-raw-TGD`, `B-raw-TGD_logreg`, `L4_cb`, `M_deriv_cb`, `L5c_cb`, `FE-v1`). Tại `seed = 42`, các mô hình đã có từ vòng trước (`L0`, `L1`, `L2`, `L3`, `L4_cb`, `L5c_cb`, `FE-v1`) tái sử dụng trực tiếp file dự báo gốc, **không chạy lại**.
  - Các luật phi-ML tất định (`L1`, `L2`) và `L0` có độ lệch chuẩn theo seed huấn luyện bằng `0`.
  - Báo cáo **Trung bình ± Độ lệch chuẩn (`Mean ± SD`)** của các mô hình đơn lẻ qua 3 seeds (không trộn trung bình xác suất dự báo để tránh hiệu ứng ensemble giả tạo).
- **Quy tắc gộp fold:**
  - **Giao thức chính:** **Pooled Out-of-Fold (`Pooled OOF`)** — ghép điểm xác suất out-of-fold của 5 mô hình fold tách rời để xếp hạng chung trên 100% danh sách khách hàng (`100.224` KH/tháng), khớp 1-1 với vòng trước và vận hành thực tế.
  - **Đối chứng phụ:** Báo cáo kèm **Trung bình ± Độ lệch chuẩn qua 5 fold riêng biệt (`Per-Fold Mean ± SD`)** để xác minh không có độ lệch thang xác suất giữa các fold làm méo thứ hạng gộp.

### 4.2. Ngân sách chiến dịch $K$, Metric chính & Ngưỡng hiệu ứng thực tế (SESOI)
- **Ngân sách chiến dịch giả định chính:** **`K = Top 1%`** (`3.007` lượt tiếp cận trên Locked Test 3 tháng $\approx 1.002$ KH/tháng; `2.005` lượt trên Validation 2 tháng).
- **Metric chính (Primary Metric):** **`PR-AUC` (`Average Precision`)** trên Pooled OOF.
- **Metric vận hành đồng hành tại ngân sách $K$:** **`Lift@1%`** (kèm báo cáo đầy đủ `Lift` & `Recall` tại `Top 1%, 5%, 10%`, `ROC-AUC`, và đường cong `Lift / Gain` liên tục cho `K = 1%, 2%, ..., 20%`).
- **Ngưỡng hiệu ứng có ý nghĩa thực tế (Practical Significance Thresholds — tự chốt và đóng băng):**
  1. **Trên Metric chính (`PR-AUC`):**
     $$\Delta \text{PR-AUC}_{\text{tương đối}} \ge +5,0\% \quad \text{VÀ} \quad \Delta \text{PR-AUC}_{\text{tuyệt đối}} \ge +0,0010$$
  2. **Tại ngân sách vận hành chính (`K = 1%`):**
     $$\Delta \text{Lift@1\%} \ge +0,30\times \quad (\text{tương đương bắt thêm tối thiểu } \approx +7 \text{ khách mua trên } 3.007 \text{ cuộc gọi Test})$$
  3. **Ngưỡng ý nghĩa thống kê:** Khoảng tin cậy 95% và $p$-value hai phía từ **Paired Client-Clustered Bootstrap (`1.000` lần lặp theo cụm `client_id`)**, lấy trung bình chênh lệch qua `S = 3` seeds trong từng lượt bootstrap, hiệu chỉnh **Holm-Bonferroni** đạt **$p_{\text{Holm}} < 0,05$**.

---

## 5. Danh sách cặp kiểm định & Cấu trúc hiệu chỉnh Holm (Bước 3)

### A. Họ kiểm định chính (Primary Holm Family — $M = 7$ cặp)
Toàn bộ 7 cặp dưới đây được hiệu chỉnh bội **Holm-Bonferroni** đồng thời ở mức $\alpha = 0,05$:

| Mã | Cặp so sánh (`B − A`) | Câu hỏi nghiên cứu tương ứng |
| :---: | --- | --- |
| **C1** | `B-raw-T − L3_logreg_rfm` | Chỉ đổi họ mô hình (`LogReg` sang `CatBoost` mặc định), cùng đúng 8 biến `TRX` |
| **C2** | `L4_cb − B-raw-T` | Đóng góp của Feature Engineering trên riêng `TRX` (`8` sang `63` biến, cùng `CatBoost`) |
| **C3** | `B-raw-TGD − B-raw-T` | Đóng góp của việc thêm `GEO` và `DIALOG` ở mức thô (`8` sang `14` biến, cùng `CatBoost`) |
| **C4** | `FE-v1 − B-raw-TGD` | Đóng góp tổng của Feature Engineering vòng 1: biến phái sinh, `geohash` và `embedding` (`14` sang `139` biến) |
| **C5** | `FE-v1 − L4_cb` | Đóng góp tổng của hai nguồn `GEO` và `DIALOG` (`63` sang `139` biến, gồm cả nguồn + FE) |
| **C6a** | `FE-v1 − L1_rule_target_lag` | `FE-v1` (thuần hành vi) có vượt luật `Target-Lag` (`L1`, có dùng nhãn quá khứ) tại ngân sách $K$ không? |
| **C6b** | `FE-v1 − L2_rule_trx_rfm` | `FE-v1` có vượt luật `TRX RFM` (`L2`, không dùng nhãn) tại ngân sách $K$ không? |
| *(C7)* | *`FE-v2 − FE-v1` (dự phòng về sau)* | *Dành cho vòng cải tiến tiếp theo; chưa thực thi ở vòng này.* |

### B. Họ phân rã thứ cấp có điều kiện (Conditional Gatekeeping Family — $M_2 = 3$ cặp)
Chỉ kích hoạt diễn giải chính thức khi **C4 (`FE-v1 − B-raw-TGD`)** vượt ngưỡng thực tế và đạt $p_{\text{Holm}} < 0,05$, sử dụng hiệu chỉnh Holm trên 3 thành phần cộng tính khép kín ($\text{D1} + \text{D2} + \text{D3} \equiv \text{C4}$):
- **D1 (`M_deriv_cb − B-raw-TGD`, `14 → 100` biến):** Đóng góp của các biến phái sinh bảng đa cửa sổ (`30d/90d/all`, tỷ lệ, liên cửa sổ, `cold_start`).
- **D2 (`L5c_cb − M_deriv_cb`, `100 → 112` biến):** Đóng góp của nội dung không gian `geohash_4/5/6`.
- **D3 (`FE-v1 − L5c_cb`, `112 → 139` biến):** Đóng góp của nội dung ngữ nghĩa `DIALOG embedding` (`PCA + L2 norms + cosine`).

---

## 6. Quy tắc đọc kết quả đã đóng băng (Bước 7)

1. **Quy tắc R1 — Feature Engineering vòng 1 có giá trị thực chất:**
   Nếu `FE-v1` vượt `B-raw-TGD` (cặp **C4**) đạt ngưỡng thực tế ($\Delta \text{PR-AUC}_{\text{rel}} \ge +5\%$ và $\Delta \text{PR-AUC}_{\text{abs}} \ge +0,0010$) **và** đạt $p_{\text{Holm}} < 0,05$: kết luận Feature Engineering vòng 1 đóng góp giá trị thực sự vượt lên trên việc ghép nguồn thô; lập tức phân rã theo **D1, D2, D3** để chỉ rõ thành phần nào (`biến phái sinh`, `geohash`, hay `embedding`) gánh phần lớn cải thiện.
2. **Quy tắc R2 — Lợi ích chủ yếu nằm ở mở rộng nguồn dữ liệu:**
   Nếu **C4** có mức chênh dưới ngưỡng thực tế **hoặc** $p_{\text{Holm}} \ge 0,05$: kết luận phần lớn lợi ích đa nguồn đến từ bản thân sự hiện diện của `GEO + DIALOG` ở mức đếm/recency thô (**C3**), còn tổ hợp 125 biến FE hiện tại chưa tương xứng độ phức tạp; ưu tiên cắt tỉa mạnh tập biến trước khi thử nghiệm `FE-v2`.
3. **Quy tắc R3 — Kiểm toán lại snapshot và cửa sổ khi thua luật cơ sở tại ngân sách $K$:**
   Nếu `B-raw-TGD` hoặc `L4_cb` vẫn thấp hơn luật thuần hành vi `L2_rule_trx_rfm` tại ngân sách chính `K = 1%` (`Lift@1%`), dừng tinh chỉnh mô hình và quay lại kiểm tra ranh giới snapshot / cửa sổ tổng hợp feature. Nếu vượt `L2` nhưng thấp hơn `L1_rule_target_lag` (luật dùng lịch sử nhãn mua) tại `K = 1%`, kiểm tra tính nhất quán cửa sổ theo từng tháng và tách bạch rõ khoảng cách do không được phép dùng biến `Target-Lag` so với tín hiệu hành vi.
