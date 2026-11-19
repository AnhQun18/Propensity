# Hướng dẫn Đọc hiểu & Cách tính Toàn bộ Biểu đồ trong Báo cáo Phân tích Chuyên sâu (MBD-mini)

Tài liệu này giải thích chi tiết **cách đọc hiểu ý nghĩa kinh doanh/thống kê** và **công thức tính toán chính xác (tử số, mẫu số, điều kiện lọc, nguồn dữ liệu)** của toàn bộ **14 biểu đồ & bảng tổng hợp** trên **8 trang slide** trong báo cáo:
- **File báo cáo PDF:** [`outputs/eda_insights_20261119/MBD_MINI_DEEP_INSIGHTS_DECK.pdf`](outputs/eda_insights_20261119/MBD_MINI_DEEP_INSIGHTS_DECK.pdf)
- **Thư mục ảnh từng trang slide (`slide_01.png` – `slide_08.png`):** [`outputs/eda_insights_20261119/slides_clean/`](outputs/eda_insights_20261119/slides_clean/)

---

## Phần 0. Quy ước Dữ liệu Nền tảng & Công thức Chung

Trước khi đi vào từng biểu đồ, toàn bộ báo cáo tuân thủ thống nhất 4 quy ước toán học sau để đảm bảo **không rò rỉ dữ liệu tương lai (zero data leakage)** và **công bằng khi so sánh các chân trời dự báo**:

### 1. Mốc cắt thời gian dự báo (`Cutoff`) và Tháng liền trước ($T-1$)
- Với mỗi tháng nhãn dự báo $T$ (ví dụ tháng `06/2022`), mốc cắt thời gian dự báo là **`00:00:00` ngày mùng 1 đầu tháng $T$ theo giờ Moscow (`Europe/Moscow`, `UTC+3`)**.
- **Lịch sử tích lũy trước cutoff (`_hist`)**: Toàn bộ sự kiện `TRX`, `GEO`, `DIALOG` phát sinh **nghiêm ngặt trước thời điểm Cutoff** ($t < \text{Cutoff}_T$), tính từ đầu lịch sử dữ liệu (`01/2021`).
- **Tháng liền trước ($T-1$)**: Các sự kiện phát sinh trong đúng tháng ngay trước mốc dự báo (ví dụ nếu dự báo cho tháng $T = \text{06/2022}$ thì tháng $T-1$ là tháng `05/2022`).

### 2. Tập quan sát chuẩn (`Common Cohort` = `1.002.240` dòng khách–tháng)
- Toàn bộ bộ dữ liệu `targets` có `100.224` khách hàng doanh nghiệp $\times$ `12` tháng nhãn (`02/2022` đến `01/2023`) = `1.202.688` dòng.
- Để so sánh đồng nhất giữa khả năng mua hàng trong **1 tháng tới (`H=1`)** và trong **3 tháng tới (`H=3`)**, mỗi mốc dự báo $T$ bắt buộc phải có đủ 3 tháng nhãn tương lai ($T, T+1, T+2$). Do đó, 2 tháng cuối (`12/2022` và `01/2023`) được dùng làm nhãn tương lai $T+1, T+2$ cho mốc `11/2022`, tạo thành **10 mốc dự báo chuẩn từ `02/2022` đến `11/2022`**:
  $$N_{\text{cohort}} = 100.224 \text{ khách hàng} \times 10 \text{ mốc dự báo} = 1.002.240 \text{ quan sát khách–tháng}$$

### 3. Định nghĩa biến mục tiêu (`y_any`, `H=1`, `H=3`)
- Tại mỗi tháng $T$, khách hàng có 4 nhãn sản phẩm nhị phân `target_1` (SP 1 – Thẻ doanh nghiệp), `target_2` (SP 2 – Tín dụng), `target_3` (SP 3 – Chi lương), `target_4` (SP 4 – POS/Acquiring).
- **Mua bất kỳ sản phẩm nào trong tháng $T$ (`H=1` hay `y_any`):**
  $$y_{i, T}^{(H=1)} = \max\left(\text{target}_{1, i, T},\ \text{target}_{2, i, T},\ \text{target}_{3, i, T},\ \text{target}_{4, i, T}\right) \in \{0, 1\}$$
  Trên `1.002.240` quan sát, có đúng **`11.420`** quan sát mua hàng ở tháng $T$, tương ứng **Tỷ lệ mua trung bình toàn tệp (`Mức chung H=1`)**:
  $$\bar{R}_{H=1} = \frac{11.420}{1.002.240} = 1,1394\% \approx 1,14\%$$
- **Mua bất kỳ sản phẩm nào trong vòng 3 tháng tới (`H=3`):**
  $$y_{i, T}^{(H=3)} = \max\left(y_{i, T}^{(H=1)},\ y_{i, T+1}^{(H=1)},\ y_{i, T+2}^{(H=1)}\right) \in \{0, 1\}$$
  Trên `1.002.240` quan sát, có **`30.749`** quan sát mua hàng trong 3 tháng tới, tương ứng **Mức chung `H=3`**:
  $$\bar{R}_{H=3} = \frac{30.749}{1.002.240} = 3,0680\% \approx 3,07\%$$

### 4. Công thức tính Tỷ lệ mua nhóm $G$ và Hệ số nâng (`Lift`)
Với bất kỳ phân khúc khách hàng $G$ nào có $N_G$ dòng quan sát và $B_G$ dòng có $y^{(H=1)} = 1$:
- **Tỷ trọng của nhóm $G$ trong tệp (`% tệp`):**
  $$\text{Share}(G) = \frac{N_G}{N_{\text{tổng}}} \times 100\%$$
- **Tỷ lệ mua tháng tới của nhóm $G$:**
  $$R_{H=1}(G) = \frac{B_G}{N_G} \times 100\%$$
- **Hệ số nâng (`Lift`) so với mức trung bình chung:**
  $$\text{Lift}_{H=1}(G) = \frac{R_{H=1}(G)}{\bar{R}_{H=1}} = \frac{B_G / N_G}{11.420 / 1.002.240}$$
  *(Ví dụ: $\text{Lift} = 4,87\times$ nghĩa là khách hàng thuộc nhóm $G$ có xác suất mua hàng cao gấp **4,87 lần** mức trung bình $1,14\%$ của toàn bộ tập khách hàng).*

---

## Slide A (`slide_01.png`): Giải mã đường cong chữ U — Tại sao nhóm '0 Giao dịch' có tỷ lệ mua cao?

File ảnh: [`outputs/eda_insights_20261119/slides_clean/slide_01.png`](outputs/eda_insights_20261119/slides_clean/slide_01.png)

### 1. Biểu đồ Trái: *Tỷ lệ mua theo Tổng số giao dịch trước cutoff*
- **Cách đọc hiểu:**
  - Trục hoành chia `1.002.240` quan sát thành 4 tầng dựa trên tổng số giao dịch tích lũy từ đầu lịch sử đến trước mốc dự báo (`trx_hist`): `0 giao dịch`, `1 – 9`, `10 – 99`, và `≥ 100`.
  - Trục tung là tỷ lệ mua hàng tháng tới (`H=1`). Đường nét đứt ngang ở mức `1,14%` là tỷ lệ mua trung bình của toàn bộ tập dữ liệu.
  - Nếu nhìn bề ngoài, đồ thị có hình **chữ U**: nhóm `0 giao dịch` mua tới **`1,41%`** (màu cam), cao gần bằng nhóm giao dịch rất nhiều `≥ 100` (**`1,48%`**) và cao hơn hẳn nhóm `1 – 9` (**`0,76%`**) hay `10 – 99` (**`0,59%`**). Điều này tạo cảm giác nghịch lý: *"Tại sao khách không có giao dịch nào lại mua sản phẩm nhiều hơn khách có 1–99 giao dịch?"*
- **Cách tính & Bảng số liệu đối chiếu:**
  - Tính biến $\text{trx\_hist}_{i, T} = \sum_{m < T} \text{trx\_events}_{i, m}$ (tổng số giao dịch trước tháng $T$).

| Nhóm tổng giao dịch trước cutoff (`trx_band`) | Số dòng quan sát ($N_G$) | Tỷ trọng toàn tệp | Số ca mua tháng tới ($B_G$) | Cách tính Tỷ lệ mua (`H=1`) | Hệ số `Lift` so với `1,14%` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`0 giao dịch`** (`trx_hist = 0`) | `84.670` | `8,45%` | `1.190` | `1.190 / 84.670 = 1,41%` | `1,23×` |
| **`1 – 9`** (`1 ≤ trx_hist ≤ 9`) | `101.092` | `10,09%` | `770` | `770 / 101.092 = 0,76%` | `0,67×` |
| **`10 – 99`** (`10 ≤ trx_hist ≤ 99`) | `296.569` | `29,59%` | `1.740` | `1.740 / 296.569 = 0,59%` | `0,51×` |
| **`≥ 100`** (`trx_hist ≥ 100`) | `519.909` | `51,87%` | `7.720` | `7.720 / 519.909 = 1,48%` | `1,30×` |
| **Tổng / Mức chung** | **`1.002.240`** | **`100,00%`** | **`11.420`** | **`11.420 / 1.002.240 = 1,14%`** | **`1,00×`** |

### 2. Biểu đồ Phải: *Bóc tách bên trong nhóm '0 Giao dịch' (`84.670` dòng)*
- **Cách đọc hiểu:**
  - Biểu đồ này zoom sâu vào đúng **`84.670` dòng** của cột `0 giao dịch` bên trái (với `1.190` ca mua) để giải thích tại sao cột đó lại cao `1,41%`.
  - Ba thanh ngang(`Có Hội thoại tháng trước: 3,4%`, `Kích hoạt TRX mới chưa DIALOG: 11,2%`, `Ngủ đông thực sự: 85,4%`) cộng lại bằng đúng **`100%` (`84.670` dòng)** của nhóm `0 giao dịch`:
    1. **`85,4%` nhóm `0 TRX` (`72.304` dòng)** thực chất là **khách ngủ đông thực sự**: tỷ lệ mua chỉ đạt **`0,56%`** (thấp hơn cả nhóm `10–99 TRX` ở mức `0,59%`). Nghĩa là khách ngủ đông thực sự **không hề mua cao**!
    2. Sở dĩ trung bình nhóm `0 TRX` bị kéo lên `1,41%` là vì **`14,6%` còn lại (`12.366` dòng)** là **khách hàng mới đang trong quá trình kích hoạt (onboarding)**: gồm **`3,4%` (`2.890` dòng)** vừa có Hội thoại tư vấn ở tháng $T-1$ (tỷ lệ mua vọt lên **`9,41%`**) và **`11,2%` (`9.476` dòng)** vừa bắt đầu phát sinh giao dịch đầu tiên ngay trong tháng mua $T$ (tỷ lệ mua **`5,45%`**). Chỉ `14,6%` này đã tạo ra **`788 / 1.190 = 66,2%`** tổng số lượt mua của toàn nhóm `0 TRX`!
  - Thanh màu xanh dương ở giữa (**`Có vị trí GEO ≥ 4 vùng trước cutoff: 23,5% nhóm 0 TRX`**) trả lời câu hỏi thực chiến: *"Tại thời điểm trước mốc dự báo (chưa được phép nhìn vào tháng $T$), ngoài cờ Hội thoại tháng trước (`3,4%` nhóm), còn biến nào giúp nhận diện khách đang hoạt động bên trong nhóm `0 TRX` không?"* Câu trả lời là **Số vùng địa lý `geohash_4` hoạt động ở tháng $T-1$**: `23,5%` nhóm `0 TRX` có di chuyển qua `≥ 4 vùng GEO` đạt tỷ lệ mua **`1,66%`** ở `H=1` (và **`5,60%`** ở `H=3`).
- **Cách tính & Bảng số liệu đối chiếu (Bên trong `84.670` dòng có `trx_hist = 0`):**

| Nhóm con bên trong `0 Giao dịch` (`trx_hist = 0`) | Điều kiện lọc toán học | Số dòng ($N_G$) | `% trong nhóm 0 TRX` (`N_G / 84.670`) | Số ca mua ($B_G$) | Cách tính Tỷ lệ mua (`H=1`) | Tỷ lệ mua 3 tháng (`H=3`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Có Hội thoại tháng trước** *(Chỉ báo trước cutoff)* | `trx_hist = 0` VÀ `dialog_events(T-1) > 0` | `2.890` | **`3,41%`** (`3,4%`) | `272` | `272 / 2.890 = 9,41%` | `14,53%` (`420` ca) |
| **2. Kích hoạt `TRX` mới (chưa `DIALOG`)** *(Kiểm chứng cơ chế ở tháng $T$)* | `trx_hist = 0` VÀ `dialog_events(T-1) = 0` VÀ `trx_events(T) > 0` | `9.476` | **`11,19%`** (`11,2%`) | `516` | `516 / 9.476 = 5,45%` | `10,12%` (`959` ca) |
| **3. Ngủ đông thực sự** *(Phần còn lại của nhóm `0 TRX`)* | `trx_hist = 0` VÀ `dialog_events(T-1) = 0` VÀ `trx_events(T) = 0` | `72.304` | **`85,40%`** (`85,4%`) | `402` | `402 / 72.304 = 0,56%` | `3,90%` (`2.818` ca) |
| **Tổng 3 nhóm phân hoạch (`1 + 2 + 3`)** | Toàn bộ `trx_hist = 0` | **`84.670`** | **`100,00%`** | **`1.190`** | **`1.190 / 84.670 = 1,41%`** | **`4,96%` (`4.197` ca)** |
| **Chỉ báo phụ trước cutoff: Có vị trí `GEO ≥ 4 vùng`** | `trx_hist = 0` VÀ `g4_unique_cells(T-1) ≥ 4` | `19.889` | **`23,49%`** (`23,5%`) | `331` | `331 / 19.889 = 1,66%` | `5,60%` (`1.114` ca) |

---

## Slide B (`slide_02.png`): Hiệu ứng Tuổi lịch sử (`Tenure`) — Cùng số giao dịch nhưng ý nghĩa trái ngược

File ảnh: [`outputs/eda_insights_20261119/slides_clean/slide_02.png`](outputs/eda_insights_20261119/slides_clean/slide_02.png)

### 1. Biểu đồ Trái: *Cùng `10–99` giao dịch theo Số tháng lịch sử*
- **Cách đọc hiểu:**
  - Nếu chỉ đưa biến tổng số giao dịch tích lũy (`trx_hist`) vào mô hình, hai khách hàng cùng có `50 giao dịch` sẽ bị coi là giống hệt nhau.
  - Biểu đồ này cố định hoàn toàn các yếu tố khác (**cùng có `10–99 giao dịch` tích lũy, cùng `0 lần mua trước đó`, và cùng `0 DIALOG` ở tháng $T-1$**) rồi chia theo **Số tháng lịch sử quan sát được trước cutoff (`history_age_months`)**:
    - Nếu `10–99 giao dịch` đó diễn ra chỉ trong **`1 – 6 tháng`** (tức tốc độ giao dịch hàng tháng cao): tỷ lệ mua tháng tới đạt **`1,14%`**.
    - Nếu diễn ra trong **`7 – 12 tháng`**: tỷ lệ mua giảm xuống **`0,70%`**.
    - Nếu rải rác suốt **`≥ 13 tháng`** (tức trung bình mỗi tháng chỉ có vài giao dịch thưa thớt): tỷ lệ mua sụt xuống chỉ còn **`0,31%`** (**thấp hơn `3,7 lần`** so với nhóm `1 – 6 tháng` dù có cùng tổng số giao dịch!).
- **Cách tính & Bảng số liệu đối chiếu:**
  - **Định nghĩa `history_age_months` (Số tháng lịch sử):**
    $$\text{history\_age\_months}_{i, T} = \text{date\_diff}\left(\text{'month'},\ \min_{m < T,\ \text{any\_event}_{i,m} > 0}(m),\ T\right)$$
    tức là số tháng tính từ tháng đầu tiên khách hàng $i$ xuất hiện bất kỳ sự kiện nào (`TRX`, `GEO`, hoặc `DIALOG`) cho tới mốc dự báo $T$.
  - Điều kiện lọc cố định của biểu đồ: `prior_purchase_state = '0_prior_buys'`, `dialog_recent = False`, và `trx_band = '10-99'`.

| Nhóm Số tháng lịch sử (`history_age_months`) | Số dòng quan sát ($N_G$) | Số ca mua ($B_G$) | Cách tính Tỷ lệ mua (`H=1`) | Tỷ lệ mua 3 tháng (`H=3`) | Tỷ số chênh lệch so với `≥ 13 tháng` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`1 – 6 tháng`** (`1 ≤ history_age_months ≤ 6`) | `9.054` | `103` | `103 / 9.054 = 1,14%` | `2,77%` | **`3,67×` (`~3,7 lần`)** |
| **`7 – 12 tháng`** (`7 ≤ history_age_months ≤ 12`) | `29.777` | `208` | `208 / 29.777 = 0,70%` | `1,96%` | **`2,25×`** |
| **`≥ 13 tháng`** (`history_age_months ≥ 13`) | `210.112` | `652` | `652 / 210.112 = 0,31%` | `0,95%` | **`1,00×` (Mốc so sánh)** |

### 2. Biểu đồ Phải: *Tỷ số Giao dịch tháng $T-1$ / Trung bình tháng trước đó*
- **Cách đọc hiểu:**
  - Biểu đồ này đo **động lượng giao dịch ngắn hạn (momentum)** của từng khách hàng bằng cách lấy số giao dịch ở tháng liền trước ($T-1$) chia cho mức giao dịch trung bình hàng tháng của chính khách hàng đó trong giai đoạn trước $T-1$ (từ đầu lịch sử đến $T-2$).
  - Kết quả tạo thành một bậc thang tăng đơn điệu tuyệt đối:
    - Khách từng có giao dịch trước $T-1$ nhưng ở tháng $T-1$ không giao dịch (**`0×`, chiếm `14,8%` tệp**): tỷ lệ mua sập xuống **`0,25%`**.
    - Giao dịch tháng $T-1$ thấp hơn một nửa mức trung bình cá nhân (**`< 0,5×`, chiếm `10,8%` tệp**): tỷ lệ mua **`0,75%`**.
    - Giao dịch tháng $T-1$ duy trì quanh mức trung bình cá nhân (**`0,5× – 1,5×`, chiếm `52,1%` tệp**): tỷ lệ mua **`1,19%`**.
    - Giao dịch tháng $T-1$ vượt trên 1,5 lần mức trung bình cá nhân (**`> 1,5×`, chiếm `12,7%` tệp**): tỷ lệ mua tăng lên **`1,88%`**.
    - Trước tháng $T-1$ chưa từng có giao dịch, và **lần đầu tiên phát sinh giao dịch ở tháng $T-1$** (**`Trước T−1 = 0, T−1 > 0`, chiếm `1,1%` tệp**): tỷ lệ mua vọt lên **`4,21%`** (**gấp `3,69 lần`** mức chung!).
- **Cách tính & Bảng số liệu đối chiếu:**
  - Gọi $\text{trx}_{i, T-1}$ là số giao dịch trong tháng liền trước $T-1$.
  - Gọi $\text{trx\_prior}_{i, \le T-2} = \sum_{m \le T-2} \text{trx}_{i, m}$ là tổng giao dịch tính đến tháng $T-2$, và $M_{i, \le T-2}$ là số tháng kể từ tháng có giao dịch đầu tiên của khách hàng đến tháng $T-2$.
  - Trung bình giao dịch hàng tháng của khách hàng trước tháng $T-1$ là:
    $$\mu_{i, \le T-2} = \frac{\text{trx\_prior}_{i, \le T-2}}{M_{i, \le T-2}}$$
  - **Tỷ số giao dịch ngắn hạn (`trx_surge_ratio`):**
    $$\text{Ratio}_{i, T-1} = \frac{\text{trx}_{i, T-1}}{\mu_{i, \le T-2}}$$

| Nhóm Tỷ số $\text{TRX}_{T-1} / \mu_{\le T-2}$ | Điều kiện phân nhóm | Số dòng ($N_G$) | `% toàn tệp` | Số ca mua ($B_G$) | Cách tính Tỷ lệ mua (`H=1`) | Hệ số `Lift` (`H=1`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`0×`** | $\text{trx\_prior}_{\le T-2} > 0$ VÀ $\text{trx}_{T-1} = 0$ | `148.660` | `14,83%` | `374` | `374 / 148.660 = 0,25%` | `0,22×` |
| **`< 0,5×`** | $\text{trx\_prior}_{\le T-2} > 0$ VÀ $0 < \text{Ratio}_{T-1} < 0,5$ | `108.443` | `10,82%` | `813` | `813 / 108.443 = 0,75%` | `0,66×` |
| **`0,5× – 1,5×`** | $\text{trx\_prior}_{\le T-2} > 0$ VÀ $0,5 \le \text{Ratio}_{T-1} \le 1,5$ | `522.281` | `52,11%` | `6.198` | `6.198 / 522.281 = 1,19%` | `1,04×` |
| **`> 1,5×`** | $\text{trx\_prior}_{\le T-2} > 0$ VÀ $\text{Ratio}_{T-1} > 1,5$ | `127.562` | `12,73%` | `2.398` | `2.398 / 127.562 = 1,88%` | `1,65×` |
| **`Trước T−1 = 0, T−1 > 0`** | $\text{trx\_prior}_{\le T-2} = 0$ VÀ $\text{trx}_{T-1} > 0$ | `10.624` | `1,06%` | `447` | `447 / 10.624 = 4,21%` | **`3,69×`** |
| *(Nhóm tham chiếu: `0 TRX` toàn lịch sử $\le T-1$)* | $\text{trx\_hist}_{\le T-1} = 0$ *(đã phân tích ở Slide A)* | `84.670` | `8,45%` | `1.190` | `1.190 / 84.670 = 1,41%` | `1,23×` |

---

## Slide C (`slide_03.png`): Kiểm định đa tầng — Tác động của Hội thoại (`DIALOG`) theo từng tầng `TRX` và Lịch sử mua

File ảnh: [`outputs/eda_insights_20261119/slides_clean/slide_03.png`](outputs/eda_insights_20261119/slides_clean/slide_03.png)

### 1. Biểu đồ Trái & 3 Thẻ chỉ số Phải
- **Cách đọc hiểu:**
  - Một câu hỏi phản biện thường gặp trong thống kê là **Nghịch lý Simpson (Simpson's Paradox)**: *"Phải chăng khách hàng có Hội thoại (`DIALOG`) mua nhiều hơn chỉ vì họ vốn là khách hàng lớn, giao dịch nhiều hoặc đã từng mua sản phẩm trước đó?"*
  - Để trả lời dứt điểm, biểu đồ này **khóa chặt đồng thời 3 biến gây nhiễu**: chỉ xét riêng nhóm khách hàng đã có lịch sử gắn bó lâu năm (**`≥ 13 tháng`**), sau đó chia nhỏ theo **Số lần mua trước đó (`0 lần mua` vs `≥ 1 lần mua`)** và **Tổng số giao dịch tích lũy (`1–9 TRX`, `10–99 TRX`, `≥ 100 TRX`)**. Trong từng nhóm nhỏ đó, ta so sánh trực tiếp cột màu xám (**`0 DIALOG ở tháng T−1`**) với cột màu xanh ngọc (**`≥ 1 DIALOG ở tháng T−1`**):
    1. **Bác bỏ Nghịch lý Simpson:** Trong tất cả các tầng (kể cả khi khách chưa từng mua và có rất ít giao dịch), cột xanh (`≥ 1 DIALOG`) luôn cao vượt trội so với cột xám (`0 DIALOG`).
    2. **Quy luật bão hòa (Tương tác âm giữa `DIALOG` và nền hoạt động sẵn có):**
       - Ở nhóm **`0 lần mua trước` & `1–9 TRX` hoặc `10–99 TRX`**, khi không có hội thoại, khách gần như không mua (**`0,20%`** và **`0,31%`**). Nhưng khi có `≥ 1 DIALOG ở T−1`, tỷ lệ mua nhảy vọt lên **`3,35%`** và **`3,29%`** — tức **tăng gấp `16,6 lần` và `10,6 lần`**!
       - Ở nhóm **`≥ 1 lần mua trước` & `≥ 100 TRX`**, ngay cả khi không có hội thoại, tỷ lệ mua tự nhiên đã rất cao (**`3,84%`** ở `H=1`, **`9,26%`** ở `H=3`). Khi có thêm `≥ 1 DIALOG ở T−1`, tỷ lệ mua đạt đỉnh tuyệt đối **`5,00%`** ở `H=1` (**`10,38%`** ở `H=3`), tương ứng hệ số nhân tương đối **`1,3×`**.
- **Cách tính & Bảng số liệu đối chiếu (Nhóm `history_age_months ≥ 13`):**

| Phân tầng (`Lịch sử mua × Tầng TRX` tại `Lịch sử ≥ 13 tháng`) | Nhóm `0 DIALOG ở T−1` ($B_{\text{False}} / N_{\text{False}}$) | Tỷ lệ mua khi `0 DIALOG` | Nhóm `≥ 1 DIALOG ở T−1` ($B_{\text{True}} / N_{\text{True}}$) | Tỷ lệ mua khi `≥ 1 DIALOG` | Hệ số nhân nội tầng (`Tỷ lệ Có / Không DIALOG`) | Tỷ lệ mua `H=3` (Không $\to$ Có `DIALOG`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`0 lần mua trước` \| `1 – 9 TRX`** | `69 / 34.331` | **`0,20%`** | `19 / 568` | **`3,35%`** | **`16,64×` (`16,6×`)** | `0,69% → 4,23%` |
| **`0 lần mua trước` \| `10 – 99 TRX`** | `652 / 210.112` | **`0,31%`** | `184 / 5.588` | **`3,29%`** | **`10,61×` (`10,6×`)** | `0,95% → 4,97%` |
| **`0 lần mua trước` \| `≥ 100 TRX`** | `4.462 / 399.513` | **`1,12%`** | `901 / 28.829` | **`3,13%`** | **`2,80×` (`2,8×`)** | `2,98% → 6,29%` |
| **`≥ 1 lần mua trước` \| `≥ 100 TRX`** | `1.093 / 28.459` | **`3,84%`** | `324 / 6.484` | **`5,00%`** | **`1,30×` (`1,3×`)** | `9,26% → 10,38%` |

---

## Slide D (`slide_04.png`): Tại sao tỷ lệ mua giảm một nửa từ tháng `04/2022` (`1,69%`) xuống `11/2022` (`0,87%`)?

File ảnh: [`outputs/eda_insights_20261119/slides_clean/slide_04.png`](outputs/eda_insights_20261119/slides_clean/slide_04.png)

### 1. Biểu đồ Trái: *Phân rã mức giảm `-0,83%` (`04/2022` → `11/2022`)*
- **Cách đọc hiểu:**
  - Tỷ lệ mua toàn tệp giảm từ đỉnh **`1,694%`** (tháng `04/2022`) xuống đáy **`0,865%`** (tháng `11/2022`), tức giảm **`-0,829 điểm %`** (`-0,83%`).
  - Khi thấy tỷ lệ mua toàn tệp giảm mạnh như vậy, có 2 giả thuyết:
    - *Giả thuyết 1 (Do cơ cấu khách hàng xấu đi):* Tỷ trọng khách hàng tiềm năng (ví dụ khách có Hội thoại) bị thu hẹp theo thời gian.
    - *Giả thuyết 2 (Do tỷ lệ mua nội nhóm giảm):* Ngay trong cùng một phân khúc khách hàng, xác suất mua hàng của họ giảm xuống theo thời gian (do yếu tố mùa vụ / kinh tế vĩ mô).
  - Phép phân rã **Shift-Share** tách con số `-0,83%` thành 3 thành phần cộng tính chính xác (`+0,21% - 0,30% - 0,75% = -0,83%`):
    1. **Dịch chuyển cơ cấu (`+0,21 điểm %`):** Cơ cấu khách hàng không hề xấu đi mà ngược lại còn tốt lên (vì tỷ lệ khách có `DIALOG` tăng hơn gấp đôi từ `3,61%` lên `7,86%`), giúp kéo tỷ lệ mua chung tăng `+0,21%`.
    2. **Tương tác pha loãng (`-0,30 điểm %`):** Khi độ phủ kênh `DIALOG` mở rộng gấp đôi, mức chênh lệch tuyệt đối của nhóm có `DIALOG` bị pha loãng nhẹ.
    3. **Giảm tỷ lệ mua nội nhóm (`-0,75 điểm %`, chiếm `90%` tổng mức giảm):** Cả nhóm có `DIALOG` lẫn nhóm không có `DIALOG` đều giảm tỷ lệ mua theo thời gian.
- **Công thức Toán học của Phân rã Shift-Share:**
  Chia tập khách hàng tại tháng $t \in \{\text{04/2022}, \text{11/2022}\}$ thành các phân khúc $g \in \{\text{Có DIALOG gần},\ \text{Không có DIALOG gần}\}$ với tỷ trọng dân số $w_{g, t}$ ($\sum_g w_{g, t} = 1$) và tỷ lệ mua nội nhóm $r_{g, t}$. Tỷ lệ mua toàn tệp tại tháng $t$ là $R_t = \sum_g w_{g, t} r_{g, t}$.
  Chênh lệch giữa tháng 11 (`11`) và tháng 04 (`04`) được phân rã hằng đẳng thức:
  $$\Delta R = R_{11} - R_{04} = \underbrace{\sum_g \left(w_{g, 11} - w_{g, 04}\right) r_{g, 04}}_{\text{1. Dịch chuyển cơ cấu } (+0,214\%)} + \underbrace{\sum_g w_{g, 04} \left(r_{g, 11} - r_{g, 04}\right)}_{\text{2. Giảm tỷ lệ nội nhóm } (-0,746\%)} + \underbrace{\sum_g \left(w_{g, 11} - w_{g, 04}\right)\left(r_{g, 11} - r_{g, 04}\right)}_{\text{3. Tương tác pha loãng } (-0,297\%)}$$

| Phân khúc $g$ | Tỷ trọng tháng 04 ($w_{g, 04}$) | Tỷ lệ mua tháng 04 ($r_{g, 04}$) | Tỷ trọng tháng 11 ($w_{g, 11}$) | Tỷ lệ mua tháng 11 ($r_{g, 11}$) | Thay đổi tỷ trọng ($\Delta w_g$) | Thay đổi tỷ lệ mua ($\Delta r_g$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Có `DIALOG` ở $T-1$** | `3,610%` | `6,468%` | `7,858%` | `2,792%` | **`+4,249%`** | `-3,676%` |
| **Không có `DIALOG` ở $T-1$** | `96,390%` | `1,515%` | `92,142%` | `0,701%` | `-4,249%` | `-0,815%` |
| **Toàn tệp ($R_t = \sum w_{g,t} r_{g,t}$)** | **`100,00%`** | **`1,694%`** | **`100,00%`** | **`0,865%`** | `0,00%` | **`-0,829%`** |

### 2. Biểu đồ Phải: *Độ phủ `DIALOG` tăng gấp đôi từ `06/22` nhưng `Lift` vẫn bền*
- **Cách đọc hiểu & Cách tính:**
  - **Cột màu xanh dương nhạt (Trục tung bên trái):** Tỷ lệ `%` khách hàng có phát sinh `DIALOG` ở tháng liền trước $T-1$ tại từng mốc dự báo từ `02/2022` đến `11/2022`:
    - Giai đoạn `02/2022 – 06/2022` (tức hoạt động `DIALOG` trong các tháng `01/2022 – 05/2022`): độ phủ duy trì ở mức **`2,64% – 3,96%`**.
    - Từ mốc dự báo `07/2022` trở đi (tức hoạt động `DIALOG` từ tháng `06/2022` sau bước nhảy chế độ thu thập log): độ phủ tăng gấp đôi lên **`7,25% – 8,55%`**.
  - **Đường gấp khúc màu xanh ngọc (Trục tung bên phải):** Hệ số `Lift` của nhóm có `DIALOG` so với tỷ lệ mua trung bình của **chính tháng đó**:
    $$\text{Lift}_T(\text{DIALOG}) = \frac{R_{H=1, T}(\text{Có DIALOG ở } T-1)}{\bar{R}_{H=1, T}}$$
    Mặc dù tỷ lệ mua nền thay đổi theo tháng và độ phủ `DIALOG` tăng gấp đôi từ giữa năm, hệ số `Lift` theo từng tháng vẫn duy trì ổn định trong dải **`3,3× – 5,1×`** suốt cả 10 tháng!

---

## Slide E (`slide_05.png`): Phân tích Đa nhãn — Đồng mua cùng tháng & Hành trình Bán chéo tuần tự ($A \to B$)

File ảnh: [`outputs/eda_insights_20261119/slides_clean/slide_05.png`](outputs/eda_insights_20261119/slides_clean/slide_05.png)

### 1. Biểu đồ Trái: *Ma trận `4 × 4` Hệ số `Lift` mua sản phẩm $B$ khi đã từng mua $A$*
- **Cách đọc hiểu:**
  - Dòng $A$ (trục dọc) là sản phẩm khách hàng **đã từng mua ít nhất 1 lần trong các tháng trước đó** ($m < T$).
  - Cột $B$ (trục ngang) là sản phẩm khách hàng **mua ở tháng dự báo hiện tại** ($T$).
  - **4 ô trên đường chéo chính (`Tái mua`):** Khách đã mua sản phẩm nào trong quá khứ thì xác suất mua lại chính sản phẩm đó ở tháng tiếp theo cao gấp **`3,2×` (`SP 1`)**, **`63,3×` (`SP 2 - Tín dụng`)**, **`3,4×` (`SP 3 - Chi lương`)**, và **`14,4×` (`SP 4 - POS`)** so với mức mua trung bình của sản phẩm đó.
  - **12 ô ngoài đường chéo chính (`Bán chéo`):** Khách đã mua sản phẩm $A$ trong quá khứ có xác suất mua mới sản phẩm $B$ ($B \neq A$) ở tháng tiếp theo cao gấp từ **`2,0×` đến `4,4×`** mức nền! Nổi bật nhất là:
    - `Đã mua SP 2 (Tín dụng) → Mua SP 1 (Thẻ DN)`: **`4,4×`** (`4,37×`)
    - `Đã mua SP 1 (Thẻ DN) → Mua SP 2 (Tín dụng)`: **`4,1×`** (`4,06×`)
    - `Đã mua SP 2 (Tín dụng) → Mua SP 3 (Chi lương)`: **`3,8×`** (`3,82×`)
    - `Đã mua SP 3 (Chi lương) → Mua SP 2 (Tín dụng)`: **`3,8×`** (`3,75×`)
    - `Đã mua SP 1 (Thẻ DN) → Mua SP 3 (Chi lương)`: **`3,1×`** (`3,07×`)
    - `Đã mua SP 3 (Chi lương) → Mua SP 4 (POS)`: **`3,0×`** (`2,97×`)
- **Cách tính & Bảng số liệu đối chiếu:**
  - Tập quan sát: Toàn bộ `11` tháng nhãn từ `03/2022` đến `01/2023` (các tháng có ít nhất 1 tháng lịch sử nhãn trước đó): $100.224 \times 11 = 1.102.464$ quan sát khách–tháng.
  - Với mỗi cặp sản phẩm $(A, B) \in \{1, 2, 3, 4\}^2$:
    - Đặt cờ lịch sử $\text{Prior}_{i, T}(A) = \mathbb{I}\left(\sum_{m < T} \text{target}_{A, i, m} > 0\right)$.
    - Xác suất mua $B$ ở tháng $T$ khi đã từng mua $A$:
      $$P\left(\text{target}_{B, T} = 1 \mid \text{Prior}_T(A) = 1\right) = \frac{\sum_{i, T} \mathbb{I}\left(\text{Prior}_{i, T}(A) = 1 \land \text{target}_{B, i, T} = 1\right)}{\sum_{i, T} \mathbb{I}\left(\text{Prior}_{i, T}(A) = 1\right)}$$
    - Hệ số `Lift` so với tỷ lệ mua trung bình của sản phẩm $B$ trên cùng tập `1.102.464` dòng ($\bar{P}(\text{target}_{B, T} = 1)$):
      $$\text{Lift}(A \to B) = \frac{P\left(\text{target}_{B, T} = 1 \mid \text{Prior}_T(A) = 1\right)}{\bar{P}\left(\text{target}_{B, T} = 1\right)}$$
  - Tỷ lệ mua trung bình (baseline) của 4 sản phẩm trên `1.102.464` dòng:
    - $\bar{P}(\text{SP 1}) = 0,4358\%$
    - $\bar{P}(\text{SP 2}) = 0,0493\%$
    - $\bar{P}(\text{SP 3}) = 0,3859\%$
    - $\bar{P}(\text{SP 4}) = 0,2560\%$

| Sản phẩm đã mua trước đó ($A$) | Số dòng có $\text{Prior}(A)=1$ | Tỷ lệ mua & Lift sang **SP 1 (Thẻ DN)** | Tỷ lệ mua & Lift sang **SP 2 (Tín dụng)** | Tỷ lệ mua & Lift sang **SP 3 (Chi lương)** | Tỷ lệ mua & Lift sang **SP 4 (POS)** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`Đã mua SP 1`** | `32.958` | `1,40%` (**`3,20×`** · `460` ca) | `0,20%` (**`4,06×`** · `66` ca) | `1,18%` (**`3,07×`** · `390` ca) | `0,74%` (**`2,89×`** · `244` ca) |
| **`Đã mua SP 2`** | `2.784` | `1,90%` (**`4,37×`** · `53` ca) | `3,13%` (**`63,33×`** · `87` ca) | `1,47%` (**`3,82×`** · `41` ca) | `0,68%` (**`2,67×`** · `19` ca) |
| **`Đã mua SP 3`** | `27.053` | `0,86%` (**`1,98×`** · `233` ca) | `0,18%` (**`3,75×`** · `50` ca) | `1,29%` (**`3,35×`** · `350` ca) | `0,76%` (**`2,97×`** · `206` ca) |
| **`Đã mua SP 4`** | `15.221` | `0,87%` (**`2,00×`** · `133` ca) | `0,12%` (**`2,40×`** · `18` ca) | `0,86%` (**`2,23×`** · `131` ca) | `3,69%` (**`14,42×`** · `562` ca) |

### 2. Biểu đồ Phải: *Các cặp sản phẩm mua gộp cùng tháng (`647` ca)*
- **Cách đọc hiểu & Cách tính:**
  - Trên toàn bộ `12` tháng nhãn (`1.202.688` dòng), có `12.854` dòng khách–tháng phát sinh mua ít nhất 1 sản phẩm (`y_any = 1`), trong đó:
    - Mua đúng **1 sản phẩm**: `12.207` dòng (**`94,97%`** số ca mua).
    - Mua gộp **$\ge 2$ sản phẩm trong cùng 1 tháng**: **`647` dòng (`5,03%` số ca mua)**, gồm `620` dòng mua 2 sản phẩm, `26` dòng mua 3 sản phẩm và `1` dòng mua cả 4 sản phẩm.
  - Biểu đồ đếm số dòng khách–tháng mà từng cặp sản phẩm $(A, B)$ cùng nhận giá trị `1` trong cùng một tháng ($\text{target}_{A, i, T} = 1 \land \text{target}_{B, i, T} = 1$):
    1. **`SP 1 + SP 3` (Thẻ DN + Chi lương):** **`263` lượt** (gồm `238` ca chỉ mua cặp `1+3`, `24` ca mua bộ ba có `1+3`, và `1` ca mua cả 4 SP).
    2. **`SP 1 + SP 4` (Thẻ DN + POS):** **`205` lượt**.
    3. **`SP 3 + SP 4` (Chi lương + POS):** **`131` lượt**.
    4. **`SP 1 + SP 2` (Thẻ DN + Tín dụng):** **`49` lượt**.
    5. **`SP 2 + SP 3` (Tín dụng + Chi lương):** **`32` lượt**.
    6. **`SP 2 + SP 4` (Tín dụng + POS):** **`24` lượt**.

---

## Slide F (`slide_06.png`): Phát hiện mới về `TRX` — Phát sinh Loại giao dịch (`event_type`) mới báo hiệu khả năng mua tăng `4,9 lần`

File ảnh: [`outputs/eda_insights_20261119/slides_clean/slide_06.png`](outputs/eda_insights_20261119/slides_clean/slide_06.png)
Nguồn dữ liệu: [`outputs/eda_figures_moscow/event_type_monthly_profiles.parquet`](outputs/eda_figures_moscow/event_type_monthly_profiles.parquet) kết hợp với [`outputs/full_population_temporal_eda_20261118/all_customer_months.parquet`](outputs/full_population_temporal_eda_20261118/all_customer_months.parquet).

### 1. Biểu đồ Trái: *Tỷ lệ mua theo Số mã giao dịch (`event_type`) mới ở $T-1$*
- **Cách đọc hiểu:**
  - Trong dữ liệu giao dịch `TRX`, mỗi giao dịch có một mã loại nghiệp vụ `event_type` (tổng cộng có `55` mã `event_type` khác nhau).
  - Một phát hiện rất mạnh là: **Không phải cứ giao dịch nhiều là sẽ mua sản phẩm mới, mà quan trọng là doanh nghiệp có đang phát sinh loại nghiệp vụ giao dịch mới chưa từng dùng trước đó hay không!**
  - Cụ thể, khi xét tháng liền trước $T-1$:
    - **`0 TRX ở T−1` (`23,3%` tệp):** Tỷ lệ mua chỉ **`0,67%`** (`0,59×`).
    - **Có giao dịch ở $T-1$, nhưng `0 mã mới` (`66,3%` tệp — chỉ lặp lại các mã giao dịch cũ đã dùng từ các tháng trước):** Tỷ lệ mua chỉ đạt **`1,02%`** (`0,90×`, thấp hơn cả mức trung bình chung `1,14%`!).
    - **Phát sinh `1 mã mới` lần đầu tiên ở tháng $T-1$ (`8,1%` tệp):** Tỷ lệ mua nhảy gấp đôi lên **`2,20%`** (**`1,93×` Lift**).
    - **Phát sinh `≥ 2 mã mới` lần đầu tiên ở tháng $T-1$ (`2,3%` tệp, `22.984` dòng):** Tỷ lệ mua vọt lên **`5,55%`** (**`4,87×` Lift** chung; và đặc biệt tăng **`6,58×`** đối với **SP 3 Chi lương** và **`6,51×`** đối với **SP 2 Tín dụng**!).
- **Cách tính & Bảng số liệu đối chiếu:**
  - Với mỗi khách hàng $i$ và tháng $m$, gọi $E_{i, m}$ là tập hợp các mã `event_type` xuất hiện trong tháng $m$.
  - Tập hợp các mã `event_type` khách hàng đã từng dùng trước tháng $T-1$ là $\mathcal{H}_{i, \le T-2} = \bigcup_{m \le T-2} E_{i, m}$.
  - **Số mã giao dịch mới xuất hiện ở tháng $T-1$ (`et_new_cells_prev`):**
    $$\text{NewTypes}_{i, T-1} = \left| E_{i, T-1} \setminus \mathcal{H}_{i, \le T-2} \right|$$

| Nhóm Số mã `event_type` mới ở $T-1$ | Số dòng ($N_G$) | `% toàn tệp` | Số ca mua ($B_G$) | Cách tính Tỷ lệ mua (`H=1`) | Hệ số `Lift` (`H=1`) | Tỷ lệ mua 3 tháng (`H=3`) | `Lift` riêng cho SP 2 (Tín dụng) / SP 3 (Chi lương) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`0 TRX ở T−1`** ($\text{trx}_{T-1} = 0$) | `233.330` | `23,28%` | `1.564` | `1.564 / 233.330 = 0,67%` | `0,59×` | `2,31%` | `0,16×` / `0,22×` |
| **`0 mã mới`** ($\text{trx}_{T-1} > 0,\ \text{NewTypes}_{T-1} = 0$) | `664.832` | `66,33%` | `6.800` | `6.800 / 664.832 = 1,02%` | `0,90×` | `2,86%` | `0,97×` / `0,95×` |
| **`1 mã mới`** ($\text{NewTypes}_{T-1} = 1$) | `81.094` | `8,09%` | `1.781` | `1.781 / 81.094 = 2,20%` | **`1,93×`** | `5,09%` | `2,09×` / `2,11×` |
| **`≥ 2 mã mới`** ($\text{NewTypes}_{T-1} \ge 2$) | `22.984` | `2,29%` | `1.275` | `1.275 / 22.984 = 5,55%` | **`4,87×`** | **`9,77%`** | **`6,51×` / `6,58×`** |

### 2. Biểu đồ Phải: *Tỷ lệ mua theo Số loại giao dịch (`event_type`) ở $T-1$*
- **Cách đọc hiểu & Cách tính:**
  - Biến **`et_unique_cells_prev` ($\left|E_{i, T-1}\right|$)** đếm số mã `event_type` khác nhau mà doanh nghiệp sử dụng trong tháng $T-1$ (đo **độ đa dạng nghiệp vụ tài chính** của doanh nghiệp trong tháng đó):
    - Chỉ dùng **`1 loại`** (`20,3%` tệp): tỷ lệ mua chỉ **`0,46%`** (`0,41×`).
    - Dùng **`2 – 3 loại`** (`27,4%` tệp): tỷ lệ mua **`0,76%`** (`0,67×`).
    - Dùng **`4 – 6 loại`** (`17,6%` tệp): tỷ lệ mua **`1,73%`** (`1,52×`).
    - Dùng **`≥ 7 loại`** (`11,5%` tệp): tỷ lệ mua đạt **`3,27%`** (**`2,87×` Lift**, cao gấp **`7,1 lần`** so với nhóm chỉ dùng 1 loại giao dịch!).

| Nhóm Số loại `event_type` ở $T-1$ ($\left|E_{i, T-1}\right|$) | Số dòng ($N_G$) | `% toàn tệp` | Số ca mua ($B_G$) | Cách tính Tỷ lệ mua (`H=1`) | Hệ số `Lift` (`H=1`) | Tỷ lệ mua 3 tháng (`H=3`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`1 loại`** ($\left|E_{T-1}\right| = 1$) | `203.005` | `20,26%` | `938` | `938 / 203.005 = 0,46%` | `0,41×` | `1,32%` |
| **`2 – 3 loại`** ($2 \le \left|E_{T-1}\right| \le 3$) | `274.838` | `27,42%` | `2.099` | `2.099 / 274.838 = 0,76%` | `0,67×` | `2,01%` |
| **`4 – 6 loại`** ($4 \le \left|E_{T-1}\right| \le 6$) | `175.933` | `17,55%` | `3.052` | `3.052 / 175.933 = 1,73%` | `1,52×` | `4,40%` |
| **`≥ 7 loại`** ($\left|E_{T-1}\right| \ge 7$) | `115.134` | `11,49%` | `3.767` | `3.767 / 115.134 = 3,27%` | **`2,87×`** | **`8,17%`** |

---

## Slide G (`slide_07.png`): Mối liên hệ `TRX × GEO` — Hai chiều mở rộng độc lập giúp phân tách người mua gấp `8,2 lần`

File ảnh: [`outputs/eda_insights_20261119/slides_clean/slide_07.png`](outputs/eda_insights_20261119/slides_clean/slide_07.png)
Nguồn dữ liệu: [`outputs/eda_figures_moscow/geohash_4_monthly_profiles.parquet`](outputs/eda_figures_moscow/geohash_4_monthly_profiles.parquet) kết hợp với `event_type_monthly_profiles.parquet`.

### 1. Biểu đồ Trái: *Tỷ lệ mua theo Số vùng `geohash_4` ở tháng $T-1$*
- **Cách đọc hiểu & Cách tính:**
  - Mỗi ô lưới `geohash_4` tương ứng một vùng địa lý rộng khoảng `39 km × 19,5 km`. Biến **`g4_unique_cells_prev`** đếm số vùng `geohash_4` khác nhau mà khách hàng có phát sinh tín hiệu vị trí trong tháng $T-1$.
  - Khi số vùng `geohash_4` trong tháng $T-1$ tăng từ `0 vùng` → `1 vùng` → `2 – 4 vùng` → `5 – 9 vùng` → `≥ 10 vùng`, tỷ lệ mua chung (`y_any`) tăng đơn điệu từ **`0,95%`** lên **`1,44%`**.
  - Đặc biệt, đối với **Sản phẩm 2 (Tín dụng doanh nghiệp)**, nhóm doanh nghiệp hoạt động rộng trên **`≥ 10 vùng geohash_4`** có tỷ lệ vay tín dụng đạt **`0,107%`**, cao gấp **`2,17 lần`** mức trung bình toàn tệp!

| Nhóm Số vùng `geohash_4` ở $T-1$ | Số dòng ($N_G$) | `% toàn tệp` | Số ca mua ($B_G$) | Cách tính Tỷ lệ mua (`H=1`) | Hệ số `Lift` chung (`H=1`) | Hệ số `Lift` riêng cho **SP 2 (Tín dụng)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`0 vùng`** (`0 GEO ở T−1`) | `419.674` | `41,87%` | `3.987` | `3.987 / 419.674 = 0,95%` | `0,83×` | `0,65×` |
| **`1 vùng`** (`1 ô geohash_4`) | `104.878` | `10,46%` | `1.174` | `1.174 / 104.878 = 1,12%` | `0,98×` | `0,77×` |
| **`2 – 4 vùng`** (`2 – 4 ô geohash_4`) | `278.677` | `27,81%` | `3.508` | `3.508 / 278.677 = 1,26%` | `1,10×` | `1,22×` |
| **`5 – 9 vùng`** (`5 – 9 ô geohash_4`) | `151.429` | `15,11%` | `2.065` | `2.065 / 151.429 = 1,36%` | `1,20×` | `1,35×` |
| **`≥ 10 vùng`** (`≥ 10 ô geohash_4`) | `47.582` | `4,75%` | `686` | `686 / 47.582 = 1,44%` | **`1,27×`** | **`2,17×`** |

### 2. Biểu đồ Phải: *Tương tác Số loại `TRX` × Số vùng `geohash_4` (ở $T-1$)*
- **Cách đọc hiểu:**
  - Khi kiểm tra tương quan hạng Spearman trên `460.028` dòng khách–tháng có mặt ở cả hai nguồn `TRX` và `GEO` trong tháng $T-1$:
    - $\rho_{\text{Spearman}}(\text{Số lượng TRX}_{T-1},\ \text{Số lượng GEO}_{T-1}) = 0,0160$
    - $\rho_{\text{Spearman}}(\text{Số loại event\_type}_{T-1},\ \text{Số vùng geohash\_4}_{T-1}) = 0,0267$
    - $\rho_{\text{Spearman}}(\text{Số mã event\_type mới}_{T-1},\ \text{Số vùng geohash\_4 mới}_{T-1}) = 0,0160$
  - Hệ số tương quan $r \approx 0,02$ chứng minh **`TRX` và `GEO` hoàn toàn độc lập (trực giao)**: `TRX` đo độ phức tạp/mở rộng về mặt **tài chính**, còn `GEO` đo độ mở rộng về mặt **địa bàn hoạt động vật lý**.
  - Do hai chiều này độc lập, khi ghép chéo **3 nhóm `TRX` ở tháng $T-1$** trên trục hoành với **3 nhóm số vùng `GEO` ở tháng $T-1$** (3 màu cột: Xám = `0 vùng GEO`, Xanh dương = `1 – 3 vùng GEO`, Xanh ngọc = `≥ 4 vùng GEO`), tỷ lệ mua tăng bậc thang theo cả hai chiều từ **`0,42%`** lên tới **`3,44%`** (**chênh lệch gấp `8,2 lần`**):
- **Cách tính & Bảng số liệu đối chiếu (`3 × 3` tổ hợp trên toàn bộ `768.910` dòng có giao dịch `TRX` ở $T-1$):**

| Nhóm Giao dịch `TRX` ở tháng $T-1$ | Nhóm Vị trí `GEO` ở tháng $T-1$ | Số dòng ($N_G$) | Số ca mua ($B_G$) | Cách tính Tỷ lệ mua (`H=1`) | Hệ số `Lift` (`H=1`) | Tỷ lệ mua 3 tháng (`H=3`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **1. `1 – 3 loại TRX (0 mã mới)`** | **`0 vùng GEO`** | `187.848` | `787` | `787 / 187.848 = 0,42%` | `0,37×` | `1,27%` |
| **1. `1 – 3 loại TRX (0 mã mới)`** | **`1 – 3 vùng GEO`** | `134.497` | `736` | `736 / 134.497 = 0,55%` | `0,48×` | `1,62%` |
| **1. `1 – 3 loại TRX (0 mã mới)`** | **`≥ 4 vùng GEO`** | `112.818` | `672` | `672 / 112.818 = 0,60%` | `0,52×` | `1,78%` |
| **2. `≥ 4 loại TRX (0 mã mới)`** | **`0 vùng GEO`** | `81.280` | `1.515` | `1.515 / 81.280 = 1,86%` | `1,64×` | `4,91%` |
| **2. `≥ 4 loại TRX (0 mã mới)`** | **`1 – 3 vùng GEO`** | `78.959` | `1.607` | `1.607 / 78.959 = 2,04%` | `1,79×` | `5,52%` |
| **2. `≥ 4 loại TRX (0 mã mới)`** | **`≥ 4 vùng GEO`** | `69.430` | `1.483` | `1.483 / 69.430 = 2,14%` | `1,87×` | `5,84%` |
| **3. `≥ 1 mã TRX mới ở tháng T−1`** | **`0 vùng GEO`** | `39.754` | `1.009` | `1.009 / 39.754 = 2,54%` | `2,23×` | `5,29%` |
| **3. `≥ 1 mã TRX mới ở tháng T−1`** | **`1 – 3 vùng GEO`** | `32.854` | `963` | `963 / 32.854 = 2,93%` | `2,57×` | `6,39%` |
| **3. `≥ 1 mã TRX mới ở tháng T−1`** | **`≥ 4 vùng GEO`** | `31.470` | `1.084` | `1.084 / 31.470 = 3,44%` | **`3,02×`** | **`6,89%`** |

---

## Slide H (`slide_08.png`): Tổng kết các phát hiện chuyên sâu & Đặc trưng cần đưa vào Mô hình

File ảnh: [`outputs/eda_insights_20261119/slides_clean/slide_08.png`](outputs/eda_insights_20261119/slides_clean/slide_08.png)

Trang cuối chuyển hóa toàn bộ các con số từ Slide A đến Slide G thành **danh mục công thức Đặc trưng (Feature Engineering)** và **Quy tắc Huấn luyện Mô hình**:

| Nhóm Khám phá | Đặc trưng cần tạo cho Mô hình Propensity | Công thức tính toán tại mốc dự báo $T$ |
| :--- | :--- | :--- |
| **1. Cơ chế chữ U & Nhóm `0 TRX`** | • `is_zero_trx_hist`<br>• `zero_trx_x_dialog_T1`<br>• `zero_trx_x_geo_g4_T1` | • $\mathbb{I}(\text{trx\_hist}_{<T} = 0)$<br>• $\mathbb{I}(\text{trx\_hist}_{<T} = 0) \times \mathbb{I}(\text{dialog}_{T-1} > 0)$<br>• $\mathbb{I}(\text{trx\_hist}_{<T} = 0) \times \text{g4\_unique\_cells}_{T-1}$ |
| **2. Số tháng lịch sử & Tỷ số `TRX(T−1)`** | • `history_age_months`<br>• `trx_rate_per_tenure_month`<br>• `trx_surge_ratio_T1` | • $\text{date\_diff}(\text{'month'}, \text{first\_any\_event\_month}, T)$<br>• $\text{trx\_hist}_{<T} / \max(\text{history\_age\_months}, 1)$<br>• $\text{trx}_{T-1} / \max(\mu_{\text{trx}, \le T-2}, \epsilon)$ |
| **3. Số mã `TRX` mới × Số vùng `GEO`** | • `et_new_cells_T1`<br>• `et_unique_cells_T1`<br>• `g4_unique_cells_T1`<br>• `et_new_x_g4_cells_T1` | • $\left| E_{T-1} \setminus \bigcup_{m \le T-2} E_m \right|$<br>• $\left| E_{T-1} \right|$<br>• $\left| G^{(4)}_{T-1} \right|$<br>• $\text{et\_new\_cells}_{T-1} \times \text{g4\_unique\_cells}_{T-1}$ |
| **4. Tương tác `DIALOG`, Drift & Bán chéo** | • `prior_buy_p1..p4`<br>• `dialog_T1_x_tenure_x_trx`<br>• Hiệu chỉnh xác suất (`Calibration`) | • $\mathbb{I}\left(\sum_{m < T} \text{target}_{k, m} > 0\right)$ với $k \in \{1, 2, 3, 4\}$<br>• Đặc trưng tương tác giữa `dialog_T1`, `history_age_months` và `trx_hist`<br>• Dùng kiểm định trượt theo thời gian (`Rolling temporal validation`) và hiệu chỉnh đầu ra mô hình trên tháng gần nhất để khử hiện tượng suy giảm tỷ lệ mua nội nhóm theo thời gian. |
