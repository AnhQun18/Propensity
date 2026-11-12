# Rà soát workspace và phản biện dự án — 10/11/2026

Kết luận: dự án có baseline chạy thật và số liệu validation kiểm tra lại được. Nút thắt hiện tại là thiết kế đối chứng, khả năng kiểm soát thay đổi và mức độ kết luận từ thí nghiệm. Hai lỗi kiểm soát trong code đã tái hiện; một yếu tố gây nhiễu trong phép so lịch sử đã đo được trên artifact thật. Các lỗi này chưa chứng minh kết quả baseline đã công bố bị sai.

## Phạm vi và bằng chứng

- Kiểm kê repository tại HEAD `edf39da`: 303 file được Git quản lý, gồm 87 file Python, 43 Markdown và 5 notebook. Rà soát sâu luồng preparation → calendar → tune → fit/calibration → compare-rule/test, bộ tests, notebook, báo cáo và metadata của các run liên quan.
- Đối chiếu tài liệu hiện hành và các snapshot EDA, PDF, DOCX, slide cũ. Snapshot có ngày/phạm vi riêng được xem theo đúng thời điểm; không coi việc chưa có baseline trong một báo cáo ngày 06/11 là lỗi của báo cáo đó.
- Kiểm kê file cho thấy `data/` khoảng 20,903 GB, `outputs/` khoảng 11,091 GB tại thời điểm rà soát. Không đọc lại từng byte raw, không quét lại toàn bộ 105 triệu sự kiện, không kiểm tra trực quan mọi trang/hình. Một số thư mục fixture tạm cũ bị chặn quyền truy cập, không nằm trong các artifact baseline đã xác minh.
- Không có bản xuất toàn bộ lịch sử chat trong workspace được tìm thấy. Các câu “người dùng quyết định” trong tài liệu là ghi chép của repo; phiên hiện tại không đủ để đối chiếu độc lập mọi quyết định với chat gốc.
- Chạy lại toàn bộ tests: **62/62 đạt**, 157,899 giây. Lần đầu bị sandbox chặn thư mục tạm; lần chạy ngoài sandbox đạt, vì vậy không quy lỗi quyền thành lỗi ứng dụng.
- `sync_preparation_notebook.py --check`: notebook 02/03/04 đồng bộ helper, output nguồn trống. Không có liên kết file cục bộ bị gãy trong 43 Markdown được Git quản lý, theo kiểm tra liên kết đơn giản.
- `locked_run(config, 'fitted')` đạt với run thật. Tính lại metric từ dự đoán validation đã lưu khớp chính xác `validation_metrics.json`; manifest vẫn `test_opened=false`.
- Không sửa code sản phẩm, raw, model, calibrator hoặc báo cáo baseline; không fit model mới, không đọc feature hay đánh giá dự đoán test tháng 01/2023.

Bằng chứng cục bộ: [reproductions.json](../outputs/workspace_review_20261110/reproductions.json), [calendar_proxy.json](../outputs/workspace_review_20261110/calendar_proxy.json), [script tái hiện](../outputs/workspace_review_20261110/reproduce_checks.py). `outputs/` bị Git ignore; các bằng chứng này chỉ có trên máy hiện tại.

## Các phát hiện ưu tiên cao

### F01 — P1 — Trainer chấp nhận protocol có tháng tương lai trong train

**Loại:** lỗi kiểm soát đã tái hiện; chưa thấy run thật dùng lịch sai.

Trong [runner.py](../src/propensity_baseline/runner.py), `source_contract()` đọc `protocol.json`, nhưng chỉ kiểm tra ngân sách chính bằng 1%. Hàm tái dựng và so sánh schema từ preparation config, chưa tái dựng và so sánh protocol. `tune()` sau đó dùng trực tiếp `fold['train']` trong JSON để đọc dữ liệu học.

Thử thay nội dung protocol **trong bộ nhớ**, thêm `2023-01` vào train của fold validation `2022-09` và đổi timezone thành `Europe/Moscow`: `source_contract()` vẫn chấp nhận. Không thực sự đọc tháng test hoặc fit theo lịch lỗi. `locked_run()` phát hiện thay đổi sau tuning, nhưng một protocol sai có sẵn trước tuning sẽ được khóa như đầu vào hợp lệ.

**Hệ quả:** hợp đồng “train chỉ dùng quá khứ” chưa được bảo đảm tại biên trainer. Việc preparation từng xuất protocol đúng không đủ nếu file đã bị sửa/ghép nhầm trước tuning.

**Cần xử lý:** tái dựng `evaluation_protocol(preparation_config, months)` và yêu cầu bằng hoàn toàn protocol đã lưu; kiểm tra train/validation/calibration/test, thời điểm nhãn khả dụng và thứ tự tháng. Thêm test từ chối tháng tương lai, train trùng validation và protocol lệch config. Protocol của run thật hiện vẫn khớp preparation.

### F02 — P1 — Cache EDA có thể giữ kết quả phụ thuộc đã lỗi thời

**Loại:** lỗi cache đã tái hiện trên SQL tổng hợp.

Trong [review_mbd_eda.py](../src/review_mbd_eda.py), `QueryCache.get()` dùng `VERSION + SQL` của chính truy vấn làm khóa. Không có digest của các bảng cache mà truy vấn đó đọc. Các phụ thuộc thật gồm `sequence_reconciliation_* → cache_month_*`, `amount_outliers_currency → cache_amount_by_currency`.

Ví dụ tái hiện: parent trả `1`, child đọc parent và lưu `1`; đổi SQL parent để trả `2`; gọi lại SQL child không đổi. Parent được tính lại thành `2`, child trả cache cũ `1`.

**Hệ quả:** sửa một truy vấn nguồn có thể tạo báo cáo trộn kết quả mới/cũ dù raw inventory vẫn đúng. Test hiện có kiểm tra invalidation của truy vấn được sửa, chưa kiểm tra các truy vấn phụ thuộc.

**Cần xử lý:** khai báo dependency và đưa digest của chúng vào khóa, hoặc vô hiệu hóa toàn bộ kết quả dẫn xuất khi một truy vấn nguồn thay đổi. Tăng `VERSION` toàn cục là cách làm thủ công, dễ bị quên. Chưa có bằng chứng cache đang công bố đã gặp trường hợp lỗi này.

### F03 — P1 — Phép so lịch sử cũ đang thêm cả biến đại diện cho thời gian

**Loại:** yếu tố gây nhiễu thí nghiệm đã đo trên feature/artifact thật; không phải bằng chứng leakage.

[features.py](../src/propensity_data/features.py) tính overlap từ mốc đầu nguồn chung cho toàn bộ khách. [calendar_schema()](../src/propensity_baseline/data.py) đưa các cột `*_older_*` vào biến thể history, bao gồm:

| Biến | Số giá trị khác nhau trong từng tháng 02–11/2022 | Importance trong model cuối |
| --- | ---: | ---: |
| `trx_older_source_window_overlap_days` | 1 | 1,61575% |
| `geo_older_source_window_overlap_days` | 1 | 1,77252% |
| `dialog_older_source_window_overlap_days` | 1 | 2,71007% |
| Tổng | — | **6,09834%** |

Ba biến chỉ tăng theo tháng và chênh nhau bởi mốc bắt đầu nguồn. Chúng cung cấp thông tin thời gian chung; không mô tả sự khác nhau về hành vi giữa hai khách trong cùng tháng. Chúng không hiện diện ở biến thể recent. Một biến hằng trong tháng vẫn có thể thay đổi cách model học qua nhiều tháng và tương tác với feature khách.

**Hệ quả:** mức tăng Recall@1% `7,55% → 8,70%` là lợi ích của **toàn bộ nhóm biến thêm vào**, chưa cô lập được lợi ích của hành vi cũ. Không lấy 6,10% importance làm tỷ lệ giải thích mức tăng metric; importance cũng không đo đóng góp nhân quả của từng nguồn.

**Cần xử lý:** đối chứng `recent + cùng biến thời gian` với history, hoặc cấp cùng thông tin tháng cho cả hai. Có thể bỏ các bản sao overlap khỏi X nhưng vẫn giữ để tính rate/audit, rồi so trên validation. Không tự bỏ các biến hay phủ nhận run hiện tại trước khi chạy đối chứng.

### F04 — P1 — Thiếu đối chứng lịch sử mua đủ mạnh

**Loại:** hạn chế đã được báo cáo hiện hành thừa nhận, chưa được khép bằng thí nghiệm.

[BASELINE_REPORT.md](../BASELINE_REPORT.md) đã ghi đúng rằng model chỉ nhận hành vi, rule nhận `y_any(M−1)`. Vượt rule một lag không cho biết hành vi bổ sung bao nhiêu so với một model biết lịch sử mua 3/6 tháng, từng sản phẩm và độ dài lịch sử nhãn.

**Hệ quả:** chưa xác định nên đầu tư vào nguồn hành vi nào, mô hình phức tạp nào hoặc giá trị tăng thêm của dữ liệu đa nguồn. Đây là thiếu đối chứng, không phải kết luận nhãn quá khứ tự động gây leakage.

**Cần xử lý trước mở rộng model:** so R = rule, H = lịch sử nhãn, B = hành vi hiện tại, B+H = kết hợp, cùng cohort/split/ngân sách và thời điểm nhãn khả dụng. Báo cả nhóm mua lại và chưa từng có nhãn dương trong lịch sử quan sát. Không diễn giải nhóm thứ hai thành khách chưa từng mua trong đời.

### F05 — P1 — Chưa kiểm chứng ngoài mẫu cho toàn bộ quy trình cuối

**Loại:** giới hạn bằng chứng, không phải lỗi số học metric.

Ba tháng 09–11 vừa chọn biến thể vừa cung cấp số liệu và khoảng bootstrap. Phân nhóm, nhiều K và overlap tiếp tục khai thác cùng các dự đoán đó; chúng không tăng số giai đoạn đánh giá độc lập. Bootstrap theo khách giữ được sự lặp ID, nhưng không tạo thêm tháng tương lai hay phản ánh đầy đủ việc chọn model/train lại.

Còn một khác biệt quy trình: validation base học đến tháng ngay trước tháng đánh giá; quy trình cuối giữ base học đến 11, calibrate 12 và đánh giá 01. Chưa có backtest ngoài mẫu của cả chuỗi **base → calibration → tháng kế tiếp** trong phần phát triển.

**Cần xử lý:** khóa ngân sách thí nghiệm, chạy backtest ba bước trên các tháng phát triển để kiểm tra raw/calibrated probability, chọn cấu hình, rồi mở holdout cuối. Giữ kết luận hiện tại ở mức validation thăm dò. Không cần “mở test ngay” để lấp mọi thiếu sót thiết kế.

### F06 — P1 — Giả định thời gian còn thiếu đánh giá độ nhạy

**Loại:** giả định đã công khai và được ghi là quyết định người dùng; không coi UTC hay mã tiền 11 là lỗi cần tự đổi.

Nhãn 12/2022 được giả định sẵn đúng 00:00 ngày 01/01/2023, đồng thời là cutoff test. Chưa mô phỏng thời gian chốt dữ liệu, huấn luyện calibrator và chấm điểm. Audit 3 giờ đã đo ảnh hưởng đầu vào trên 71.467 cặp khách–tháng, chưa đo ảnh hưởng lên metric. Các kịch bản độ trễ mới kiểm tra lịch khả thi, chưa có model tương ứng.

**Hệ quả:** có thể bảo vệ đây là mô phỏng offline dưới giả định đã nêu; chưa suy ra lịch vận hành khả thi hoặc độ bền của mức tăng metric.

**Cần xử lý:** khi mục tiêu chuyển sang mô phỏng triển khai, chạy ít nhất một kịch bản trễ dương với lịch fit/calibration/rule được dựng lại và một đối chứng cutoff nếu cần so benchmark ngoài repo. Không chỉ đổi `label_delay_days` rồi giữ nguyên lịch còn lại. Xem [audit nhãn khả dụng](propensity_feasibility/remaining_questions/assessment.md).

## Các khoảng cách cần đóng tiếp theo

### F07 — P2 — Chưa tách được đóng góp của từng nguồn và nội dung

Run thật chỉ có hai model đa nguồn. `trx-only` được code hỗ trợ nhưng chưa có trong kết quả; chưa có hai tổ hợp `trx+geo`, `trx+dialog`. Dialog chỉ có metadata, src/dst chủ yếu là cardinality/missing, mã giao dịch chưa được biểu diễn đầy đủ trong X.

Feature importance gộp theo nguồn chưa thay thế ablation, đặc biệt khi nhóm nguồn còn chứa biến lịch như F03. Chưa có cơ sở chọn dialog/geo là hướng ưu tiên chỉ từ số lượng feature hoặc importance.

**Hành động:** ablation nguồn trên cùng tập H, cùng cửa sổ và thông tin thời gian; sau đó mới so metadata với embedding hoặc tần suất mã. Vocabulary/PCA cần học trong từng train fold. Việc chưa giải mã tên thương mại sản phẩm không chặn thí nghiệm any-product.

### F08 — P2 — Độ dài lịch sử và độ phủ khách còn trộn trong aggregate

`older` là toàn bộ quá khứ trước tháng gần nhất; khách có lịch sử lâu/khác nguồn được quan sát khác nhau. Rate chia theo ngày bắt đầu nguồn chung không phải exposure riêng của khách; hai khách có thời điểm bắt đầu sử dụng khác nhau vẫn cùng mẫu số. Repo đã ghi hạn chế này, chưa có đối chứng giải quyết tác động trên metric.

**Hành động:** thử cửa sổ cố định 7/14/30/90 ngày, xu hướng gần/xa và cờ đủ lịch sử; đánh giá theo cohort có/không nguồn và độ dài lịch sử trước cutoff. Không suy ngày mở tài khoản từ log đầu, không tự xóa khách thưa hoặc cap heavy user để làm metric đẹp hơn.

### F09 — P2 — Mức thử nghiệm model còn hẹp

Hai biến thể × một cấu hình × ba fold = sáu fit. Run chưa có `eval_set`, early stopping hoặc đường học validation/AP theo số cây; model theo fold không được lưu. Đường train Logloss của model cuối không cho biết fold bị underfit hay overfit.

**Hành động:** sau F03/F04, thực hiện một grid nhỏ có ngân sách, tập dừng bên trong train và lưu fold model/đường học. Chưa có bằng chứng để khẳng định encoder mạnh hơn, CatBoost đã đạt trần hoặc thêm cây chắc chắn cải thiện.

### F10 — P2 — Khóa artifact và tái lập còn thiếu một phần chuỗi phụ thuộc

`runner` hash các file trong `propensity_baseline/`; các helper nhập từ `propensity_data/` chưa được bao phủ trọn bởi hash code. Preparation/calendar dùng inventory path/size/mtime thay vì checksum nội dung; giới hạn này được tài liệu nêu đúng. Prediction/metric validation chưa được đóng dấu nội dung trong manifest chính như model/calibrator; script review có bổ sung hash nhưng là bước riêng.

**Hệ quả:** cơ chế hiện có bắt được nhiều thay đổi thông thường, nhưng câu “đổi code/input là bị từ chối” cần giới hạn theo những file thực sự được theo dõi. Việc metadata khớp không chứng minh phép tính aggregate đúng hoặc nội dung mọi artifact còn nguyên.

**Hành động:** lưu Git commit + trạng thái dirty, hash các helper có ảnh hưởng, checksum feature/prediction/schema/protocol tại lúc tạo và manifest liên kết cha–con. Không bắt buộc hash lại toàn bộ raw ở mọi lệnh chỉ đọc; có thể dùng manifest checksum được tạo một lần.

### F11 — P2 — Cài đặt và bàn giao chưa có một hợp đồng môi trường thống nhất

`pyproject.toml` công bố Python >=3.10, trong khi DuckDB trong extra prepare/baseline chỉ được khai báo cho Python >=3.11; tài liệu workflow yêu cầu >=3.11. Người cài extra trên 3.10 có thể không nhận DuckDB rồi gặp lỗi import. Extra `dev` chỉ có `build`, chưa đủ cho toàn bộ tests vốn nhập cả `nbformat` và các thư viện optional khác. Chưa có lockfile chung để dựng lại môi trường trên máy khác; run có lưu version nên vẫn có bằng chứng môi trường thực nghiệm.

**Hành động:** làm rõ Python tối thiểu theo workflow, thêm một extra/test environment đủ dependency và bản khóa môi trường nghiên cứu. Kiểm tra cài đặt trong môi trường mới; 62 test đạt trên `.venv` hiện có chưa thay thế kiểm tra đó.

### F12 — P2 — Đích nghiên cứu, đích liên hệ và đích kinh doanh cần phân biệt rõ khi bàn giao

`y_any` đo mua quan sát được của bốn target. K=1% là ngân sách mô phỏng. Chưa có ràng buộc được liên hệ, khách sở hữu sản phẩm, chi phí, lợi nhuận hoặc phân bổ can thiệp. Quần thể cố định `client_split × tháng` không chứng minh từng khách đủ điều kiện tiếp cận ở mọi cutoff; cold-start cũng không tự xác định khách mới.

Trong 3.009 lượt chọn khách–tháng validation, model có 238 lượt dương, rule 169. Chênh 69 là lượt mua quan sát trong danh sách, chưa phải 69 lượt mua do chiến dịch tạo ra. Model top 1% còn bỏ ngoài khoảng 91,3% ca dương trung bình tháng. Hai cách nhìn phải được trình bày cùng nhau.

**Hành động:** nếu phạm vi vẫn là nghiên cứu, ghi rõ giới hạn và tiêu chí kết thúc nghiên cứu. Nếu chuyển sang CRM, cần eligibility tại cutoff, khả năng liên hệ, cooldown và mục tiêu giá trị được xác nhận; hiệu quả can thiệp cần thiết kế đánh giá riêng. Không bắt dự án nghiên cứu phải có ROI trước khi tiếp tục baseline.

### F13 — P2 — Chưa có luồng scoring cho một tháng tương lai chưa có targets

Luồng preparation yêu cầu mỗi tháng cấu hình có ít nhất một target; trainer/evaluator đọc nhãn và kiểm tra summary theo manifest. Có lưu model/calibrator, nhưng CLI chưa có lệnh inference độc lập cho tháng mới và quần thể đủ điều kiện tại cutoff. Đây là khoảng cách bàn giao, nằm ngoài phần baseline offline đã làm.

**Hành động:** khi cần dùng model, tách feature generation phục vụ scoring khỏi yêu cầu có bảng nhãn tháng đích; lưu contract cho model + calibrator + feature builder và kiểm thử reload/scoring. Không cần triển khai API để hoàn thành nghiên cứu hiện tại.

### F14 — P3 — Chi phí duy trì nhiều snapshot và luồng chạy còn cao

Notebook 02/03 cùng tạo feature v2; notebook 04 merge v2 nhưng không nằm trên đường baseline v3; notebook 05 đọc trực tiếp v3. Hai output preparation chiếm khoảng 4,838 GB và 5,200 GB; một run calendar v1 bị ngắt vẫn được giữ. Các báo cáo cũ và README có ngày/phạm vi, nhưng người mới vẫn phải ghép nhiều trang mới biết đường chạy hiện tại.

**Hành động:** thêm một sơ đồ lệnh và bảng trạng thái duy nhất: input → output → schema → run hiện hành → kết quả được công bố. Giữ snapshot cần truy vết, chỉ rõ artifact chuẩn. Không tự xóa notebook, dữ liệu hay báo cáo mà người dùng từng yêu cầu. Hạn chế tiếp tục tạo audit/tài liệu mới cho những câu hỏi đã có câu trả lời; ưu tiên các thí nghiệm còn thiếu ở F03/F04/F07.

## Kiểm tra bổ sung: ảnh hưởng của phá hòa rule

Rule nhị phân tạo nhiều ties. Seed cố định giúp tái lập nhưng tạo một danh sách cụ thể, không phải kỳ vọng của mọi danh sách hợp lệ. Từ dự đoán rule validation đã lưu, kỳ vọng chọn ngẫu nhiên trong nhóm đồng điểm cho kết quả:

| Tháng | Lượt trúng rule đã công bố | Kỳ vọng lượt trúng trong ties | Độ lệch chuẩn do ties |
| --- | ---: | ---: | ---: |
| 09/2022 | 57 | 62,023 | 2,576 |
| 10/2022 | 55 | 54,457 | 0,713 |
| 11/2022 | 57 | 56,103 | 1,045 |

Macro Recall kỳ vọng của rule khoảng **6,289%**, so với **6,179%** của seed hiện tại. Chênh model với rule kỳ vọng khoảng **2,414 điểm phần trăm**, thay cho **2,524**. Không thay thế kết quả đã khóa hoặc khoảng bootstrap bằng các số này; chúng trả lời câu hỏi khác. Kết luận model hơn rule trong thí nghiệm hiện tại không bị đảo ngược. Có thể bổ sung kỳ vọng hoặc sensitivity theo seed để giảm phụ thuộc cách phá hòa khi so hard rule.

## Thứ tự xử lý đề xuất

1. Đóng hai lỗi F01/F02 bằng sửa nhỏ và test tái hiện; không cần mở test thật.
2. Khóa một bảng thí nghiệm R/H/B/B+H, đồng thời kiểm soát biến thời gian của F03. Giữ cohort/cutoff/ngân sách nhất quán.
3. Chạy ablation nguồn và một phạm vi tuning nhỏ; lưu đường học/model theo fold, chẩn đoán cohort và chi phí chạy. Với probability, kiểm tra chuỗi base → calibration → tháng tiếp theo trong phần phát triển.
4. Chọn model/calibrator theo quy tắc đã định, đánh giá holdout cuối sau khi các quyết định hoàn tất. Báo cáo cả chất lượng và giới hạn của một tháng test.
5. Hoàn thiện môi trường tái lập và luồng scoring nếu đầu ra cần bàn giao sử dụng.

Những điều chưa có bằng chứng để kết luận: baseline đã dùng tương lai; các metric công bố sai; UTC theo quyết định người dùng là lỗi code; thiếu tên thương mại sản phẩm chặn any-product; phải xóa heavy user; model chắc chắn underfit; embedding chắc chắn tốt hơn; hoặc feature importance chứng minh giá trị tăng thêm của từng nguồn.
