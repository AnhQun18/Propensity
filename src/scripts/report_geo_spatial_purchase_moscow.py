"""Render the human-readable report from the completed Moscow association audit."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

OUT=Path('outputs/geo_spatial_purchase_moscow_20261117')


def main():
    d=pd.read_parquet(OUT/'customer_snapshot.parquet')
    s=pd.read_csv(OUT/'geo_direct_standardization.csv')
    support=pd.read_csv(OUT/'visitor_cell_support.csv').query('horizon==3')
    tests=pd.read_csv(OUT/'global_cell_permutation_all_definitions.csv')
    visitor=tests[tests.condition=='within_trx_strata_all_visitors']
    cell=pd.read_parquet(OUT/'visitor_hierarchical_cell_rates.parquet',filters=[('horizon','=',3),('strength','=',100),('level','in',['g4','g5'])])
    fig,axes=plt.subplots(1,2,figsize=(14,6),dpi=170)
    for ax,level in zip(axes,['g4','g5']):
        z=cell[(cell.level==level)&cell.n.ge(100)].nlargest(12,'n').sort_values('n')
        yy=np.arange(len(z))
        ax.errorbar(z.rate,yy,xerr=np.vstack([np.maximum(0,z.rate-z.wilson_low),np.maximum(0,z.wilson_high-z.rate)]),fmt='o',color='#8293a8',label='Raw rate + 95% Wilson')
        ax.scatter(z.smoothed,yy,marker='D',s=24,color='#208575',label='Hierarchical shrinkage m=100',zorder=3)
        ax.axvline(d.y_3.mean(),color='grey',ls='--',lw=1,label='All-customer baseline')
        ax.set_yticks(yy,[f'{int(r.cell)} (n={int(r.n):,})' for r in z.itertuples()])
        ax.xaxis.set_major_formatter(PercentFormatter(1));ax.set_title(f'{level}: 12 most-visited cells')
        ax.set_xlabel('3-month cumulative purchase rate');ax.grid(axis='x',alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Cell purchase rates among unique visitors | 01 Nov 2022 Moscow cutoff',fontsize=15)
    fig.tight_layout(rect=(0,.05,1,.94))
    fig.text(.02,.015,'A customer may visit multiple cells. Wilson intervals are pointwise. Cells displayed by support, not by purchase rate.',fontsize=10)
    fig.savefig(OUT/'visitor_cell_rates_wilson.png');plt.close(fig)
    lines=['# GEO và tỷ lệ mua — kiểm tra theo giờ Moscow','',
        '**Cutoff:** 01/11/2022 00:00 Europe/Moscow (UTC+3), tương đương 31/10/2022 21:00 UTC.',
        f'**Cohort:** {len(d):,} khách duy nhất; {int(d.has_geo.sum()):,} khách đã có GEO trước cutoff. Mỗi khách chỉ xuất hiện một lần.',
        '**Lịch sử:** toàn bộ lịch sử có sẵn của từng nguồn trước cutoff, không giới hạn 7/30/90 ngày. Không dùng cache đặc trưng UTC cũ.',
        '**Nhãn:** OR của bốn sản phẩm trong tháng 11/2022; tháng 11–12/2022; tháng 11/2022–01/2023, lần lượt cho H=1/2/3.',
        '**Giả định nguồn:** timestamp không timezone của raw tiếp tục được hiểu là UTC như quy ước chuẩn hóa trước đây. Giờ Moscow xác định cutoff và ngày hoạt động; đây không phải xác minh timezone ETL gốc.','',
        '## 1. Tỷ lệ mua theo ô và khoảng tin cậy','',
        'Có hai bảng khác nhau: (a) mọi ô từng ghé, đếm khách riêng biệt trong từng ô; (b) ô chính theo nhánh g4 → g5 → g6, mỗi khách được gán một ô ở mỗi cấp. Bảng mọi ô là kết quả chính cho câu hỏi vị trí đã ghé. Không cộng số khách/người mua giữa các ô vì có giao nhau.','',
        '| Cấp, mọi ô từng ghé | Số ô | Trung vị khách/ô | Ô có ≥100 khách | Ô có <10 khách |',
        '|---|---:|---:|---:|---:|']
    for r in support.itertuples():lines.append(f'| {r.level} | {r.cells:,} | {r.median_n:g} | {r.cells_n_ge100:,} | {r.cells_n_lt10:,} |')
    lines += ['', 'Ngưỡng n≥100 là tiêu chí hiển thị mô tả, không bảo đảm đủ lực kiểm định. Toàn bộ ô vẫn được xuất; CI 95% Wilson là khoảng tin cậy từng ô, không điều chỉnh đồng thời cho hàng nghìn ô. Ô hiển thị trên hình được chọn theo số khách, không chọn theo tỷ lệ mua cao.', '',
        '![Tỷ lệ mua theo mọi ô từng ghé](visitor_cell_rates_wilson.png)','',
        '## 2. Làm trơn phân cấp','',
        '`p_smoothed = (buyers + m × p_parent_smoothed) / (n + m)`; g6 co về g5, g5 co về g4, g4 co về tỷ lệ mua chung của toàn cohort, độc lập cho từng horizon.',
        'Xuất m=20,100,500 để xem độ nhạy; hình dùng m=100. Đây là làm trơn giả đếm phân cấp theo tham số định trước, chưa phải empirical Bayes ước lượng từ dữ liệu. Tỷ lệ cha chứa cả khách của ô con; đây là ước lượng mô tả dạng plug-in, không phải posterior đầy đủ. Wilson chỉ áp dụng cho tỷ lệ thô; không gắn Wilson thô vào ước lượng đã co.',
        'Xuất cả định nghĩa mọi ô từng ghé và ô chính. Ước lượng này dùng nhãn của kỳ phân tích để mô tả; không đưa trực tiếp vào mô hình dự báo cùng kỳ. Target encoding cần fit bằng nhãn quá khứ đã sẵn có và đánh giá ngoài thời gian.','',
        '## 3. Khác biệt giữa các ô có vượt nhiễu mẫu không?','',
        'Dùng tổng bình phương sai lệch số người mua quan sát so với kỳ vọng, chuẩn hóa theo phương sai từng ô. Phân phối tham chiếu lấy bằng hoán vị, không dùng xấp xỉ chi-square cho ô rất thưa.',
        'Đối với mọi ô từng ghé: 499 hoán vị nhãn khách trong từng tầng TRX của nhóm có GEO; mọi cạnh của cùng khách luôn mang cùng nhãn hoán vị. Do đó giữ được việc một khách ghé nhiều ô. Đối với ô chính: 999 hoán vị, thêm đối chiếu hoán vị không phân tầng. Hiệu chỉnh Holm chung cho 18 kiểm định toàn cục.', '',
        '| Horizon | p hoán vị g4, mọi ô | p hoán vị g5, mọi ô |', '|---|---:|---:|']
    for h in [1,2,3]:
        z=visitor[visitor.horizon==h].set_index('level')
        lines.append(f'| {h} tháng | {z.loc["g4","permutation_p"]:.3f} | {z.loc["g5","permutation_p"]:.3f} |')
    lines += ['',f'Khoảng p sau Holm (mọi ô): {visitor.holm_p.min():.3f}–{visitor.holm_p.max():.3f}. Không bác bỏ giả thuyết không chỉ có nghĩa là chưa có bằng chứng khác biệt theo thiết kế này; không chứng minh mọi ô có cùng tỷ lệ mua. Phân tầng TRX còn thô; không kiểm soát mọi yếu tố nhiễu.','',
        '## 4. Có GEO sau phân tầng TRX','',
        'T0: không có bản ghi TRX quan sát trước cutoff (không đồng nghĩa khách thực sự không giao dịch). T1–T5: quintile số sự kiện TRX dương trên toàn lịch sử. Giữ các giá trị bằng nhau trong cùng tầng. Chuẩn hóa trực tiếp hai nhóm GEO về cùng phân bố sáu tầng của toàn cohort.', '',
        '| Horizon | Nền toàn cohort | RR có/không GEO thô | RR chuẩn hóa TRX | CI 95% RR chuẩn hóa |', '|---|---:|---:|---:|---|']
    for r in s.itertuples():
        lines.append(f'| {r.horizon} tháng | {d[f"y_{r.horizon}"].mean():.2%} | {r.crude_rr:.3f} | {r.standardized_rr:.3f} | {r.standardized_rr_low:.3f}–{r.standardized_rr_high:.3f} |')
    lines += ['', 'RR ở đây là tỷ số tỷ lệ mua của nhóm có GEO so với không GEO, khác với lift so với nền toàn cohort. CI của RR chuẩn hóa dùng xấp xỉ delta trên log RR với trọng số tầng cố định; không phải Wilson. Tất cả sáu tầng đều có cả hai nhóm nên giữ 100% trọng số cohort.',
        'Chênh lệch giảm sau điều chỉnh nhưng còn tồn tại. Đây là liên hệ quan sát, chưa phải tác động nhân quả của GEO.', '',
        '![Có GEO trong từng tầng TRX](has_geo_stratified.png)','',
        '## 5. Lift của nhóm chỉ báo trong tầng TRX','',
        'Ba chỉ báo: số sự kiện GEO toàn lịch sử, số ô g4 khác nhau, tỷ lệ sự kiện tại ô g4 chính. Trong mỗi tầng TRX, chia quartile chỉ trên khách có GEO; giữ ties và bỏ ranh giới trùng. Q1–Q4 vì vậy có ngưỡng khác nhau giữa các tầng. Nhóm không GEO giữ riêng.',
        'Lift chính = tỷ lệ mua của quartile / tỷ lệ mua của tất cả khách có GEO trong cùng tầng TRX. CSV còn có lift so với toàn bộ khách trong tầng, số khách, số người mua và Wilson 95%.','',
        'Ở H=3, Q4 số sự kiện GEO có tỷ lệ mua cao hơn Q1 trong cả T1–T5; Q4 số ô g4 cũng cao hơn Q1 trong cả T1–T5. Tỷ lệ ô chính không có chiều nhất quán (Q4 cao hơn Q1 ở 1/5 tầng TRX dương). Đây là so sánh mô tả nhiều nhóm, chưa phải kết luận ý nghĩa thống kê từng ô heatmap. Không chọn nhóm/horizon chỉ dựa trên hình.', '',
        '![Lift từng nhóm chỉ báo](feature_lift_stratified.png)','',
        '## Giới hạn và kiểm chứng','',
        '- Một cutoff duy nhất; chưa chứng minh ổn định qua nhiều kỳ hoặc cải thiện dự báo ngoài thời gian.',
        '- Nguồn GEO và TRX có ngày bắt đầu khác nhau; phân tầng số log chưa kiểm soát độ dài lịch sử/tuổi quan hệ ngân hàng, thiếu dữ liệu và các yếu tố khách hàng khác.',
        '- Ô là mã ẩn danh; không diễn giải thành địa điểm thật, vùng trung tâm hay thu nhập.',
        '- Tổng GEO/TRX trước cutoff đối chiếu trực tiếp raw: 55.335.697 / 35.288.744. Kiểm tra cohort duy nhất, nhãn tích lũy, số khách/người mua, Wilson, co giữa tỷ lệ thô và cha, và phân hoạch tầng đều đạt.',
        '- Không huấn luyện mô hình, không chọn horizon, không báo top-K từ bảng tỷ lệ dùng cùng nhãn.', '',
        '## Tệp kết quả','',
        '- `visitor_g4_g5_wilson.csv`: tỷ lệ mua mọi ô g4/g5 và CI, đủ H=1/2/3.',
        '- `visitor_hierarchical_cell_rates.parquet`: tỷ lệ thô và làm trơn g4/g5/g6, m=20/100/500.',
        '- `global_cell_permutation_all_definitions.csv`: kiểm định mọi ô/ô chính, p Holm chung.',
        '- `geo_direct_standardization.csv`, `has_geo_by_trx_stratum.csv`: RR và tỷ lệ mua phân tầng.',
        '- `feature_lift_by_trx_stratum.csv`: lift nhóm chỉ báo, đủ H=1/2/3.',
        '- `edges_*.csv`, `trx_stratum_edges.csv`: ngưỡng chia tầng và nhóm.',
        '- `customer_snapshot.parquet`, `visitor_memberships.parquet`: snapshot và quan hệ khách–ô đã kiểm soát cutoff.',
        '- `manifest.json`, `validation.json`: giả định, quy trình và kiểm tra.', '',
        'Tái lập theo thứ tự: `src/scripts/analyze_geo_spatial_purchase_moscow.py`, `src/scripts/analyze_geo_visitor_cell_rates.py`, `src/scripts/report_geo_spatial_purchase_moscow.py` với Python trong `.venv`.', '',
        'Phương pháp tham khảo: [Wilson trong statsmodels](https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.proportion_confint.html); [nguyên tắc kiểm định hoán vị và p Monte Carlo](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html).']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    manifest.update(visitor_definition='Each customer counted once within every historical cell visited; overlapping membership across cells',
        visitor_permutations=499,visitor_shuffle_unit='Customer label; all cell memberships retain same permuted label',
        holm_family_size=18,authoritative_global_test_file='global_cell_permutation_all_definitions.csv',
        standardized_rr_ci='Fixed-weight delta method on log risk ratio; distinct from pointwise Wilson rate CI')
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('REPORT.md and all-visitor forest plot generated')


if __name__=='__main__':main()
