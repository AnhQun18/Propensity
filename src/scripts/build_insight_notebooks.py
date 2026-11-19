"""Build and execute three focused notebooks, preserving inline chart outputs."""
from pathlib import Path
import os
# Windows managed workspace cannot change the DACL of local kernel files.
os.environ.setdefault('JUPYTER_ALLOW_INSECURE_WRITES','1')
import argparse
import hashlib
import json
import sys
import nbformat as nb
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

ROOT = Path(__file__).resolve().parents[2]
SETUP = '''from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from IPython.display import display
ROOT = Path.cwd()
if not (ROOT / 'data/MBD-mini').exists():
    ROOT = ROOT.parent
assert (ROOT / 'data/MBD-mini').exists(), 'Mở notebook trong thư mục dự án'
AUDIT = ROOT / 'outputs/eda_insights_20261119'
TARGET = ROOT / 'outputs/target_focused_moscow_20261119'
FIG = ROOT / 'outputs/eda_insights_20261119/charts'
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans', 'figure.dpi':120,
    'axes.spines.top':False, 'axes.spines.right':False, 'axes.titlesize':13})
def save_chart(fig, name):
    fig.tight_layout()
    for ext in ('png','svg'):
        fig.savefig(FIG / f'{name}.{ext}', bbox_inches='tight')
    plt.show()
def read_audit(name):
    return pd.read_csv(AUDIT / name)
'''

def md(s): return nb.v4.new_markdown_cell(s)
def code(s): return nb.v4.new_code_cell(s)
def notebook(name, cells):
    book=nb.v4.new_notebook(cells=cells)
    book.metadata.update(kernelspec={'name':'python3','display_name':'Python (.venv)','language':'python'},
        language_info={'name':'python','version':sys.version.split()[0]},
        analysis_protocol={'timezone':'Europe/Moscow','history':'all events before cutoff',
            'population':'all 100224 clients','horizon_cohort':'10 cutoffs, February-November 2022'})
    p=ROOT/'notebooks'/name
    nb.write(book,p)
    return p

def build():
    paths=[]
    audit_source=(ROOT/'src/scripts/audit_insights_checklist.py').read_text(encoding='utf-8')
    audit_source=audit_source.replace('ROOT = Path(__file__).resolve().parents[2]', '# ROOT được xác định ở cell thiết lập notebook')
    p=notebook('18_join_quality_audit.ipynb',[
        md('''# 18 · Nối bảng và kiểm tra chất lượng

**Câu hỏi:** phép nối có giữ đúng tổng thể khách–tháng và lịch sử trước cutoff không?

Notebook quét toàn bộ raw TRX/GEO/DIALOG, kiểm tra khóa, orphan và nhãn; tính lại count theo tháng Moscow và đối chiếu cache/panel. Không lấy mẫu. Cell audit có thể mất vài phút và cần DuckDB; dừng nếu thiếu file hoặc đối soát sai.

Đơn vị mẫu: `(client_id, tháng nhãn)`. Cutoff 00:00 Moscow ngày đầu tháng; raw được diễn giải theo quy ước UTC rồi cộng 3 giờ để phân tháng. Quy ước này chưa xác minh timezone ETL. `mon` là định danh tháng nên không dịch timezone.

`client_split` (1/khách) → `targets` (1/khách–tháng) ← từng nguồn **đã tổng hợp** (1/khách–cutoff). LEFT JOIN giữ cả khách không có log; không nối hai bảng sự kiện chỉ theo client_id vì sẽ nhân dòng.'''),
        code(SETUP),
        md('## Audit từ raw và dựng lại thẻ phân khúc\nCell dưới là mã audit được nhúng từ script, không chỉ đọc bảng kết quả cũ. Nó kiểm tra 12 tháng nhãn và dùng 10 cutoff đủ tương lai để so H1/H2/H3. H2/H3 là có mua trong ít nhất một trong 2/3 tháng kể từ tháng đích. Feature dùng mọi tháng trước cutoff.'),
        code(audit_source),
        md('## Kết quả kiểm tra\nMỗi dòng kiểm tra phải đạt. Count cache khớp raw chưa chứng minh ETL gốc đầy đủ; nó xác nhận tính nhất quán giữa các artifact hiện có.'),
        code("checks=json.loads((AUDIT/'validation.json').read_text())\ndisplay(pd.DataFrame({'check':checks.keys(),'passed':checks.values()}))\nassert all(checks.values())\njoin=read_audit('join_audit.csv')\ndisplay(join)"),
        code('''fig, axes=plt.subplots(1,2,figsize=(11,4))
colors=['#2563eb','#0d9488','#d97706']
for ax,col,title,scale in [(axes[0],'rows','Số sự kiện raw (triệu)',1e6),(axes[1],'customers','Khách có log toàn kỳ (nghìn)',1e3)]:
    vals=join[col]/scale
    ax.bar(join.source, vals, color=colors)
    for i,v in enumerate(vals): ax.text(i,v,f'{v:,.2f}',ha='center',va='bottom')
    ax.set_title(title); ax.set_ylim(0,max(vals)*1.18)
save_chart(fig,'18_source_inventory')'''),
        md('**Diễn giải:** độ phủ khách khác nhau giữa các nguồn; inner join cả ba sẽ làm đổi tổng thể. Số khách có log toàn kỳ không phải số khách có lịch sử tại mọi cutoff. 0 count là không ghi nhận log, không chứng minh không hoạt động.'),
        md('## Checklist join\n- [x] Khóa khách và khách–tháng duy nhất; đủ 12 tháng nhãn.\n- [x] Null khóa/thời gian, orphan, count theo tháng và nhãn được đối soát.\n- [x] Giữ khách thiếu nguồn; phân khúc cộng đúng tổng thể.\n- [ ] Xác minh timezone ETL và thời điểm nhãn sẵn có.\n- [ ] Kiểm tra near-duplicate và mapping detail–PTLS nếu dùng PTLS.\n\nXem notebook 19 cho đơn biến và notebook 20 cho quan hệ với target.')])
    b=nb.read(p,4); b.metadata['embedded_audit_sha256']=hashlib.sha256(audit_source.encode()).hexdigest(); nb.write(b,p); paths.append(p)
    paths.append(notebook('19_univariate_distributions.ipynb',[
        md('''# 19 · Phân tích đơn biến: nhãn, amount và độ phủ

**Câu hỏi:** phân phối nào ảnh hưởng đến xử lý dữ liệu? Đọc kết quả raw audit từ notebook 18; không dùng mẫu 1.000 khách. Biểu đồ lưu PNG và SVG trong `outputs/eda_insights_20261119/charts/`.'''),code(SETUP),
        code("assert all(json.loads((AUDIT/'validation.json').read_text()).values())\nlabels=read_audit('label_monthly.csv')\nlabels['month']=pd.to_datetime(labels.month)\namount=read_audit('amount_by_currency.csv')\ncards=read_audit('segment_cards.csv')\ndisplay(labels)\ndisplay(amount)"),
        md('## Nhãn qua đủ 12 tháng\nTỷ lệ dương dùng số khách–tháng có nhãn làm mẫu số. Biến động nền ảnh hưởng diễn giải lift và hiệu chỉnh xác suất.'),
        code('''fig,ax=plt.subplots(figsize=(11,4))
ax.plot(labels.month,labels.rate,marker='o',color='#2563eb')
pooled=labels.buyers.sum()/labels.n.sum()
ax.axhline(pooled,ls='--',color='#d97706',label=f'Toàn kỳ: {pooled:.2%}')
for r in labels.itertuples(): ax.annotate(f'{r.buyers:,}',(r.month,r.rate),xytext=(0,7),textcoords='offset points',ha='center',fontsize=8)
ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.set_ylim(0,labels.rate.max()*1.25)
ax.set_title('Tỷ lệ y_any theo tháng · nhãn cạnh điểm = số khách–tháng dương')
ax.set_ylabel('Tỷ lệ mua'); ax.legend(); fig.autofmt_xdate()
save_chart(fig,'19_target_monthly')
print(f'{labels.buyers.sum():,}/{labels.n.sum():,} khách–tháng dương = {pooled:.4%}')'''),
        md('## Amount theo currency\nSo median, p99 và max trên thang log. Đây là tóm tắt phân vị, không phải histogram raw. Giữ riêng mã tiền tệ; chưa có đơn vị đã xác minh để diễn giải thành doanh thu.'),
        code('''a=amount.head(6).copy()
names=a.currency.map(lambda x: 'missing' if pd.isna(x) else str(int(x)))
fig,ax=plt.subplots(figsize=(10,4.5)); x=np.arange(len(a))
for shift,col,label,color in [(-.24,'median','Median','#0d9488'),(0,'p99','P99','#2563eb'),(.24,'maximum','Max','#d97706')]:
    ax.bar(x+shift,a[col],width=.24,label=label,color=color)
ax.set_yscale('log'); ax.set_xticks(x,names); ax.set_xlabel('Mã currency · 6 mã nhiều dòng nhất')
ax.set_ylabel('Amount · đơn vị dữ liệu · log scale'); ax.set_title('Đuôi amount theo từng currency'); ax.legend()
save_chart(fig,'19_amount_quantiles')
display(amount[['currency','n','missing_amount','zero_amount','negative_amount','minimum','median','p99','maximum']])'''),
        md('**Diễn giải:** max rất xa median nên trung bình đơn lẻ dễ bị chi phối. Không tự xem điểm cực trị là lỗi. Ngưỡng biến đổi/cap nếu dùng cho mô hình phải được học từ phần train.'),
        md('## Quy mô phân khúc feature\nCount là tổng toàn bộ log trước cutoff. Mỗi khách có thể xuất hiện ở nhiều cutoff; không diễn giải tổng này là số khách duy nhất.'),
        code('''fig,axes=plt.subplots(1,3,figsize=(13,4))
for ax,feature,title in zip(axes,['trx_band','dialog_recent','source_count'],['Count TRX toàn lịch sử','DIALOG tháng trước','Số nguồn có lịch sử']):
    a=cards[(cards.feature==feature)&(cards.horizon==1)]
    ax.bar(a.group.astype(str),a.population_share,color='#2563eb')
    ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.set_title(title)
    ax.set_ylim(0,a.population_share.max()*1.18)
    for i,v in enumerate(a.population_share): ax.text(i,v,f'{v:.1%}',ha='center',va='bottom',fontsize=9)
save_chart(fig,'19_segment_sizes')'''),
        md('## Checklist đơn biến\n- [x] Nhãn theo 12 tháng; amount theo currency; quy mô nhóm count và masks.\n- [ ] Refresh missing/cardinality/mức hiếm cho toàn bộ feature.\n- [ ] Recency theo ngày, gap xuyên tháng, history age và active days.\n- [ ] So độ nhạy kết luận khi xử lý ngoại lai.\n\nBiểu đồ phân vị không thay thế histogram, và count phân nhóm không thay thế phân phối raw từng khách.')]))
    paths.append(notebook('20_multivariate_insights.ipynb',[
        md('''# 20 · Quan hệ với target, tương tác và độ ổn định

**Câu hỏi:** nhóm nào có tỷ lệ mua cao, đủ quy mô và ổn định qua tháng? Dùng 100.224 khách × 10 cutoff 02–11/2022, cùng cohort H1/H2/H3. Đọc các thẻ phân khúc đã dựng lại ở notebook 18. Hai bảng tương tác cuối đọc từ phân tích target-focused hiện có và được ghi rõ nguồn.

Tất cả là EDA mô tả. Lift = tỷ lệ nhóm / tỷ lệ nền cùng cohort hoặc cùng tháng. Các quan sát lặp khách, nên chưa coi là các dòng độc lập để tính CI hay p-value.'''),code(SETUP),
        code("cards=read_audit('segment_cards.csv')\nmonthly=read_audit('monthly_segment_cards.csv')\nmonthly['month']=pd.to_datetime(monthly.month)\nh1=cards[cards.horizon==1].copy()\nassert monthly.groupby(['feature','month','horizon']).n.sum().eq(100224).all()\ndisplay(h1)"),
        md('## Tỷ lệ mua theo feature\nQuan hệ TRX có thể phi tuyến. Nhóm không có log được giữ riêng; không đánh đồng với khách mới hoặc không hoạt động.'),
        code('''fig,axes=plt.subplots(1,3,figsize=(14,4.5))
baseline=h1[h1.feature=='trx_band'].buyers.sum()/h1[h1.feature=='trx_band'].n.sum()
for ax,feature,title in zip(axes,['trx_band','dialog_recent','source_count'],['TRX: quan hệ không đơn điệu','DIALOG tháng trước','Số nguồn có lịch sử']):
    a=h1[h1.feature==feature]
    ax.bar(a.group.astype(str),a.rate,color='#0d9488')
    ax.axhline(baseline,color='#d97706',ls='--',label=f'Nền {baseline:.2%}')
    for i,r in enumerate(a.itertuples()): ax.text(i,r.rate,f'{r.rate:.2%}',ha='center',va='bottom',fontsize=9)
    ax.set_ylim(0,a.rate.max()*1.25); ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.set_title(title); ax.legend(fontsize=8)
save_chart(fig,'20_feature_target_rates')'''),
        md('## Quy mô và đóng góp người mua\nTỷ trọng người mua là phần khách–tháng dương nằm trong nhóm; đây chưa phải recall trên dữ liệu kiểm chứng ngoài thời gian.'),
        code('''a=h1[h1.feature=='dialog_recent']; x=np.arange(len(a))
fig,ax=plt.subplots(figsize=(8,4))
ax.bar(x-.17,a.population_share,width=.34,label='Tỷ trọng quan sát',color='#94a3b8')
ax.bar(x+.17,a.buyer_share,width=.34,label='Tỷ trọng khách–tháng dương',color='#0d9488')
ax.set_xticks(x,['Không có DIALOG tháng trước','Có DIALOG tháng trước'])
ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.set_title('DIALOG gần: quy mô và đóng góp'); ax.legend()
save_chart(fig,'20_dialog_contribution')'''),
        md('## Lift theo tháng: tránh kết luận từ tỷ lệ gộp\nĐường nền 1× là tỷ lệ mua của toàn bộ khách trong cùng tháng. Kiểm tra xem nhóm có giữ cùng hướng quan hệ qua mọi cutoff không.'),
        code('''fig,axes=plt.subplots(1,2,figsize=(13,4))
for ax,feature,title in zip(axes,['dialog_recent','trx_band'],['DIALOG gần: kiểm tra độ bền','TRX: nhóm 0 log có thể đổi hướng']):
    a=monthly[(monthly.feature==feature)&(monthly.horizon==1)]
    for group,z in a.groupby('group'):
        ax.plot(z.month,z.lift,marker='o',label=str(group))
    ax.axhline(1,ls='--',color='#475569'); ax.set_title(title); ax.set_ylabel('Lift theo nền cùng tháng'); ax.legend(fontsize=8)
fig.autofmt_xdate(); save_chart(fig,'20_monthly_lift')
display(h1[['feature','group','min_monthly_lift','max_monthly_lift']])'''),
        md('**Insight:** DIALOG gần giữ lift cao ở mọi cutoff; nhóm không có TRX dao động từ dưới nền sang trên nền. Cần kiểm tra cơ cấu cohort và thời gian quan sát trước khi dùng nhóm 0 TRX làm quy tắc.'),
        md('## H1/H2/H3 trên cùng cohort\nHorizon dài hơn có nhiều cơ hội ghi nhận nhãn dương hơn. Tỷ lệ tăng chưa chứng minh khả năng xếp hạng tốt hơn.'),
        code('''a=cards[cards.feature=='dialog_recent']
fig,axes=plt.subplots(1,2,figsize=(11,4))
for group,z in a.groupby('group'):
    for ax,col in zip(axes,['rate','lift']): ax.plot(z.horizon,z[col],marker='o',label=f'DIALOG tháng trước: {group}')
for h in (1,2,3):
    z=a[a.horizon==h]; axes[0].scatter(h,z.buyers.sum()/z.n.sum(),color='#d97706',zorder=3)
axes[0].yaxis.set_major_formatter(PercentFormatter(1)); axes[0].set_title('Tỷ lệ mua · chấm cam = nền')
axes[1].set_title('Lift theo horizon'); axes[1].axhline(1,ls='--',color='#475569')
for ax in axes: ax.set_xticks([1,2,3]); ax.set_xlabel('Horizon (tháng)'); ax.legend(fontsize=8)
save_chart(fig,'20_horizon_comparison')'''),
        md('## Tương tác TRX × DIALOG\nNguồn tái sử dụng: `outputs/target_focused_moscow_20261119/dialog_by_trx_state.csv`. Các ô cộng đủ cohort; bảng phân tầng không loại trừ toàn bộ yếu tố nhiễu.'),
        code('''interaction=pd.read_csv(TARGET/'dialog_by_trx_state.csv')
assert interaction.n.sum()==100224*10
display(interaction[['trx_state','dialog_recent','n','buyers_h1','rate_h1']])
p=interaction.pivot(index='trx_state',columns='dialog_recent',values='rate_h1')
counts=interaction.pivot(index='trx_state',columns='dialog_recent',values='n')
fig,ax=plt.subplots(figsize=(8,4)); im=ax.imshow(p.to_numpy()*100,cmap='YlGnBu',aspect='auto')
for i in range(len(p)):
    for j in range(len(p.columns)):
        v=p.iloc[i,j]; ax.text(j,i,f'{v:.2%}\\nn={counts.iloc[i,j]:,}',ha='center',va='center',color='white' if v>.06 else '#172554')
ax.set_xticks(range(len(p.columns)),[f'DIALOG gần: {v}' for v in p.columns]); ax.set_yticks(range(len(p)),p.index)
ax.set_title('Tỷ lệ mua H1 trong từng trạng thái TRX'); fig.colorbar(im,ax=ax,label='Tỷ lệ mua (%)')
save_chart(fig,'20_trx_dialog_interaction')'''),
        md('## Lịch sử mua quan sát được\nNguồn tái sử dụng: `dialog_by_prior_purchase.csv`. “Chưa có mua quan sát được” chỉ nói về phần nhãn từ 02/2022; không khẳng định chưa từng mua. Prior-purchase feature chỉ được dùng khi nhãn quá khứ thực sự sẵn có trước cutoff.'),
        code('''prior=pd.read_csv(TARGET/'dialog_by_prior_purchase.csv')
assert prior.n.sum()==100224*10
display(prior[['prior_purchase_state','dialog_recent','n','buyers_h1','rate_h1']])
p=prior.pivot(index='prior_purchase_state',columns='dialog_recent',values='rate_h1')
fig,ax=plt.subplots(figsize=(10,4)); x=np.arange(len(p))
for j,col in enumerate(p.columns): ax.bar(x+(j-.5)*.35,p[col],width=.35,label=f'DIALOG gần: {col}')
ax.set_xticks(x,['Chưa có tháng mua quan sát','Đã có tháng mua quan sát','Trước kỳ nhãn đầu: chưa biết'])
ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.set_title('DIALOG trong từng trạng thái lịch sử nhãn'); ax.legend()
save_chart(fig,'20_prior_purchase_interaction')'''),
        md('''## Kết luận và checklist tiếp theo

Ưu tiên kiểm tra DIALOG recency/presence, TRX phi tuyến, masks đa nguồn và tương tác DIALOG×TRX. Nhóm không TRX nhưng DIALOG gần cần đối chiếu cơ chế ghi nhận dữ liệu.

- [x] So nhóm và horizon với mẫu số nhất quán; báo quy mô và đóng góp.
- [x] Kiểm tra lift theo 10 tháng, không chỉ tỷ lệ gộp.
- [ ] Bootstrap theo khách để báo CI, giữ nguyên chuỗi tháng của mỗi khách.
- [ ] Kiểm soát đồng thời history age, prior purchase và coverage; kiểm tra Simpson's paradox.
- [ ] Fit ngưỡng trên tháng train; kiểm chứng ngoài thời gian với cùng top-K.
- [ ] Xác minh độ trễ nhãn, drift và giá trị bổ sung từng nguồn bằng ablation.

Chưa có bằng chứng nhân quả, cải thiện mô hình hoặc hiệu quả chiến dịch từ các biểu đồ này. [Sổ insight và checklist đầy đủ](../docs/eda_insights_checklist_20261119.md).''')]))
    return paths

def execute(paths):
    run=ROOT/'outputs/eda_insights_20261119/notebook_execution'
    kernel=run/'kernels/insights-local'; kernel.mkdir(parents=True,exist_ok=True)
    runtime=run/'runtime'; runtime.mkdir(exist_ok=True)
    os.environ['IPYTHONDIR']=str(run/'ipython')
    os.environ['JUPYTER_RUNTIME_DIR']=str(runtime)
    (kernel/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-B','-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'Insights local','language':'python'}),encoding='utf-8')
    results=[]
    for p in paths:
        print('Execute',p.name,flush=True)
        book=nb.read(p,4)
        km=KernelManager(kernel_name='insights-local',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(run/'kernels')]),connection_file=str(runtime/f'{p.stem}.json'))
        def start(cell,cell_index):
            if cell.cell_type=='code': print(f'  cell {cell_index}: {cell.source.splitlines()[0][:70]}',flush=True)
        client=NotebookClient(book,km=km,timeout=1200,startup_timeout=120,resources={'metadata':{'path':str(ROOT)}},on_cell_start=start)
        try:
            client.execute()
            book.metadata['execution_status']='complete'
        finally:
            nb.write(book,p)
            if km.has_kernel: km.shutdown_kernel(now=True)
        errors=[o for cell in book.cells if cell.cell_type=='code' for o in cell.outputs if o.output_type=='error']
        images=sum('image/png' in o.get('data',{}) for cell in book.cells if cell.cell_type=='code' for o in cell.outputs)
        assert not errors
        results.append({'notebook':p.name,'code_cells':sum(c.cell_type=='code' for c in book.cells),'inline_charts':images,'errors':len(errors)})
    (run/'validation.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps(results,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--execute',action='store_true'); args=parser.parse_args()
    paths=build()
    if args.execute: execute(paths)
