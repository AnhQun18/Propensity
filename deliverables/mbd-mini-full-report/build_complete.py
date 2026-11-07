"""Full analytical report: every manifest artifact has a source and page mapping."""
from pathlib import Path
import json, hashlib, re
import pandas as pd
BASE=Path(__file__).resolve().parent/'build_base.py'
prefix=BASE.read_text(encoding='utf-8').split('# 27. Conclusions and evidence')[0]
prefix=prefix.replace('MBD_mini_Bao_cao_phan_tich_toan_lich_su.pdf','MBD_mini_Bao_cao_day_du.pdf')
exec(compile(prefix,str(BASE),'exec'),globals())
DOC=ROOT/'docs/mbd_mini_eda'
MANIFEST=json.loads((DOC/'manifest.json').read_text(encoding='utf-8'))
COMMIT='3e4b95d79d172d2ef525ff3d26e2528bce290e8e'
PUBLIC=f'https://github.com/AnhQun18/Propensity/blob/{COMMIT}/docs/mbd_mini_eda/'
RAW=f'https://raw.githubusercontent.com/AnhQun18/Propensity/{COMMIT}/docs/mbd_mini_eda/'
LEDGER={t['section']+'/'+t['name']:{**t,'pages':[],'representation':''} for t in MANIFEST['tables']}
def load(key):return pd.read_csv(DOC/LEDGER[key]['path'])
def mark(keys,pages,mode='Bảng đầy đủ'):
    for key in keys:
        LEDGER[key]['pages']=sorted(set(LEDGER[key]['pages']+list(pages)))
        LEDGER[key]['representation']=mode
def fmt(v):
    if pd.isna(v):return 'null'
    if isinstance(v,(float,np.floating)) and np.isinf(v):return '+∞' if v>0 else '−∞'
    if isinstance(v,(int,np.integer)):return f(v)
    if isinstance(v,(float,np.floating)):
        if v==0:return '0'
        if abs(v)<.0001:return f(v,10)
        return f(v,3 if abs(v)<1 else 2).rstrip('0').rstrip(',') if v!=int(v) else f(v)
    return str(v)
def source_links(keys):
    for j,key in enumerate(keys[:3]):
        label='CSV đầy đủ: '+key
        text(M,64-j*12,label,7.7,'Arial',TEAL)
        C.linkURL(RAW+LEDGER[key]['path'],(M,61-j*12,W-M,73-j*12),relative=0)
def grid(headers,rows,y,widths,font=8.7,draw=True):
    ps=ParagraphStyle('compact',fontName='Arial',fontSize=font,leading=font*1.27,textColor=INK)
    hs=ParagraphStyle('compact_h',parent=ps,fontName='Arial-Bold',textColor=NAVY)
    tb=Table([[Paragraph(escape(str(v)),hs) for v in headers]]+
        [[Paragraph(escape(str(v)),ps) for v in r] for r in rows],colWidths=widths,repeatRows=1)
    tb.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),LIGHT),('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),
        ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),
        ('LINEBELOW',(0,0),(-1,-1),.35,LINE)]))
    _,h=tb.wrap(CW,H)
    if draw:tb.drawOn(C,M,y-h)
    return h,tb
def pages_table(title,headers,rows,keys,explain='',widths=None,font=8.7,mode='Bảng đầy đủ',links=None):
    rows=list(rows); widths=widths or [CW/len(headers)]*len(headers)
    start=PAGE+1;offset=0;part=0
    while offset<len(rows):
        part+=1;y=page(title+(' · tiếp' if part>1 else ''),'PHỤ LỤC PHÂN TÍCH ĐẦY ĐỦ')
        if explain:y=para(explain,y,size=10.2,leading=14)-12
        take=0
        while offset+take<len(rows):
            h,_=grid(headers,rows[offset:offset+take+1],y,widths,font,False)
            if y-h<96:break
            take+=1
        if not take:raise ValueError((title,'row overflow',rows[offset]))
        h,tb=grid(headers,rows[offset:offset+take],y,widths,font)
        if links:
            cursor=y
            for i,rh in enumerate(tb._rowHeights):
                if i:C.linkURL(links[offset+i-1],(M,cursor-rh,W-M,cursor),relative=0)
                cursor-=rh
        source_links(keys);offset+=take
    out=list(range(start,PAGE+1));mark(keys,out,mode);return out

# Existing narrative pages use native, linear renderings of the same evidence.
existing={
 'review/source_schema':[3,4,5], 'review/missing_values':[6], 'review/null_target_context':[7],
 'review/null_pairs':[7], 'review/duplicates':[8], 'review/sequence_reconciliation':[9],
 'review/embedding_element_quality':[9], 'review/behavior_rates':[10],
 'review/behavior_combinations':[10], 'review/monthly_coverage':[11],
 'review/purposeful_coverage':list(range(12,17)), 'review/events_per_customer':[17],
 'review/daily_behavior':[18], 'review/target_monthly':[21,22], 'review/sample_inventory':[23],
 'review/sampling_strata':[23], 'review/sampling_validation':[24], 'review/sampling_labels':[24],
 'review/customer_count_outliers':[25], 'review/daily_count_outliers':[25],
 'review/amount_outliers_currency':[26]}
for k,p in existing.items():mark([k],[n if n<19 else n-2 for n in p],'Tóm tắt / biểu diễn trực tiếp')

y=page('Hướng dẫn tra cứu báo cáo đầy đủ','PHẠM VI VÀ PHƯƠNG PHÁP')
for title,body in [
 ('Phần chính · trang 1–24','Các kết luận, ma trận minh họa có chủ đích và biểu đồ được vẽ lại từ dữ liệu toàn lịch sử. Đã bỏ các biểu đồ phân phối Amount theo yêu cầu; các thống kê số gốc được giữ trong phụ lục để tra cứu, chưa diễn giải thành giá trị tiền đã quy đổi.'),
 ('Phụ lục phân tích · từ trang 26','Bổ sung toàn bộ chủ đề trong 61 bảng và 24 hình của manifest: độ phủ dân số, phân phối theo khách, thời gian, tần suất mã, nhãn, chất lượng dữ liệu, mẫu và ngoại lai.'),
 ('Dữ liệu chi tiết và tính đầy đủ','Các bảng dài được trình bày bằng ma trận, thống kê hoặc nhóm mã nổi bật. Mỗi bảng đều có liên kết CSV đầy đủ và trang đối chiếu. Báo cáo không in tuần tự 752.805 dòng tần suất vị trí hoặc 12.829 dòng tần suất giao dịch.'),
 ('Quy ước diễn giải','Số liệu trình bày được làm tròn; CSV giữ độ chính xác đã xuất. null ở thuộc tính, null do thiếu quan sát và ô không áp dụng trong bảng thống kê có ý nghĩa khác nhau. Mỗi phần ghi rõ mẫu số và đơn vị.')]:
    y=section_title(title,y);y=para(body,y,size=11,leading=16)-18
source('Manifest: docs/mbd_mini_eda/manifest.json. Phiên bản nguồn được cố định tại commit 3e4b95d để bảo đảm khả năng đối chiếu.')

# Population group matrix, 175 cells, all customers.
k='review/group_monthly_coverage';d=load(k)
groups=d[['activity_group','population','min_total','max_total']].drop_duplicates()
gp=pages_table('Độ phủ dân số theo mức độ hoạt động',['Nhóm','Số khách','Min tổng log','Max tổng log'],
 [[r.activity_group,f(r.population),f(r.min_total),f(r.max_total)] for r in groups.itertuples()],[k],
 'B0: không có log; B1–B6 chia theo P25/P50/P75/P95/P99 của khách có log. Biên: 201; 651; 1.428; 3.362; 5.664,8 log. Tổng ba nguồn được tính trên toàn lịch sử.',
 widths=[90,160,230,CW-480])
for a,b in [(0,13),(13,25)]:
    y=page('Ma trận độ phủ của toàn bộ nhóm dân số','ĐỘ PHỦ DÂN SỐ')
    y=para('Mỗi ô là % khách có ít nhất một log bất kỳ trong tháng, mẫu số là số khách của chính nhóm. Màu dùng thang tuyến tính chung 0–100%. Đây là độ phủ log quan sát được, chưa thay bằng null chính sách nguồn sớm.',y,size=10.5,leading=15)-14
    p=d.pivot(index='activity_group',columns='month',values='coverage_percent').sort_index();p=p.iloc[:,a:b]
    matrix([str(x)[2:] for x in p.columns],p.values,y,list(p.index),100)
    note('Không dùng 40 khách minh họa để suy rộng. B0 gồm 468 khách không có log trong toàn kỳ.',180)
    source_links([k]);mark([k],[PAGE],'Toàn bộ 175 ô + quy mô và biên nhóm')

# Customer distributions; split columns to keep precision legible.
k='descriptive.customers/customer_activity';d=load(k)
for cols,title in [(['rows','finite_n','missing_n','infinite_n','mean','std'],'Quần thể và trung bình của 20 biến hành vi'),
 (['min','p25','median','p75','p95','p99','max'],'Phân vị và cực trị của 20 biến hành vi')]:
    pages_table(title,['Biến']+cols,[[r['feature']]+[fmt(r[c]) for c in cols] for _,r in d.iterrows()],[k],
      'Mỗi biến tổng hợp theo khách; rows = 100.224. *_span_days: từ log đầu đến cuối, đơn vị ngày; *_mean_gap_seconds: khoảng cách trung bình, giây; geo_unique_locations: số mã geohash_6 phân biệt. Mean và phân vị chỉ tính trên giá trị hữu hạn.',
      widths=[190]+[(CW-190)/len(cols)]*len(cols),font=8.1)
pages_table('Null trong các biến tổng hợp theo khách',['Trường hợp','Ý nghĩa'],[
 ['event_count, active_days, unique_locations','0 nghĩa là không có log quan sát được; giữ cả khách 0 log.'],
 ['span_days','null khi khách không có log ở nguồn. Có đúng một log thì span bằng 0.'],
 ['mean_gap_seconds','null khi không đủ hai log để tính khoảng cách; không tự thay bằng 0.'],
 ['trx_amount_mean','null khi khách không có Amount hữu hạn; amount_sum và amount_n giữ 0 nếu không có giao dịch.'],
 ['has_trx / has_geo / has_dialog','Cờ 0/1 cho sự hiện diện ít nhất một log trong toàn lịch sử.'],
 ['Bảng thống kê dạng rộng','Một số ô null ở cột n hoặc value của embedding_quality là cột không áp dụng cho loại phép kiểm tra đó, không phải vector bị thiếu.']],[k,'descriptive.modalities/embedding_quality'],
 'Phân biệt null của dữ liệu gốc với null do không đủ thông tin để tính một thống kê.',widths=[210,CW-210],font=10)
for key,title in [('review/events_per_customer','Số sự kiện mỗi khách · thống kê đầy đủ'),
 ('descriptive.temporal/history_length_summary','Độ dài lịch sử của khách có log')]:
    d=load(key);idcols=['source','population'] if 'population' in d else ['source','metric']
    for cols in [['rows','finite_n','missing_n','infinite_n','mean','std'],['min','p25','median','p75','p95','p99','max']]:
        pages_table(title,['Nguồn / quần thể']+cols,
         [[' / '.join(str(r[c]) for c in idcols)]+[fmt(r[c]) for c in cols] for _,r in d.iterrows()],[key],
         'all_customers giữ cả khách không có log; customers_with_log chỉ giữ khách có log. span_days là số ngày giữa sự kiện đầu và cuối; active_days là số ngày có ít nhất một sự kiện.',
         widths=[195]+[(CW-195)/len(cols)]*len(cols),font=8.2)
for key,title in [('descriptive.modalities/geo_summary','Tổng hợp vị trí theo khách'),('descriptive.modalities/dialog_summary','Tổng hợp hội thoại theo khách')]:
    d=load(key);cols=['rows','mean','std','min','p25','median','p75','p95','p99','max']
    pages_table(title,['Biến']+cols,[[r['metric']]+[fmt(r[c]) for c in cols] for _,r in d.iterrows()],[key],
      'Thống kê trên toàn bộ 100.224 khách, bao gồm khách 0 log. geo unique_locations là số vị trí phân biệt; các phân vị giữ nguyên định nghĩa ở CSV.',widths=[125]+[(CW-125)/len(cols)]*len(cols),font=8.2)

# Temporal full rows (monthly activity is not imputed).
k='descriptive.temporal/monthly_activity';d=load(k)
pages_table('Hoạt động theo tháng · khối lượng và khách',['Nguồn','Tháng','Số log','Khách có log','Ngày quan sát','Log / khách có log'],
 [[r.source,r.month,f(r.events),f(r.customers),f(r.observed_days),f(r.events_per_active_customer,3)] for r in d.itertuples()],[k],
 '61 nhóm nguồn–tháng có log. observed_days là số ngày phân biệt có log trong tháng; tháng biên có thể không đầy đủ. Không cộng số khách qua nhiều tháng để suy ra khách duy nhất.',widths=[104,82,126,133,130,CW-575])
k='descriptive.customers/first_last_event_month';d=load(k)
p=d.pivot(index=['source','month'],columns='milestone',values='customers').fillna(0).reset_index()
pages_table('Khách bắt đầu và kết thúc lịch sử theo tháng',['Nguồn','Tháng','Log đầu trong tháng','Log cuối trong tháng','Mẫu số khách có log'],
 [[r.source,r.month,f(r.first),f(r.last),f(d.loc[d.source.eq(r.source),'logged_customers'].iloc[0])] for r in p.itertuples()],[k,'descriptive.customers/first_last_event_coverage'],
 'Chỉ xác định mốc từ log quan sát được. Vắng hàng nguồn–mốc–tháng trong CSV được biểu diễn 0 trong ma trận này. Không kết luận khách rời bỏ hay mới tham gia chỉ từ tháng log cuối/đầu.',widths=[120,84,180,180,CW-564],mode='Toàn bộ số khách theo mốc; tỷ lệ từ mẫu số + CSV')

# Categorical distributions: every column summarized, top codes explicitly selected.
catkeys=['descriptive.transactions/transaction_categories_full','descriptive.modalities/location_categories_full']
for k in catkeys:
    d=load(k);summary=[];tops=[]
    for col,g in d.groupby('column',sort=False):
        v=g[g.kind.eq('value')].sort_values('n',ascending=False);other=g[~g.kind.eq('value')]
        summary.append([col,f(g.rows.iloc[0]),f(g.distinct_n.iloc[0]),f(v['n'].sum()),f(other['n'].sum()),pct(v.head(5)['n'].sum()/g.rows.iloc[0]*100,3)])
        for r in v.head(5).itertuples():tops.append([col,fmt(r.value),f(r.n),pct(r.rate*100,4)])
    pages_table('Phân phối mã · '+('giao dịch' if 'transactions' in k else 'vị trí'),
      ['Cột mã','Dòng nguồn','Số mã khác null','Log có mã','Log null','Tỷ trọng 5 mã đầu'],summary,[k],
      'Mỗi dòng sự kiện có trọng số bằng nhau. Mã được nhà phát hành ẩn danh; tần suất không cho biết ý nghĩa nghiệp vụ hoặc vị trí địa lý thật.',
      widths=[140,127,125,125,108,CW-625],mode='Tất cả cột + top 5 mỗi cột; toàn bộ mã trong CSV')
    pages_table('Các mã xuất hiện nhiều nhất · '+('giao dịch' if 'transactions' in k else 'vị trí'),
      ['Cột','Mã','Số log','Tỷ lệ trên toàn nguồn'],tops,[k],
      'Trình bày tối đa 5 mã khác null mỗi cột. Tệp CSV chứa mọi mã, gồm đuôi phân phối và nhóm null. Cột rate dùng tổng số dòng nguồn làm mẫu số.',widths=[220,140,180,CW-540],mode='Tất cả cột + top 5 mỗi cột; toàn bộ mã trong CSV')
    del d

# Labels: prevalence, history, concurrent positives and monthly/fold grid.
k='descriptive.targets/target_prevalence';d=load(k)
pages_table('Nhãn dương và âm · mẫu số đầy đủ',['Nhãn','Tổng dòng','Có nhãn','Dương','Âm','Null','Tỷ lệ dương'],
 [[r.target,f(r.rows),f(r.labelled_n),f(r.positive_n),f(r.negative_n),f(r.missing_n),pct(r.rate*100,6)] for r in d.itertuples()],[k],
 'Nhãn ở cấp khách–tháng; mỗi nhãn có 1.202.688 dòng hợp lệ. Tỷ lệ dương = positive_n / labelled_n. Không đồng nhất tỷ lệ dòng dương với tỷ lệ khách từng dương.',widths=[95,119,119,95,119,65,CW-612])
k='descriptive.targets/customer_positive_history';d=load(k)
pages_table('Khách từng dương và số tháng dương',['Nhãn','Khách','Khách quan sát','Khách từng dương','Tỷ lệ từng dương','Mean số tháng dương'],
 [[r.target,f(r.customers),f(r.observed_customers),f(r.ever_positive_customers),pct(r.rate*100,4),f(r.mean_positive_months,6)] for r in d.itertuples()],[k],
 'Mean số tháng dương dùng toàn bộ khách quan sát làm mẫu số, gồm người không có tháng dương. Không diễn giải đây là số tháng dương trung bình chỉ trong khách từng dương.',widths=[90,95,120,150,145,CW-600])
k='descriptive.targets/target_cooccurrence';d=load(k)
pages_table('Đồng xuất hiện nhiều nhãn dương',['Nhãn trái','Nhãn phải','Cặp quan sát','Cùng dương','Tỷ lệ cùng dương'],
 [[r.left,r.right,f(r.observed_pairs),f(r.both_positive),pct(r.both_positive/r.observed_pairs*100,6)] for r in d.itertuples()],[k],
 'Cùng dương nghĩa là hai nhãn bằng 1 trong cùng dòng khách–tháng. Mẫu số là số dòng quan sát được cả hai nhãn. Bốn nhãn không loại trừ lẫn nhau.',widths=[130,130,170,150,CW-580])
k='descriptive.targets/positive_labels_per_row';d=load(k)
pages_table('Số nhãn dương trên cùng dòng',['Số nhãn dương','Số dòng khách–tháng','Tỷ lệ'],
 [[f(r.positive_labels),f(r.rows),pct(r.rows/d.rows.sum()*100,6)] for r in d.itertuples()],[k],
 'Mỗi dòng có bốn nhãn nhị phân. Tổng số dòng bằng 1.202.688; số nhãn dương là tổng target_1 đến target_4.',widths=[200,300,CW-500])
k='descriptive.targets/target_by_month_fold';d=load(k);coverage=load('descriptive.targets/label_coverage')
p=d.pivot(index=['mon','fold'],columns='target',values='positive_n').reset_index().merge(coverage,on=['mon','fold'])
pages_table('Nhãn theo tháng và fold · toàn bộ 60 nhóm',['Tháng','Fold','Dòng / khách','T1 dương','T2 dương','T3 dương','T4 dương','Ô nhãn null'],
 [[str(r.mon)[:7],f(r.fold),f(r.customer_month_rows),f(r.target_1),f(r.target_2),f(r.target_3),f(r.target_4),f(r.missing_label_cells)] for r in p.itertuples()],
 [k,'descriptive.targets/label_coverage'],
 'Mỗi nhóm có số khách = số dòng; 4 nhãn đều quan sát, không có ô null. Tỷ lệ từng nhãn = số dương / số dòng của nhóm. Ma trận giữ toàn bộ 240 số đếm dương; CSV có thêm các cột tỷ lệ và số nhãn hợp lệ.',
 widths=[89,44,120,95,95,95,95,CW-633],mode='Toàn bộ nhóm và số đếm; cột suy ra + CSV')
k='review/target_monthly';d=load(k)
pages_table('Nhãn theo tháng · số đếm và tỷ lệ',['Tháng','Nhãn','Dương','Có nhãn','Tỷ lệ dương'],
 [[str(r.mon)[:7],r.target,f(r.positive_n),f(r.labelled_n),pct(r.positive_percent,6)] for r in d.itertuples()],[k],
 'Bảng số bổ sung hai trang biểu đồ zoom Y (trang 19–20). Mẫu số từng tháng và từng target là 100.224 khách; không thay tỷ lệ thấp bằng 0.',widths=[120,130,160,170,CW-580])

# Complete column dictionary + missingness in one view.
k='review/source_schema';d=load(k);miss=load('review/missing_values')
d=d.merge(miss[['source','column','missing_n','missing_rate']],on=['source','column'],validate='one_to_one')
pages_table('Từ điển 61 cột và kiểm tra null đầy đủ',['Bảng','Biến','Kiểu dữ liệu','Null','Tỷ lệ null'],
 [[r.source,r.column,r.dtype,f(r.missing_n),pct(r.missing_rate*100,6)] for r in d.itertuples()],
 [k,'review/missing_values','descriptive.structure/column_roles'],
 'clients: một khách; targets: một khách–tháng; trx/geo/dialog: một sự kiện; ptls_*: một chuỗi/khách. *_id là khóa; fold là phân vùng; mon/event_time là thời gian; target_* là nhãn. Các mã nguyên là biến phân loại, amount là số thực, embedding là vector.',
 widths=[112,170,257,90,CW-629],font=8.3,mode='61 cột đầy đủ; vai trò giải thích theo nhóm')
k='review/null_by_event_type';d=load(k)
pages_table('Null giao dịch theo từng event_type',['Mã loại','Dòng','currency','src11/12','dst11/12','src21/22','src31/32'],
 [[f(r.event_type),f(r.rows),f(r.null_currency),f(r.null_src11),f(r.null_dst11),f(r.null_src21),f(r.null_src31)] for r in d.itertuples()],[k],
 'Giá trị là số dòng null, không phải tỷ lệ. Thành viên của mỗi cặp thuộc tính luôn thiếu đồng thời. Dùng rows mỗi event_type làm mẫu số nếu tính tỷ lệ; bảng không xác nhận nguyên nhân nghiệp vụ của null.',widths=[74,129,107,112,112,112,CW-646])
for k,title in [('descriptive.quality/invalid_values','Kiểm tra giá trị không hợp lệ'),
 ('descriptive.customers/customer_aggregation_checks','Kiểm tra tổng hợp và giữ khách'),
 ('descriptive.structure/relationship_summary','Quan hệ giữa các bảng')]:
    d=load(k);pages_table(title,list(d.columns),[[fmt(v) for v in r] for r in d.itertuples(index=False,name=None)],[k],
      'Đây là kết quả kiểm tra cấu trúc trên toàn bộ dữ liệu. Giá trị 0 ở cột đếm lỗi nghĩa là không phát hiện lỗi trong phép kiểm tra tương ứng.',font=9)
k='descriptive.quality/key_coverage';d=load(k)
pages_table('Liên kết khóa với tập khách tham chiếu',['Nguồn','Khách tham chiếu','Có log hợp lệ','Không có log','ID lạ','ID null','% phủ'],
 [[r.source,f(r.reference_customers),f(r.customers_with_valid_time_log),f(r.customers_without_valid_time_log),f(r.unknown_client_rows),f(r.missing_id_rows),pct(r.coverage*100,4)] for r in d.itertuples()],[k],
 'distinct_ids_with_valid_time bằng customers_with_valid_time_log ở cả ba nguồn; không có ID ngoài tập tham chiếu hoặc ID null.',widths=[112,130,125,120,80,80,CW-647])
for k,title in [('review/duplicates','Duplicate · kết quả từng phép kiểm tra'),('review/sequence_quality','Chất lượng từng cột chuỗi ptls'),
 ('review/sequence_reconciliation','Đối chiếu sự kiện detail và chuỗi ptls'),('review/embedding_element_quality','Chất lượng vector đến từng phần tử'),
 ('descriptive.modalities/embedding_quality','Số chiều và chuẩn vector hội thoại')]:
    d=load(k);headers=list(d.columns)
    # Long reconciliation headings are explained verbatim in the source catalogue.
    if k.endswith('sequence_reconciliation'):headers=['Nguồn','Log detail','Phần tử ptls','Max detail','Max ptls','KH chuỗi ngắn','KH chuỗi dài','Lệch hiện diện']
    if k.endswith('embedding_element_quality'):headers=['Nguồn','Dòng','Vector null','Khác 768 chiều','Phần tử null','Phần tử vô hạn']
    if k.endswith('sequence_quality'):headers=['Nguồn','Cột','Dòng chuỗi','Phần tử','Phần tử null','Dòng lệch độ dài']
    if k.endswith('duplicates'):headers=['Nguồn','Phép kiểm tra','Dòng dư','Nhóm trùng','Dòng quét','Phạm vi']
    widths=None
    if k.endswith('embedding_quality'):widths=[CW*.59,CW*.2,CW*.21]
    if k.endswith('sequence_quality'):widths=[112,162,112,136,110,CW-632]
    pages_table(title,headers,[[fmt(v) for v in r] for r in d.itertuples(index=False,name=None)],[k],
      'Giữ nguyên kết quả quét. Vector dùng 768 chiều; không có phần tử null/vô hạn. ptls/trx bị thiếu 2.020.020 sự kiện tại 13 khách; detail là nguồn tính toàn lịch sử. Ô null trong bảng phép kiểm tra có thể là cột không áp dụng.',widths=widths,font=8.2)

# Sampling diagnostics all metrics + all 178 strata, including zero allocations.
k='review/sampling_validation';d=load(k)
pages_table('Đối chiếu mẫu · đầy đủ 24 chỉ tiêu',['Nguồn','Chỉ tiêu','Dân số','Mẫu'],
 [[r.source,r.metric,fmt(r.population_value),fmt(r.sample_value)] for r in d.itertuples()],[k],
 'P95 và max phản ánh phần đuôi. zero_percent có đơn vị %, KS_distance là khoảng cách CDF từ 0 đến 1. Dân số gồm 100.224 khách; mẫu gồm 1.000 khách, không dùng trọng số trong các thống kê so sánh này.',widths=[170,190,200,CW-560])
k='review/sampling_strata';d=load(k)
zero=d[d.sample_n.eq(0)]
pages_table('Phân bổ mẫu theo tầng · toàn bộ 178 tầng',['Tầng f:m:b','Dân số','Mẫu','Kỳ vọng','% dân số','% mẫu','Chênh đpt','Trọng số'],
 [[r.stratum,f(r.population_n),f(r.sample_n),f(r.expected_n,3),pct(r.population_rate*100,4),pct(r.sample_rate*100,3),f(r.difference_pp,4),'Không có mẫu' if r.sample_n==0 else f(r.population_n/r.sample_n,3)] for r in d.itertuples()],[k],
 'Tầng = fold : source_mask : activity_band. mask = 1×có trx + 2×có geo + 4×có dialog. Band 0 không log; 1–6 theo các mốc P25/P50/P75/P95/P99. Kỳ vọng = 1.000×tỷ trọng dân số; chênh đpt = % mẫu − % dân số; trọng số = dân số/mẫu.',
 widths=[91,80,60,90,111,100,104,CW-636],font=8.4)
pages_table('Giới hạn của mẫu và cách sử dụng',['Nội dung','Kết quả / ý nghĩa'],[
 ['Tầng không nhận mẫu',f'{len(zero)} / {len(d)} tầng; {f(zero.population_n.sum())} khách trong dân số nằm ở các tầng này. Trọng số không khôi phục tầng vắng.'],
 ['Tái lập','Seed 42; Hamilton theo tỷ trọng; lấy ngẫu nhiên không hoàn lại; 1.000 khách không lặp.'],
 ['Biên hoạt động','201; 651; 1.428; 3.362; 5.664,8 tổng log trong nhóm có log. Mốc được tính từ toàn lịch sử.'],
 ['Bảo toàn liên kết','Giữ toàn bộ dòng detail, targets và ptls của khách được chọn; lưới customer_month đủ 25 tháng.'],
 ['Giới hạn suy luận','Mẫu phục vụ EDA; không bảo toàn chính xác cực trị, nhãn hiếm hoặc các tầng rất nhỏ. Khi mô hình hóa tại cutoff quá khứ phải tạo lại tầng chỉ từ thông tin trước cutoff.']],[k,'review/sample_inventory'],
 'Mẫu không thay thế kết quả thống kê toàn dân số trong báo cáo.',widths=[190,CW-190],font=10)
k='review/sampling_labels';d=load(k)
pages_table('Kiểm định tỷ lệ nhãn trong mẫu',['Nhãn','Dương dân số','% dân số','Dương mẫu','% mẫu','Chênh đpt'],
 [[r.target,f(r.population_positive_n),pct(r.population_percent,6),f(r.sample_positive_n),pct(r.sample_percent,4),f(r.sample_percent-r.population_percent,6)] for r in d.itertuples()],[k],
 'Dân số: 1.202.688 dòng nhãn, mẫu: 12.000 dòng. Đây là đối chiếu mô tả, chưa phải kiểm định thống kê về tính tương đương của tỷ lệ nhãn.',widths=[100,130,150,120,120,CW-620])

# Amount full quantiles by currency (no log or currency conversion).
k='descriptive.transactions/amount_by_currency';d=load(k)
for cols,title in [(['rows','finite_n','mean','std','min','max'],'Amount theo currency · quy mô và cực trị'),
 (['p25','median','p75','p95','p99'],'Amount theo currency · đầy đủ phân vị')]:
    pages_table(title,['Currency']+cols,
      [[fmt(r['currency'])]+[fmt(r[c]) for c in cols] for _,r in d.iterrows()],[k,'review/amount_by_currency'],
      'Dùng toàn bộ Amount hữu hạn; giữ nguyên giá trị nguồn. currency null được giữ thành nhóm riêng (118 dòng). Không quy đổi hay gộp để diễn giải giá trị tiền tệ; các nhóm rất nhỏ có phân vị không ổn định.',widths=[80]+[(CW-80)/len(cols)]*len(cols),font=8.2)
for k,title in [('review/amount_full','Amount toàn kỳ · đầy đủ thống kê'),('descriptive.transactions/amount_summary','Phương pháp và kiểm tra Amount toàn kỳ'),
 ('descriptive.transactions/transaction_outliers','Amount toàn kỳ · hàng rào và số ngoại lai')]:
    d=load(k);pages_table(title,['Biến','Giá trị'],[[c,fmt(d.iloc[0][c])] for c in d.columns],[k],
      'Phân vị tính chính xác theo nội suy tuyến tính type 7 trên toàn bộ dữ liệu. Sai khác rất nhỏ giữa hai lần cộng mean/std là sai số số thực. Whisker là giá trị quan sát trong hàng rào, khác giá trị hàng rào Tukey.',widths=[310,CW-310],font=9.1)
for k,title in [('review/amount_raw_bins','Bảng bin Amount · toàn miền giá trị'),('review/amount_zoom','Bảng bin Amount · vùng 0 đến P99')]:
    d=load(k);pages_table(title,['Bin','Biên trái','Biên phải','Số Amount','% toàn bộ Amount'],
      [[f(r.bin),fmt(r.left),fmt(r.right),f(r.n),pct(r.n/38764930*100,6)] for r in d.itertuples()],[k],
      'Biên −∞ và +∞ là đầu mở vô hạn, không phải Amount null. Đây là bảng bin của phiên bản trước, giữ để đối chiếu nguồn. Các biểu đồ phân phối Amount đã được bỏ khỏi báo cáo; bảng số gốc được giữ để tra cứu, không có bước quy đổi tiền tệ.',widths=[55,190,190,160,CW-595])
for k,title in [('review/amount_outliers_currency','Ngoại lai Amount · hai hàng rào theo currency'),
 ('review/customer_count_outliers','Ngoại lai theo khách · phân vị và hàng rào'),('review/daily_count_outliers','Ngoại lai theo ngày · đầy đủ thống kê')]:
    d=load(k);pages_table(title,list(d.columns),[[fmt(v) for v in r] for r in d.itertuples(index=False,name=None)],[k],
      'Hàng rào Tukey = Q1 − 1,5×IQR và Q3 + 1,5×IQR. Đây là cờ rà soát thống kê; giữ nguyên quan sát. above_p99_n đếm vượt P99, outlier_n đếm ngoài hàng rào; hai điều kiện không tương đương.',font=8)

# Smaller structural/population tables not redundant in exact representation.
for k,title in [('descriptive.structure/table_inventory','Thống kê phân mảnh và schema của nguồn'),
 ('descriptive.customers/source_count_distribution','Khách theo số nguồn có log'),
 ('descriptive.temporal/source_time_ranges','Thời gian nguồn · số dòng quét và hợp lệ')]:
    d=load(k);pages_table(title,list(d.columns),[[fmt(v) for v in r] for r in d.itertuples(index=False,name=None)],[k],
      'Thống kê trên toàn bộ dữ liệu. rate của source_count_distribution ở thang 0–1; timestamp giữ nguyên không chuyển múi giờ. Số sự kiện không cộng thêm dạng ptls.',font=8.6)

# Shared/equivalent evidence: preserve explicit source links and honest mapping.
equivalent={
 'descriptive.customers/modality_coverage':'review/behavior_combinations',
 'descriptive.quality/duplicates':'review/duplicates',
 'descriptive.quality/missing_values':'review/missing_values',
 'descriptive.transactions/amount_histogram':'review/amount_raw_bins',
 'descriptive.transactions/transaction_categories':'descriptive.transactions/transaction_categories_full',
 'descriptive.modalities/location_categories':'descriptive.modalities/location_categories_full'}
for k,v in equivalent.items():mark([k],LEDGER[v]['pages'],'Cùng chủ đề/biểu diễn; tệp gốc đầy đủ liên kết')
assert all(v['pages'] for v in LEDGER.values()),[k for k,v in LEDGER.items() if not v['pages']]

# Visual evidence index: all original figures mapped to the revised representations.
figure_keys={
 'amount_boxplot':['descriptive.transactions/amount_summary'],
 'amount_histogram':['review/amount_raw_bins'], 'amount_quantiles':['review/amount_full'],
 'amount_zoom_linear':['review/amount_zoom'], 'behavior_monthly_matrix':['review/monthly_coverage'],
 'behavior_rates_matrix':['review/behavior_rates','review/behavior_combinations'],
 **{f'coverage_group_{i}':['review/purposeful_coverage'] for i in range(1,6)},
 'coverage_population_groups':['review/group_monthly_coverage'],
 'customer_event_medians':['review/events_per_customer'], 'daily_behavior_linear':['review/daily_behavior'],
 'embedding_dimensions':['descriptive.modalities/embedding_quality'],
 'first_event_month':['descriptive.customers/first_last_event_month'],
 'last_event_month':['descriptive.customers/first_last_event_month'],
 'location_codes':['descriptive.modalities/location_categories_full'], 'missing_attributes':['review/missing_values'],
 'monthly_activity':['descriptive.temporal/monthly_activity'],'monthly_customers':['descriptive.temporal/monthly_activity'],
 'source_timeline':['descriptive.temporal/source_time_ranges'], 'target_by_month':['review/target_monthly'],
 'transaction_event_types':['descriptive.transactions/transaction_categories_full']}
def ranges(nums):
    nums=sorted(set(nums));out=[];a=b=nums[0]
    for n in nums[1:]:
        if n==b+1:b=n
        else:out.append(str(a) if a==b else f'{a}–{b}');a=b=n
    out.append(str(a) if a==b else f'{a}–{b}');return ', '.join(out)
figrows=[];figledger=[]
for fig in MANIFEST['figures']:
    name=Path(fig).stem;keys=figure_keys[name];p=sorted({n for k in keys for n in LEDGER[k]['pages']})
    if name.startswith('coverage_group_'):p=[11+int(name.rsplit('_',1)[1])]
    removed=name in {'amount_histogram','amount_zoom_linear'}
    representation='Đã bỏ biểu đồ; chỉ giữ bảng nguồn tại trang '+ranges(p) if removed else 'Biểu đồ tuyến tính / ma trận / bảng tương ứng; hình gốc có liên kết'
    figrows.append([name,'Đã bỏ' if removed else ranges(p),representation])
    figledger.append({'path':fig,'pages':p,'representation':'Chart removed; source table retained' if removed else 'Revised chart, matrix or source table','url':PUBLIC+fig})
pages_table('Đối chiếu đủ 24 hình đã thực hiện',['Hình nguồn','Trang phân tích','Hình thức trong bản đầy đủ'],figrows,[],
 'Báo cáo dùng dạng trình bày phù hợp yêu cầu mới; không nhất thiết lặp lại hình PNG gốc. Nhấp vào dòng để mở hình gốc. Các biểu đồ mới không dùng hàm log. Các biểu đồ phân phối Amount đã được bỏ theo yêu cầu; danh mục giữ thông tin bảng nguồn để đối chiếu.',widths=[245,150,CW-395],font=9,links=[x['url'] for x in figledger])

# Full table catalogue, with every exported field name and stable hyperlink.
catalog_start=PAGE+1
rows=[];urls=[]
for key,v in LEDGER.items():
    rows.append([key+'\n'+f(v['rows'])+' dòng · trang '+ranges(v['pages'])+'\n'+v['representation'],', '.join(v['columns'])])
    urls.append(RAW+v['path'])
pages_table('Đối chiếu đủ 61 bảng và toàn bộ tên biến',['Bảng · quy mô · trang phân tích','Các biến trong bảng CSV'],rows,[],
 'Nhấp vào từng dòng để mở CSV đầy đủ tại phiên bản cố định. Một bảng có thể được chuyển thành ma trận hoặc thống kê; hình thức được ghi rõ. Các bảng bản sao / bản mở rộng cùng chủ đề có liên kết riêng.',
 widths=[310,CW-310],font=8.3,links=urls)

y=page('Kết luận và điều kiện sử dụng kết quả','KẾT LUẬN')
for title,body in [
 ('Toàn lịch sử là phạm vi phân tích chính','100.224 khách và 105.277.859 sự kiện detail được giữ. 468 khách không có log vẫn thuộc quần thể. B0–B6 phản ánh toàn dân số; 40 khách chỉ minh họa các nhóm có chủ đích.'),
 ('Giữ đúng nghĩa thiếu dữ liệu','Không điền 0 cho thuộc tính null. Trước 01/2022, geo/dialog không có log được gắn null trong lưới với cờ unobserved_early_source; số đếm raw vẫn được bảo toàn. Có log sớm thì giữ nguyên.'),
 ('Cơ sở để xây dựng mô hình','Cần chốt cutoff đặc trưng, kiểm tra ý nghĩa và thời điểm tạo trường phụ trợ, đánh giá theo tháng và từng nhãn. Các phân tích hiện tại chưa chứng minh chất lượng dự báo hoặc hiệu quả kinh doanh.'),
 ('Tính đầy đủ và khả năng kiểm tra','Danh mục cuối báo cáo đối chiếu 61/61 bảng và 24/24 hình của manifest; các bảng dài có bản CSV đầy đủ theo liên kết. Báo cáo trình bày mọi chủ đề phân tích, không sao chép tuần tự hàng trăm nghìn dòng tần suất.')]:
    y=section_title(title,y);y=para(body,y,size=11,leading=16)-16
source('Nguồn ai-lab/MBD-mini (CC BY 4.0); EDA repository AnhQun18/Propensity, commit 3e4b95d. Các mã nguồn đã được nhà phát hành ẩn danh.')
C.save()
audit=[]
for key,v in LEDGER.items():
    path=DOC/v['path'];audit.append({'key':key,**v,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'url':RAW+v['path']})
(BUILD/'coverage_ledger.json').write_text(json.dumps({'commit':COMMIT,'tables':audit,'figures':figledger},ensure_ascii=False,indent=2),encoding='utf-8')
pd.DataFrame([{'table':x['key'],'rows':x['rows'],'pages':ranges(x['pages']),'representation':x['representation'],'url':x['url'],'columns':', '.join(x['columns'])} for x in audit]).to_csv(BUILD/'coverage_ledger.csv',index=False,encoding='utf-8-sig')
(BUILD/'build_summary.json').write_text(json.dumps({'pdf':str(OUTPUT),'pages':PAGE,'page_titles':PAGES,'source_commit':COMMIT,'tables':len(audit),'figures':len(figledger),'catalog_start':catalog_start,'amount_distribution':{'status':'removed_by_user_request','reason':'currency codes not mapped and monetary unit not verified'}},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'pdf':str(OUTPUT),'pages':PAGE,'tables':len(audit),'figures':len(figledger)},ensure_ascii=False))
