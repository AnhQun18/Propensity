"""Evidence-backed slide revision and explicit outline coverage ledger."""
import json
from pathlib import Path

def revise_deck(old,h):
    rows,fmt,tx,rect,line,svg,table,bars,trend,esc=(h[k] for k in ['rows','fmt','tx','rect','line','svg','table','bars','trend','esc'])
    root=h['ROOT']; base=h['base'];analysis='deliverables/propensity-data-story/analysis/'
    def read(name):return rows(analysis+name+'.csv')
    manifest=json.loads((root/analysis/'model_manifest.json').read_text(encoding='utf-8'))
    assert manifest['status']=='complete' and not manifest['test_opened'] and not manifest['calibration_opened']
    def slide(section,title,chart,meaning,method,source,notes=''):
        return dict(section=section,title=title,chart=chart,meaning='Ý nghĩa cho mô hình: '+meaning,
                    method=method,source=source,notes=notes,appendix=False)
    def dual_monthly(data,title,event_scale,event_unit,event_max,active_max):
        b='';l=60;r=1070;t=30;bottom=275
        for i in range(5):
            y=bottom-(bottom-t)*i/4
            b+=line(l,y,r,y,'--border')+tx(l-12,y+5,fmt(event_max*i/4,1),16,'--accent','end')+tx(r+12,y+5,fmt(active_max*i/4),16,'--accent-2')
        coords=[];width=(r-l)/len(data)
        for i,row in enumerate(data):
            x=l+(i+.5)*width;v=float(row['events'])/event_scale;active=float(row['customers'])/1000
            b+=rect(x-width*.28,bottom-(bottom-t)*v/event_max,width*.56,(bottom-t)*v/event_max,'--accent',2)
            coords.append((x,bottom-(bottom-t)*active/active_max))
            if i%max(1,len(data)//10)==0:b+=tx(x,300,row['month'][5:]+'/'+row['month'][2:4],15,'--text-2','middle')
        b+='<polyline points="'+' '.join(f'{x},{y}' for x,y in coords)+'" stroke="var(--accent-2)" stroke-width="3" fill="none"/>'
        b+=tx(l,20,'Dòng sự kiện ('+event_unit+')',17,'--accent')+tx(r,20,'ID active (nghìn)',17,'--accent-2','end')
        b+=rect(80,329,20,8,'--accent',1)+tx(110,340,'Cột: số dòng / tháng · trục trái',18,'--text-2')
        b+=line(630,332,650,332,'--accent-2',3)+tx(660,340,'Đường: ID active / tháng · trục phải',18,'--text-2')
        return svg(b,title)
    def decile_chart(feature,title):
        rr=[r for r in read('feature_deciles') if r['feature']==feature and int(r['band'])>0]
        b='';l=70;r=1120;bottom=275;top=35;vmax=max(float(v['lift']) for v in rr)*1.15
        for i in range(5):
            value=vmax*i/4;y=bottom-(bottom-top)*i/4
            b+=line(l,y,r,y,'--border')+tx(l-12,y+6,fmt(value,1)+'×',16,'--text-2','end')
        y=bottom-(bottom-top)/vmax;b+=line(l,y,r,y,'--warn',2)+tx(r,y-8,'Lift = 1',16,'--warn','end')
        for i,row in enumerate(rr):
            x=l+(i+.5)*(r-l)/len(rr);v=float(row['lift'])
            b+=rect(x-27,bottom-(bottom-top)*v/vmax,54,(bottom-top)*v/vmax,'--accent')
            b+=tx(x,bottom-(bottom-top)*v/vmax-9,fmt(v,2)+'×',18,'--text-1','middle',650)+tx(x,301,'D'+row['band'],18,'--text-2','middle')
        missing=next((v for v in read('feature_deciles') if v['feature']==feature and int(v['band'])==0),None)
        b+=tx(l,338,('Missing: '+fmt(missing['n'])+' mẫu; lift '+fmt(missing['lift'],2)+'×') if missing else 'Count = 0 nằm trong phân phối; ties giữ cùng nhóm.',18,'--text-2')
        return svg(b,title)

    problem=old[0]
    challenges=slide('A2 · THÁCH THỨC','Một năm nhãn và 1,07% dương giới hạn cách đánh giá',
        table(['Thách thức','Số liệu đã kiểm tra','Thiết kế kéo theo'],[
        ['Quy mô','100.224 ID · 105,28M sự kiện','Aggregate theo cutoff; đọc theo cột'],
        ['Nhãn lệch','12.854 / 1.202.688 = 1,07%','AP và Recall@K; giữ phân bố thật'],
        ['Lịch sử lệch nguồn','Trx 2 năm; geo từ 11/12/2021','Masks và cờ cửa sổ bị cắt'],
        ['Nhãn bắt đầu muộn','02/2022; SLA chưa xác minh','Không gán âm trước khi có nhãn']],[23,38,39]),
        'khóa split theo tháng và giả định nhãn khả dụng trước khi đọc metric.',
        'Số dòng nhãn là khách–tháng; số dòng detail là sự kiện. Khoảng thiếu nhãn đầu kỳ khác độ trễ công bố nhãn.',
        'docs/mbd_inventory/assessment.md; remaining_questions/assessment.md')
    agenda=slide('A3 · CÂU HỎI DỮ LIỆU','Bốn câu hỏi nối EDA với quyết định mô hình',
        table(['Câu hỏi','Bằng chứng cần đọc','Quyết định'],[
        ['01 · Có tín hiệu ngoài mua tháng trước?','Persistence + B+H vs rule','Chọn baseline đủ mạnh'],
        ['02 · Tín hiệu nằm ở nguồn nào?','Lift đơn biến + bỏ từng nguồn','Giữ nguồn có giá trị thêm'],
        ['03 · Snapshot và cửa sổ thế nào?','31 biến cố định, 4 cửa sổ','Chọn lịch sử theo validation'],
        ['04 · Rủi ro nào làm sai kết luận?','Cohort drift, thiếu nguồn, cutoff','Khóa protocol và test']],[40,34,26]),
        'mỗi câu hỏi phải có số liệu quan sát và một quyết định tiếp theo.',
        'Lộ trình đọc: B toàn cảnh → C từng file → D đối chứng → E mức phù hợp.',
        'Đề cương yêu cầu; coverage_ledger.json')
    combined_agenda=old[1]

    cohort=read('target_cohort_drift');months=sorted(set(v['month'] for v in cohort))
    series=[]
    for key,name,color in [('history_180d','Lịch sử ≥180 ngày','--accent'),('history_under180d','Lịch sử <180 ngày','--accent-2'),('no_trx','Chưa có trx','--accent-3')]:
        rr=[v for v in cohort if v['cohort']==key]
        series.append((name,[float(v['rate_pct']) for v in rr],color))
    target_drift=slide('C1 · TARGET / CẤU TRÚC → ĐỘNG HỌC','Tỷ lệ mua biến động cả trong các cohort lịch sử cố định',
        trend([m[5:]+'/22' for m in months],series,2.5,[0,.5,1,1.5,2,2.5],'%'),
        'cần theo dõi drift trong từng nhóm khách, bên cạnh prevalence toàn bộ.',
        'Cohort chốt tại 01/02/2022: trx chưa có / span <180 / ≥180 ngày; cố định ID qua 02–11/2022. Tỷ lệ = dương / ID cohort.',
        analysis+'target_cohort_drift.csv',
        'Targets có 100.224 ID/tháng, 4 nhãn 0/1, không unknown; khóa client_id×month duy nhất. Cohort dùng lịch sử trước cutoff đầu tiên, không dùng first/last event tương lai. Cohort lần lượt có 13.221, 5.737, 81.266 ID. Bản toàn kỳ prevalence 02/2022–01/2023 nằm phụ lục.')
    # Long-tail conditional chart excludes the overwhelming never-positive population.
    dist=h['aggregate']['positive_month_histogram'];ever=sum(v['customers'] for v in dist if v['months']>0)
    tail=slide('C2 · TARGET / PHÂN PHỐI','18,4% người từng dương xuất hiện ở ít nhất hai tháng',
        bars(['Dương 1 tháng','Dương 2 tháng','Dương 3 tháng','Dương ≥4 tháng'],[100*sum(v['customers'] for v in dist if v['months']==1)/ever,100*sum(v['customers'] for v in dist if v['months']==2)/ever,100*sum(v['customers'] for v in dist if v['months']==3)/ever,100*sum(v['customers'] for v in dist if v['months']>=4)/ever],maximum=100),
        'lịch sử mua có đuôi phải; vẫn phải tìm khách chưa từng dương trong dữ liệu.',
        'Mẫu số 10.411 ID có ≥1 tháng y_any=1 trong 02/2022–01/2023. Max 8 tháng; 89.813 ID chưa dương. Đây là số tháng, không phải số đơn.',
        'aggregate_data.json → positive_month_histogram',
        'Phân phối nhị phân tháng chỉ cho biết số tháng dương. Không xác nhận heavy tail theo định nghĩa power law, không đo số lượt mua trong tháng. Các nhãn cuối kỳ chỉ dùng mô tả như EDA đã công bố.')
    persistence=old[6]

    trx=[r for r in h['monthly'] if r['source']=='transactions' and r['month']>='2021-01']
    trx_time=slide('C4 · TRX / CẤU TRÚC & ĐỘNG HỌC','Trx active tăng từ 69 nghìn lên khoảng 78 nghìn ID/tháng',
        dual_monthly(trx,'Trx: số dòng và ID theo tháng',1e6,'triệu',2,100),
        'dùng frequency/rate và cửa sổ lịch để tách cường độ khỏi độ dài tháng.',
        '14 cột + fold; 38.764.930 sự kiện, 98.721 ID. Cột = dòng/tháng (triệu); đường = ID có ≥1 trx/tháng (nghìn).',
        'descriptive.temporal/monthly_activity.csv; docs/mbd_inventory/assessment.md')
    sparse=read('sparsity_monthly')
    sparsity=slide('C5 · TRX / PHÂN PHỐI & IM LẶNG','Khách im lặng 84 ngày vẫn có người mua trong tháng kế tiếp',
        trend([r['month'][5:]+'/22' for r in sparse],[('Không trx 30d',[100*int(r['silent_30d'])/int(r['n']) for r in sparse],'--accent'),('Không trx 84d',[100*int(r['silent_84d'])/int(r['n']) for r in sparse],'--accent-2')],40,[0,10,20,30,40],'%'),
        'giữ khách im lặng; thêm recency và cờ không log thay vì loại khỏi scoring.',
        'Mẫu số 100.224 ID/cutoff; không log trong [T−30d,T) hoặc [T−84d,T). Toàn kỳ: median 164, P99 2.234, max 1.020.773 trx/ID.',
        analysis+'sparsity_monthly.csv; review/events_per_customer.csv',
        'Đường này dùng snapshot quá khứ ở từng cutoff, không dùng ngày cuối tương lai của khách. Cửa sổ 84 ngày khác 3 tháng lịch. Không có sự kiện không tự chứng minh mất log. Tỷ lệ mua của nhóm im lặng được lưu trong CSV.')
    rfm=slide('C6 · TRX / LIÊN HỆ NHÃN','Nhóm frequency cao nhất có lift mua khoảng 2,74 lần',
        decile_chart('F','Lift theo decile số trx 30 ngày'),
        'R/F/M có tín hiệu khác nhau; dùng cả count, recency và amount tách currency.',
        'Train 02–08/2022; lift = tỷ lệ mua nhóm / 1,2361%. Decile theo midpoint rank của giá trị; ties cùng nhóm nên không đủ 10 cột.',
        analysis+'feature_deciles.csv',
        'D10 frequency gồm 42–45.908 trx/30d; 68.836 mẫu, 2.329 dương, prevalence 3,3834%. R và M có biểu đồ riêng ở phụ lục. M là tổng |amount| mã 11; chưa biết tiền tệ thật. Các decile chỉ mô tả trên train.')
    cc=read('category_lift')[:5];b='';x=0
    for i,row in enumerate(cc):
        w=1160*float(row['event_share_pct'])/100
        b+=rect(x,40,w-4,200,['--accent','--accent-2','--accent-3','--text-2','--good'][i])
        b+=tx(x+10,80,'Mã '+row['category'],19,'--accent-ink',weight=650)
        b+=tx(x+10,114,fmt(row['event_share_pct'],1)+'%',18,'--accent-ink')
        x+=w
    b+=rect(x,40,1160-x,200,'--surface-2')+tx(x+12,85,'Khác',19,'--text-1')
    b+=tx(0,278,'Lift khi có mã trong 30 ngày trước cutoff:',21,'--text-1',weight=650)
    for i,row in enumerate(cc):b+=tx(i*230,319,'Mã '+row['category']+': '+fmt(row['lift'],2)+'×',22,'--accent',weight=650)
    category=slide('C7 · TRX / DANH MỤC & NHÃN','Có giao dịch mã 40 gắn với lift mua 2,21 lần trên train',svg(b,'Treemap tần suất và lift theo event_type'),
        'giữ thống kê theo mã; kiểm tra giá trị tăng thêm ngoài tổng frequency.',
        'Diện tích = tỷ trọng số trx trong cửa sổ 30d của train. Lift = P(y=1|có mã)/P(y=1). Các nhóm mã có thể chồng khách–tháng.',
        analysis+'category_lift.csv',
        '53 event_type có mặt trong cửa sổ train, khác 55 mã toàn lịch sử. Mã 40: 192.493 mẫu có mã, 5.253 dương, tỷ lệ 2,7289%; nhóm không có mã 0,6716%. Đây là liên hệ đơn biến chưa kiểm soát cường độ giao dịch. Không gán tên nghiệp vụ cho mã ẩn danh.')

    dr=[r for r in h['monthly'] if r['source']=='dialog']
    dialog_coverage=slide('C8 · DIALOG / CẤU TRÚC & ĐỘ PHỦ','Dialog chỉ có ở 46,2% khách và đổi nhịp mạnh đầu 2022',
        dual_monthly(dr,'Dialog: số sự kiện và khách active theo tháng',1000,'nghìn',20,10),
        'giữ cờ không dialog và phân tích thời gian; không coi vắng hỗ trợ là lỗi dữ liệu.',
        '3 cột + fold: ID, timestamp, embedding 768D. Bắt đầu 10/02/2021; 217.205 dòng / 46.320 ID. Median 3 sự kiện/ID có log toàn kỳ.',
        'monthly_activity.csv; descriptive.modalities/dialog_summary.csv',
        'Không có text, thời lượng hay độ dài hội thoại trong schema. Mọi embedding có chiều 768 theo audit; số chiều cố định không phải độ dài văn bản. Không báo kết quả nội dung chưa chạy. Giờ/ngày trong tuần có thể suy từ timestamp với giả định UTC.')
    dm=read('dialog_metadata');selected=[]
    for key,band,name in [('frequency','0','Không dialog'),('frequency','1','1 dialog'),('frequency','2+','≥2 dialog'),('recency','0_7d','Dialog gần ≤7 ngày'),('night_00_06','yes','Có dialog 00–06 UTC')]:
        rr=next(v for v in dm if v['feature']==key and v['band']==band);selected.append((name,float(rr['lift'])))
    dialog_signal=slide('C9 · DIALOG / METADATA & NHÃN','Tần suất và recency dialog phân tách nhóm nhãn trên train',
        bars([v[0] for v in selected],[v[1] for v in selected],unit='×',maximum=max(v[1] for v in selected)*1.2),
        'count và recency là mốc; kiểm tra giờ tương tác khi đã kiểm soát tần suất.',
        'Lift so với prevalence train 1,2361%; 30 ngày trước cutoff; nhóm có chồng lấn. Schema không có text/duration/độ dài hội thoại; embedding chưa thử.',
        analysis+'dialog_metadata.csv',
        'Giờ timestamp đang giả định UTC. Nhóm 00–06 không đủ để kết luận giờ là tín hiệu độc lập: đó có thể là proxy của số hội thoại hoặc hành vi hỗ trợ. CSV cũng có nhóm cuối tuần và nhóm đối chứng trong số khách có hội thoại. Nội dung/embedding chưa thử; không đưa thành một slide kết quả.')

    gr=[r for r in h['monthly'] if r['source']=='geo']
    quality=read('geo_quality_proxy')[0]
    geo_coverage=slide('C10 · GEO / CẤU TRÚC, THỜI GIAN & NHIỄU','Geo chỉ có từ 11/12/2021, phủ 72,4% ID toàn lịch sử',
        dual_monthly(gr,'Geo: số sự kiện và khách active theo tháng',1e6,'triệu',8,80),
        'đánh dấu nguồn đến muộn; chất lượng vị trí vật lý vẫn cần xác minh riêng.',
        '5 cột + fold; 66.295.724 dòng / 72.573 ID. Audit toàn kỳ không có null; mẫu mã vị trí và chuyển mã xem phụ lục chất lượng.',
        'monthly_activity.csv; '+analysis+'geo_quality_proxy.csv',
        'Mã geohash_4/5/6 đã ẩn danh. Không suy khoảng cách hoặc vận tốc từ mã. Kiểm tra timestamp trùng nhưng khác mã và đổi mã trong 60 giây chỉ là proxy, không xác nhận lỗi GPS hoặc hành vi di chuyển thực.')
    source={r['model']:r for r in read('source_ablation_summary')}
    geo_loss=float(source['without_geo']['ap_loss_vs_full'])
    gd=[r for r in read('feature_deciles') if r['feature']=='geo_diversity']
    geo_signal=slide('C11 · GEO / VÙNG VÀ GIÁ TRỊ BỔ TRỢ',f'Bỏ geo làm AP {"giảm" if geo_loss>=0 else "tăng"} {fmt(abs(geo_loss),4)} trên validation',
        decile_chart('geo_diversity','Lift theo số mã vị trí khác nhau trong 30 ngày'),
        'dùng geo như nguồn bổ trợ theo ablation; chưa suy được quãng đường thực.',
        'Chart: lift theo nunique geohash_6 trên train. Tiêu đề: AP full − AP bỏ geo, trung bình 09–11/2022, giữ cùng nền lịch sử mua.',
        analysis+'feature_deciles.csv; source_ablation_summary.csv',
        'Các đặc trưng vùng gồm số mã khác nhau và tỷ trọng mã phổ biến nhất. Dấu/độ lớn của AP loss phụ thuộc cấu hình cố định và các nguồn còn lại; không phải kiểm định nhân quả. Mẫu số D0 là missing, count 0 vẫn là nhóm hợp lệ. So sánh concentration nằm phụ lục.')

    losses=[float(source['without_'+s]['ap_loss_vs_full']) for s in ['trx','dialog','geo']]
    b=tx(0,25,'Full B+H: AP '+fmt(source['combined']['ap'],4)+' · Recall@1% '+fmt(float(source['combined']['recall1'])*100,2)+'%',25,'--accent',weight=700)
    vmax=max(abs(x) for x in losses)*1.25
    for i,(name,value) in enumerate(zip(['Bỏ trx','Bỏ dialog metadata','Bỏ geo'],losses)):
        y=68+i*70
        b+=tx(0,y+26,name,23,weight=650)+line(550,y,550,y+37,'--border-strong',2)
        width=400*abs(value)/vmax
        b+=rect(550 if value>=0 else 550-width,y,width,35,'--accent' if value>=0 else '--warn')
        b+=tx(1015,y+26,('+' if value>=0 else '−')+fmt(abs(value),4),24,'--text-1',weight=700)
    b+=tx(550,320,'0',18,'--text-3','middle')+tx(0,345,'ΔAP = AP(full) − AP(bỏ nguồn); dương nghĩa nguồn có giá trị bổ sung.',21,'--text-2')
    source_signal=slide('D1 · ABLATION THEO NGUỒN','Trx và dialog bổ sung AP khoảng 0,0083; geo khoảng 0,0003',svg(b,'Mức giảm AP khi bỏ từng nguồn'),
        'ưu tiên nguồn theo ΔAP và độ ổn định từng tháng, cùng một nền H.',
        'Full 156 biến đã lưu; mỗi lần bỏ toàn bộ biến của một nguồn, giữ 15 biến lịch sử mua, cùng split/cohort/seed/config. AP macro 09–11/2022.',
        analysis+'source_ablation_summary.csv; model_manifest.json',
        'Depth 4, 200 cây, learning rate 0,04, L2 5. Chín model mới = ba nguồn × ba fold. Full model được tái dùng sau kiểm tra hash artifact. Đây là so sánh khám phá trên validation đã được dùng phát triển, chưa mở test. Các đóng góp có điều kiện và không cộng thành tổng AP.')

    errors=read('error_segments')
    segs=['bought_lag1','not_bought_lag1','no_dialog_history','has_dialog_history']
    labels=['Mua T−1','Không mua T−1','Chưa có dialog','Đã có dialog']
    b=tx(0,22,'Recall trong từng nhóm · top 1% chung rồi mới phân nhóm',22,'--text-2')
    for i,(key,label) in enumerate(zip(segs,labels)):
        y=52+i*64;b+=tx(0,y+26,label,21,weight=650)
        for j,(model,color) in enumerate([('combined','--accent'),('rule','--accent-2')]):
            rr=[v for v in errors if v['segment']==key and v['model']==model]
            v=sum(int(v['hits']) for v in rr)/sum(int(v['positives']) for v in rr)*100
            b+=rect(420,y+j*22,560*v/100,16,color,3)+tx(1005,y+16+j*22,fmt(v,2)+'%',18,color,weight=650)
    b+=rect(420,332,20,5,'--accent',1)+tx(450,342,'B+H 156 biến',18,'--text-2')+rect(730,332,20,5,'--accent-2',1)+tx(760,342,'Rule mua T−1',18,'--text-2')
    error=slide('D2 · LỖI CỦA BASELINE MỚI NHẤT','Khách chưa có dialog là điểm yếu của B+H ở ngân sách 1%',svg(b,'Recall phân nhóm của B+H so với rule'),
        'kiểm tra ranking riêng khách thiếu dialog, bên cạnh khách quay lại và khách chưa dương.',
        'Gộp TP / dương của ba tháng 09–11/2022. Mỗi phương pháp chọn 1.003 ID/tháng trên toàn cohort; các hàng 1–2 và 3–4 là hai cách chia riêng.',
        analysis+'error_segments.csv; error_overlap.csv',
        'Dùng đúng dự đoán B+H 156 biến, thay chẩn đoán cũ 153 biến. Nhóm lịch sử và nguồn chỉ dùng quá khứ trước cutoff. Chưa có dialog nghĩa không có log dialog trước T, không khẳng định lỗi thu thập. Nhóm chưa từng dương và đã từng dương, cùng geo coverage, có trong CSV và phụ lục dương giả.')

    windows={r['model']:r for r in read('window_sensitivity_summary')}
    worder=['recent','30d','84d','all'];wlabels=['Tháng lịch T−1','30 ngày','84 ngày','Tất cả quá khứ']
    best=max(worder,key=lambda k:float(windows['window_'+k]['ap']))
    snapshot=slide('D3 · SNAPSHOT & CỬA SỔ',f'Với cùng 31 biến, cửa sổ {wlabels[worder.index(best)].lower()} đạt AP cao nhất',
        bars(wlabels,[float(windows['window_'+k]['ap']) for k in worder],unit='',maximum=max(float(v['ap']) for v in windows.values())*1.25),
        'thay độ dài cửa sổ phải so cùng cutoff và bộ biến; mốc T bất kỳ cần nhãn tương thích.',
        '12 fit mới: 4 cửa sổ × 3 fold, cùng 16 thống kê hành vi + 15 biến H, 100.224 ID/tháng, config cố định. AP trung bình 09–11/2022.',
        analysis+'window_sensitivity_summary.csv; model_manifest.json',
        'Recent là tháng lịch 28–31 ngày; 30/84/all đều dừng trước đầu tháng nhãn. Nghiên cứu này đổi cửa sổ, chưa đổi ngày dự đoán. Labels sản phẩm chỉ có tháng nên không thể xác minh y trong 30 ngày sau một ngày T bất kỳ; bảng khả thi cutoff ngày ở phụ lục chỉ mô tả feature, không tạo nhãn giả.')

    heatmap=old[16].copy();heatmap['notes']+=' Rà soát mới bổ sung source ablation 9 fit và matched-window 12 fit; bảng định tính không phải điểm chuẩn bên ngoài.'
    risk=old[17]
    conclusion=slide('E3 · KẾT LUẬN','Dữ liệu đủ để nghiên cứu; ưu tiên nguồn và cửa sổ theo đối chứng',
        table(['Ưu tiên','Bằng chứng hiện tại','Việc tiếp theo'],[
        ['01 · Khóa protocol','SLA nhãn / timezone chưa xác minh','Xác nhận thời điểm khả dụng; chốt lịch fit/calibration'],
        ['02 · Chốt cấu hình và nhóm yếu','Ablation nguồn + cửa sổ; khách thiếu dialog','Lặp nhiều seed / mốc; kiểm tra ranking nhóm thiếu nguồn'],
        ['03 · Kiểm chứng độc lập','B+H đạt Recall@1% validation 9,74%','Khóa lựa chọn rồi mở test 01/2023; theo dõi drift']],[27,35,38]),
        'khóa protocol → xác nhận cấu hình → đánh giá test độc lập và theo dõi drift.',
        'Kết quả là offline validation; chưa đo uplift chiến dịch, doanh thu hay ROI. Test 01/2023 chưa mở.',
        analysis+'model_manifest.json; deliverables/baseline-final/REPORT.md')

    main=[problem,combined_agenda,old[2],old[3],old[4],target_drift,persistence,trx_time,rfm,category,
          dialog_coverage,dialog_signal,geo_signal,source_signal,error,snapshot,heatmap,risk,conclusion]
    detailed=[problem,challenges,agenda,old[2],old[3],old[4],target_drift,tail,persistence,trx_time,sparsity,rfm,category,
              dialog_coverage,dialog_signal,geo_coverage,geo_signal,source_signal,error,snapshot,heatmap,risk,conclusion]
    appendix=[]
    def app(s):
        s=s.copy();s['appendix']=True;s['section']='PHỤ LỤC · '+s['section'];appendix.append(s)
    # Put every omitted narrative page within reach of the short deck.
    for s in [challenges,agenda,tail,sparsity,geo_coverage]:app(s)
    app(old[5])
    for key,name in [('R','R · Recency'),('M','M · Monetary'),('geo_concentration','Geo · mức tập trung vùng')]:
        app(slide(name,'Lift theo '+name.lower()+' bổ sung góc nhìn ngoài frequency',decile_chart(key,name),
            'đọc cùng missing và kích thước nhóm; không dùng lift đơn biến thay ablation.',
            'Train 02–08/2022; giữ ties cùng nhóm, D0=missing. R là ngày từ trx cuối, M là tổng |amount| currency 11; geo là share mã phổ biến nhất.',
            analysis+'feature_deciles.csv'))
    app(old[8])
    app(slide('GEO / NHIỄU','Chuyển mã trong 60 giây là dấu hiệu kiểm tra, chưa phải lỗi vị trí',
        table(['Phép kiểm tra trên mẫu ID','Kết quả','Giới hạn'],[
        ['Mẫu hash(client_id)%100=0',fmt(quality['clients'])+' ID · '+fmt(quality['rows'])+' dòng','Không phải toàn bộ 66M dòng'],
        ['Cặp sự kiện trong ≤60 giây',fmt(quality['pairs_within60s']),'Timestamp khác nhau'],
        ['Trong đó đổi mã geohash_6',fmt(quality['changes_within60s']),'Có thể là di chuyển / cập nhật mã'],
        ['Cùng timestamp, khác mã',fmt(quality['simultaneous_code_conflicts']),'Cần rà soát phía nguồn'],
        ['Null / duplicate toàn kỳ','0 theo audit đã công bố','Không xác nhận độ đúng vật lý']],[44,29,27]),
        'giữ cờ và kiểm tra độ bền đặc trưng, không tự xóa sự kiện theo proxy.',
        'Chỉ events trước 01/09/2022. Cặp trong mỗi ID xếp theo timestamp, source_file, source_row; không diễn giải mã thành tọa độ.',
        analysis+'geo_quality_proxy.csv; descriptive.quality/duplicates.csv'))
    # Per-month ablation makes the source conclusion auditable.
    sm=read('source_ablation_monthly')
    sr=[]
    for month in ['2022-09','2022-10','2022-11']:
        vals={r['model']:float(r['ap']) for r in sm if r['month']==month}
        sr.append([month,fmt(vals['combined'],4),*[fmt(vals['combined']-vals['without_'+s],4) for s in ['trx','dialog','geo']]])
    app(slide('ABLATION / TỪNG THÁNG','Đóng góp nguồn cần được đọc cùng biến động từng tháng',
        table(['Tháng','AP full','ΔAP bỏ trx','ΔAP bỏ dialog','ΔAP bỏ geo'],sr,[18,19,21,21,21]),
        'nếu dấu chênh thay đổi, cần thêm mốc thời gian trước khi bỏ nguồn.',
        'ΔAP = AP full − AP reduced, số dương ủng hộ giữ nguồn trong cấu hình này; ba tháng không phải ba thế giới độc lập.',
        analysis+'source_ablation_monthly.csv'))
    imp=read('importance_by_source')
    app(slide('IMPORTANCE / CHẨN ĐOÁN','Importance và ablation trả lời hai câu hỏi khác nhau',
        bars([r['source'] for r in imp],[float(r['mean_importance']) for r in imp],unit='%',maximum=100),
        'dùng importance để hiểu model đã fit; dùng ΔAP để đo giá trị giữ nguồn.',
        'CatBoost PredictionValuesChange của full B+H, cộng theo prefix nguồn và trung bình ba model fold; tổng 100%.',
        analysis+'importance_by_source.csv',
        'Đây là model importance có điều kiện, không phải phân rã nhân quả; nhóm tương quan có thể chia sẻ/tước importance. PURCHASE là lịch sử target khả dụng, không phải target cùng tháng.'))
    eo=read('error_overlap');totals=[sum(int(r[k]) for r in eo) for k in ['common_hits','combined_only_hits','rule_only_hits','missed_by_both']]
    app(slide('LỖI / GIAO DANH SÁCH','Hai cách chọn khách có cả phần bổ sung và phần cùng bỏ sót',
        bars(['Cả B+H và rule bắt đúng','Chỉ B+H bắt đúng','Chỉ rule bắt đúng','Cả hai bỏ sót'],totals,unit=' khách–tháng',maximum=max(totals)*1.15),
        'khách ngoài top-K là phần dư địa lớn; cải thiện rank phải giữ cùng ngân sách.',
        'True positive trong giao/hiệu hai danh sách top 1%; cộng 09–11/2022. Khách có thể lặp, không phải ID unique.',
        analysis+'error_overlap.csv'))
    erows=[]
    for key,label in [('no_purchase_history','Chưa dương trong lịch sử'),('prior_purchase_history','Đã dương trong lịch sử'),('no_dialog_history','Chưa có dialog'),('has_dialog_history','Đã có dialog'),('no_geo_history','Chưa có geo')]:
        rr=[v for v in errors if v['segment']==key and v['model']=='combined']
        hits=sum(int(v['hits']) for v in rr); selected_n=sum(int(v['selected']) for v in rr); positives=sum(int(v['positives']) for v in rr)
        erows.append([label,fmt(selected_n),fmt(hits),fmt(selected_n-hits),fmt(100*hits/selected_n,2)+'%' if selected_n else '—'])
    app(slide('LỖI / DƯƠNG GIẢ THEO NHÓM','Phần lớn lựa chọn top 1% vẫn là dương giả ở các nhóm khách',
        table(['Nhóm khách ở cutoff','Được chọn','TP','FP','Precision'],erows,[40,16,13,13,18]),
        'tăng Recall@K cần cân bằng với chi phí dương giả; báo precision riêng theo nhóm.',
        'B+H 156 biến, top 1% chung, gộp ba tháng. FP = được chọn − TP; các hàng về lịch sử mua / nguồn là các cách phân nhóm có chồng lấn.',
        analysis+'error_segments.csv'))
    wm=read('window_sensitivity_monthly');wr=[]
    for tag,label in zip(worder,wlabels):
        rr=[r for r in wm if r['model']=='window_'+tag]
        wr.append([label,*[fmt(r['ap'],4) for r in rr],fmt(sum(float(r['recall1']) for r in rr)/3*100,2)+'%'])
    app(slide('SNAPSHOT / KIỂM SOÁT BIẾN','So cửa sổ giữ nguyên 31 predictor và ba cutoff validation',
        table(['Cửa sổ','AP 09/22','AP 10/22','AP 11/22','Recall@1% macro'],wr,[30,16,16,16,22]),
        'kết quả thuộc bộ biến nhỏ cố định; không so trực tiếp thành tác dụng cửa sổ của full 156 biến.',
        '16 biến hành vi chung: has_event, log count, active days, recency mỗi nguồn + 2 trx + 2 geo; cộng 15 H.',
        analysis+'window_sensitivity_monthly.csv; model_manifest.json'))
    app(slide('SNAPSHOT / T BẤT KỲ','Có thể tạo feature tại T bất kỳ, nhưng nhãn sản phẩm mới là giới hạn',
        table(['Thiết kế','Lịch sử','Đáp án có thể xác minh'],[
        ['T = đầu tháng M','[T−L,T)','y_any của tháng M: có nhãn'],
        ['T giữa tháng, dự đoán 30 ngày tới','[T−L,T)','Không suy được từ nhãn nhị phân tháng'],
        ['T giữa tháng, dự đoán tháng M+1','[T−L,T)','Có nhãn tháng sau; là lead time khác'],
        ['Cutoff ngày với 84d, đủ ba nguồn','301 cutoff; max 11.188 ID/cutoff','Chỉ đo feature, không phải target sản phẩm']],[35,27,38]),
        'phải chốt horizon và availability; không gán một nhãn tháng cho nhiều horizon khác nhau.',
        '301 cutoff 06/03–31/12/2022 trong 384 ngày chung; bộ lọc đủ ba nguồn chỉ mô tả, không áp vào model toàn cohort.',
        'docs/mbd_temporal_coverage/assessment.md'))
    leak=read('leakage_audit')[0]
    leakage=slide('TRX / KIỂM TRA LEAKAGE','Giao dịch trước nhãn đầu tiên là lịch sử, không tạo nhãn âm',
        table(['Kiểm tra','Kết quả / quy tắc'],[
        ['Trx trước 01/02/2022',fmt(leak['before_first_label'])+' sự kiện · giữ làm lịch sử'],
        ['Bảng feature ở từng cutoff','Chỉ event_time < cutoff; không dùng nhãn cùng tháng'],
        ['Tháng không có target','Không biến thành y_any=0; không tham gia supervised loss'],
        ['trans_count / diff_trans_date','Công thức / cutoff nguồn chưa xác minh → loại khỏi X'],
        ['Top mã, ngưỡng và encoder','Fit trên train; audit toàn kỳ chỉ dùng mô tả']],[36,64]),
        'cần kiểm soát sự kiện và availability từng dòng, kể cả trước thời kỳ có nhãn.',
        'Đếm từ cache chuẩn hóa chỉ đọc; phạm vi audit đến trước 01/09/2022. Chưa có available_at thực tế để xác minh SLA của nhãn.',
        analysis+'leakage_audit.csv; docs/propensity_feasibility/remaining_questions/assessment.md')
    app(leakage);app(old[25])

    # Each original request maps to concrete content and evidence. No silent omission.
    groups=[
      ('A1','Phát biểu bài toán và định nghĩa nhãn',problem,'Đã có'),
      ('A2','Lệch nhãn, quy mô, target đến muộn',challenges,'Đã đo; SLA 6 tháng không xác minh'),
      ('A3','3–4 câu hỏi dẫn dắt',agenda,'Đã có'),
      ('B1','4 nguồn: dòng, ID, cột, vai trò',old[2],'Đã đo'),
      ('B2','Gantt và vùng trx chưa có target',old[3],'Đã đo'),
      ('B3','UpSet tổ hợp đủ trx+geo+dialog',old[4],'Đã đo'),
      ('C.Target1','Tỷ lệ dương theo tháng và cohort; drift',target_drift,'Bổ sung cohort cố định qua 10 tháng'),
      ('C.Target2','Số lần mua / heavy tail',tail,'Đo số tháng dương; không có số đơn trong nhãn'),
      ('C.Target3','P(mua T|mua T−1) và đối chứng',persistence,'Đã tính đúng consecutive-month'),
      ('C.Trx1','Số dòng và ID active theo tháng',trx_time,'Sửa hai trục về đơn vị thực'),
      ('C.Trx2','Sparsity và khách im lặng lâu',sparsity,'Bổ sung đường 30/84d theo cutoff'),
      ('C.Trx3','R/F/M theo nhóm nhãn',rfm,'Bổ sung decile lift cả R/F/M, giữ ties'),
      ('C.Trx4','Treemap + lift danh mục / loại giao dịch',category,'Bổ sung lift từng event_type'),
      ('C.Trx5','Kiểm tra leakage trước khi có target',leakage,'Bổ sung số sự kiện trước nhãn; có giới hạn availability'),
      ('C.Dialog1','Độ phủ thời gian và khách',dialog_coverage,'Có cả dòng, active ID, cấu trúc'),
      ('C.Dialog2','Metadata frequency / length / hour và lift',dialog_signal,'Bổ sung recency, giờ, weekend; không có độ dài text/duration'),
      ('C.Dialog3','Nội dung/embedding nếu đã thử',dialog_signal,'Điều kiện chưa phát sinh: chưa thử embedding, ghi rõ giới hạn'),
      ('C.Geo1','Độ phủ và mức nhiễu',geo_coverage,'Bổ sung quality proxy mẫu hash ID'),
      ('C.Geo2','Di chuyển / vùng và liên hệ nhãn',geo_signal,'Đo đa dạng/tập trung mã vùng; không có tọa độ thật'),
      ('C.Geo3','Kết luận bổ trợ hay không đáng kể',geo_signal,'Bổ sung AP loss khi bỏ geo'),
      ('D1','Importance / lift theo nguồn / ΔAP ablation',source_signal,'9 fit mới; importance và từng tháng trong phụ lục'),
      ('D2','Baseline vs rule sai ở nhóm nào',error,'Bổ sung B+H 156 biến, đúng model mới nhất'),
      ('D3','Snapshot tháng lịch / T bất kỳ / cửa sổ',snapshot,'12 fit matched-window; T bất kỳ giới hạn bởi nhãn'),
      ('E1','Heatmap mức phù hợp sáu chiều',heatmap,'Đã có; đánh giá định tính theo bằng chứng'),
      ('E2','Rủi ro → xử lý',risk,'Đã có; không gán target thiếu thành âm'),
      ('E3','Mức phù hợp và 2–3 việc tiếp theo',conclusion,'Cập nhật theo đối chứng mới')]
    ledger=[]
    for key,request,s,status in groups:
        def matches(deck):return [i+1 for i,v in enumerate(deck) if v['title']==s['title']]
        main_match=[2] if key in ('A2','A3') else matches(main)
        app_match=matches(appendix)
        if key=='C.Trx3':app_match=[i+1 for i,v in enumerate(appendix) if v['section'].startswith(('PHỤ LỤC · R ·','PHỤ LỤC · M ·'))]
        ledger.append(dict(requirement=key,request=request,status=status,main_slides=main_match,
                           detailed_slides=matches(detailed),appendix_slides=app_match,evidence=s['source']))
    assert len(main)==19 and len(detailed)==23
    assert all(v['main_slides'] or v['detailed_slides'] or v['appendix_slides'] for v in ledger)
    return main,detailed,appendix,ledger
