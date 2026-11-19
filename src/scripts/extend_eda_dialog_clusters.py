"""Past-only dialog clusters, deterministic fold stability and monthly composition."""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import adjusted_rand_score
from complete_eda_figures_moscow import OUT, BASE, publish, INDEX


def model(k):
    return MiniBatchKMeans(n_clusters=k,random_state=42,n_init=5,batch_size=2048,max_iter=100)


def main():
    d=pd.read_parquet(OUT/'dialog_embedding_metadata.parquet')
    z=np.load(OUT/'dialog_past_pca32_scores.npy')
    cols=[f'pc{i}' for i in range(z.shape[1])]
    frame=pd.DataFrame(z,columns=cols)
    frame['client_id']=d.client_id.to_numpy();frame['fold']=d.fold.to_numpy();frame['month']=d.month.to_numpy()
    train=frame[frame.month<pd.Timestamp('2022-01-01')]
    customer=train.groupby(['client_id','fold'])[cols].mean().reset_index()
    x=normalize(customer[cols].to_numpy())
    rows=[];models={}
    for k in [2,4,8,16]:
        print('Dialog cluster k',k,flush=True)
        fit=model(k).fit(x);ref=fit.predict(x);models[k]=fit
        frac=np.bincount(ref,minlength=k)/len(ref)
        for fold in sorted(customer.fold.unique()):
            keep=customer.fold.to_numpy()!=fold
            alt=model(k).fit(x[keep]);pred=alt.predict(x)
            rows.append(dict(k=k,excluded_fold=int(fold),ari_all_reference_customers=adjusted_rand_score(ref,pred),
                             min_cluster_share=float(frac.min()),train_customers=len(x)))
    stable=pd.DataFrame(rows);agg=stable.groupby('k').agg(median_ari=('ari_all_reference_customers','median'),min_share=('min_cluster_share','min')).reset_index()
    eligible=agg[(agg.median_ari>=.7)&(agg.min_share>=.01)]
    if len(eligible):
        best=eligible.median_ari.max();chosen=int(eligible[eligible.median_ari>=best-.02].k.min());status='đạt ngưỡng ổn định đã khai báo'
    else:
        chosen=int(agg.sort_values(['median_ari','k'],ascending=[False,True]).iloc[0].k);status='chỉ thăm dò, không đạt ngưỡng ổn định'
    fit=models[chosen]
    stable.to_csv(OUT/'dialog_cluster_fold_stability.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for fold,q in stable.groupby('excluded_fold'):axes[0].plot(q.k,q.ari_all_reference_customers,'o-',label=f'bỏ fold {fold}')
    axes[0].axhline(.7,color='gray',ls=':');axes[0].set_xlabel('k');axes[0].set_ylabel('ARI so với fit mọi fold');axes[0].legend(fontsize=7)
    axes[1].plot(agg.k,agg.min_share*100,'o-');axes[1].axhline(1,color='gray',ls=':');axes[1].set_ylabel('% cụm nhỏ nhất');axes[1].set_xlabel('k')
    publish('dialog_cluster_stability',fig,agg,'Dialog: độ ổn định cụm theo fold trên lịch sử trước 2022',
        'Dùng PCA32 đã fit sự kiện trước 01/01/2022 Moscow. Trung bình vector theo khách train rồi L2. '
        'MiniBatchKMeans k=2/4/8/16, seed42, n_init5, batch2048. Bỏ lần lượt mỗi fold, fit lại bốn fold, '
        'so ARI trên toàn khách tham chiếu. Chọn k nhỏ nhất trong 0,02 ARI so mức tốt nhất, cần median ARI≥0,7 và cụm nhỏ nhất≥1%.',
        'Đánh giá độ nhạy cấu trúc cụm khi thay khách dùng fit; không đo chất lượng ngữ nghĩa hoặc dự báo mua.',
        f'Chọn k={chosen}: {status}; bảng CSV giữ mọi k và fold, không chỉ phương án được chọn.',
        'ARI đánh giá trên tập tham chiếu gồm cả khách fit nên không phải đánh giá holdout; không thay thế bootstrap cụm. '
        'PCA fit mọi fold lịch sử; kiểm tra này đánh giá bước clustering, chưa đánh giá toàn pipeline khi bỏ fold.')
    monthly=frame.groupby(['client_id','month'])[cols].mean().reset_index()
    mx=normalize(monthly[cols].to_numpy());monthly['cluster']=fit.predict(mx)
    distance=np.linalg.norm(mx-fit.cluster_centers_[monthly.cluster.to_numpy()],axis=1)
    train_dist=np.linalg.norm(x-fit.cluster_centers_[fit.predict(x)],axis=1)
    threshold=float(np.quantile(train_dist,.99));monthly['far_from_reference']=distance>threshold
    monthly['distance']=distance
    labels=pd.read_parquet(BASE/'labels_any.parquet')
    ids=labels.client_id.unique();months=sorted(labels.month.unique())
    panel=pd.MultiIndex.from_product([ids,months],names=['client_id','month']).to_frame(index=False)
    panel=panel.merge(labels,on=['client_id','month'],validate='one_to_one')
    panel=panel.merge(monthly[['client_id','month','cluster','far_from_reference','distance']],on=['client_id','month'],how='left',validate='one_to_one')
    panel['group']=panel.cluster.map(lambda v: f'Cụm {int(v)}' if pd.notna(v) else 'Không thấy dialog')
    # Entire source unavailable in Jan2023, not an absence-of-dialog group.
    panel.loc[panel.month>=pd.Timestamp('2023-01-01'),'group']='Nguồn chưa quan sát'
    summary=panel.groupby(['month','group']).agg(customers=('client_id','size'),buyers=('y','sum')).reset_index()
    summary['purchase_rate']=summary.buyers/summary.customers
    assert summary.groupby('month').customers.sum().eq(100224).all()
    fig,axes=plt.subplots(2,1,figsize=(13,8))
    summary.pivot(index='month',columns='group',values='customers').fillna(0).div(100224).mul(100).plot.bar(stacked=True,ax=axes[0])
    summary.pivot(index='month',columns='group',values='purchase_rate').mul(100).plot(ax=axes[1],marker='o',ms=3)
    axes[0].set_ylabel('% toàn khách');axes[1].set_ylabel('% khách mua cùng tháng');axes[0].legend(fontsize=7,ncol=4);axes[1].legend(fontsize=7,ncol=4)
    publish('dialog_fixed_clusters_monthly_purchase',fig,summary,'Dialog: cơ cấu cụm cố định và mua cùng tháng',
        f'Dùng k={chosen} và hệ PCA/kmeans cố định từ lịch sử trước năm 2022. Mỗi khách–tháng có dialog lấy mean vector rồi L2; '
        'mỗi khách thuộc đúng một nhóm/tháng. Không dialog giữ riêng; 01/2023 ngoài nguồn giữ unavailable. '
        'Tỷ lệ mua dùng nhãn cùng tháng, không phải đánh giá tín hiệu đi trước.',
        'So cơ cấu biểu diễn và liên hệ đồng thời với tháng mua mà không đổi nhãn cụm qua thời gian.',
        f'Mỗi tháng phân hoạch đủ 100.224 khách; k={chosen} {status}. Nhóm cao có thể ít người; CSV có n và số mua.',
        'Nội dung không có text nên không đặt tên chủ đề. Mean tháng có thể gồm dialog sau thời điểm mua chưa biết trong tháng; '
        'không dùng tỷ lệ này như kết quả dự báo, chưa có bootstrap CI hoặc điều chỉnh TRX.')
    far=monthly.groupby('month').agg(customer_months=('client_id','size'),far_share=('far_from_reference','mean'),mean_distance=('distance','mean')).reset_index()
    fig,ax=plt.subplots(figsize=(12,4.5));ax.plot(far.month,far.far_share*100,'o-');ax.set_ylabel('% khách–tháng xa tâm tham chiếu');ax.grid(alpha=.2)
    publish('dialog_fixed_cluster_novelty',fig,far,'Dialog: khách–tháng xa cụm tham chiếu quá khứ',
        f'Khoảng cách mean-L2 khách–tháng đến tâm kmeans gần nhất; ngưỡng cố định P99 khoảng cách mean-L2 khách train={threshold:.5g}. '
        'Giữ nguyên PCA và tâm, không gọi nhãn KMeans là cụm mới.',
        'Đo mức ngoài phân phối tương đối so tập tham chiếu cố định, không khẳng định xuất hiện chủ đề mới.',
        f'Tỷ lệ xa tâm cao nhất {far.far_share.max()*100:.2f}%; CSV ghi n khách–tháng mỗi tháng.',
        'Ngưỡng fit mean toàn lịch sử khách train nhưng áp mean tháng nên có khác biệt độ dài chuỗi; '
        'đây là chỉ báo thăm dò, chưa kết luận drift thuần hoặc độ mới ngữ nghĩa.')
    panel.to_parquet(OUT/'dialog_fixed_cluster_label_panel.parquet',index=False)
    old=pd.read_csv(OUT/'figure_index.csv');idx=pd.concat([old,pd.DataFrame(INDEX)]).drop_duplicates('figure',keep='last');idx.to_csv(OUT/'figure_index.csv',index=False)
    statusj=json.loads((OUT/'execution_status.json').read_text(encoding='utf-8'));statusj['done']+=['dialog_past_clusters_fold_stability','dialog_monthly_fixed_clusters_purchase','dialog_far_from_cluster_reference'];statusj['figures']=len(idx)
    (OUT/'execution_status.json').write_text(json.dumps(statusj,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
