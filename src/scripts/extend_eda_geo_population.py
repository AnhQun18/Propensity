"""All-history categorical GEO profiles, IDF, sparse graph and SVD.

Retrospective source description only; not a past-cutoff feature table.
"""
import json
import numpy as np
import pandas as pd
import duckdb
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.sparse import coo_matrix, bmat
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from complete_eda_figures_moscow import ROOT, OUT, BASE, publish, INDEX


def main():
    ids=np.sort(pd.read_parquet(BASE/'labels_any.parquet').client_id.unique())
    c=duckdb.connect();c.execute("SET memory_limit='3GB'");c.execute('SET threads=2')
    c.execute(f"SET temp_directory='{OUT.as_posix()}/tmp_geo'")
    path=(ROOT/'data/MBD-mini/detail/geo/**/*.parquet').as_posix()
    profiles=[]; edges=None
    for level in [4,5,6]:
        print('GEO population level',level,flush=True)
        cache=OUT/f'geo_g{level}_customer_cells.parquet'
        if not cache.exists():
            c.execute(f"COPY (SELECT client_id,geohash_{level} cell,count(*) events FROM read_parquet('{path}') GROUP BY 1,2) TO '{cache.as_posix()}' (FORMAT PARQUET)")
        profile=c.execute(f"""WITH cells AS (SELECT *,sum(events) OVER(PARTITION BY client_id) total FROM read_parquet('{cache.as_posix()}'))
            SELECT client_id,{level} level,max(total) events,count(*) unique_cells,
              max(events)::DOUBLE/max(total) top_share,
              -sum((events::DOUBLE/total)*ln(events::DOUBLE/total)) entropy,
              sum(pow(events::DOUBLE/total,2)) hhi
              FROM cells GROUP BY 1""").df()
        assert int(profile.events.sum())==66295724
        profiles.append(profile)
        if level==4: edges=c.execute(f"SELECT * FROM read_parquet('{cache.as_posix()}')").df()
    c.close()
    p=pd.concat(profiles,ignore_index=True)
    p.to_parquet(OUT/'geo_all_history_customer_profiles.parquet',index=False)
    fig,axes=plt.subplots(3,3,figsize=(13,11))
    for row,level in enumerate([4,5,6]):
        q=p[p.level==level]
        axes[row,0].hist(q.top_share,bins=40);axes[row,0].set_xlabel(f'g{level}: top-1 share')
        axes[row,1].hist(q.entropy,bins=40);axes[row,1].set_xlabel(f'g{level}: entropy, nats')
        axes[row,2].hist2d(np.log1p(q.events),np.log1p(q.unique_cells),bins=50,norm=LogNorm())
        axes[row,2].set_xlabel('log1p số sự kiện');axes[row,2].set_ylabel('log1p số ô')
    summary=p.groupby('level').agg(customers=('client_id','size'),median_cells=('unique_cells','median'),median_entropy=('entropy','median'),median_hhi=('hhi','median'),median_topshare=('top_share','median')).reset_index()
    summary['no_geo_customers']=len(ids)-summary.customers
    publish('geo_concentration_all_levels',fig,summary,'GEO: đa dạng và tập trung ở cả ba cấp',
        'Toàn bộ 66.295.724 sự kiện, nhóm khách–ô ở từng cấp; p=count ô/tổng sự kiện khách. '
        'Entropy=−sum(p ln p), HHI=sum(p²). Histogram chỉ khách có GEO; CSV ghi rõ khách không GEO, không gán entropy=0 cho họ. '
        'Bảng cá thể đầy đủ có trong geo_all_history_customer_profiles.parquet.',
        'So tính tập trung giữa cấp mã và liên hệ cơ học giữa số sự kiện với số ô.',
        '; '.join(f'g{r.level}: median số ô={r.median_cells:.0f}, top-share={r.median_topshare:.3f}, HHI={r.median_hhi:.3f}' for r in summary.itertuples()),
        'Toàn lịch sử nguồn dùng để mô tả, không dùng như feature trước mua. Quan hệ count–số ô không chứng minh di chuyển. '
        'Các histogram không gồm khách chưa có GEO; số đó được ghi rõ ở CSV.')
    cells=np.sort(edges.cell.unique());ci=pd.Index(ids).get_indexer(edges.client_id);gi=pd.Index(cells).get_indexer(edges.cell)
    mat=coo_matrix((edges.events.to_numpy(float),(ci,gi)),shape=(len(ids),len(cells))).tocsr()
    degree=np.asarray((mat>0).sum(axis=1)).ravel()
    cell_degree=np.asarray((mat>0).sum(axis=0)).ravel()
    idf=np.log((len(ids)+1)/(cell_degree+1))+1
    total=np.asarray(mat.sum(axis=1)).ravel()
    mean_idf=np.divide(mat@idf,total,out=np.full(len(ids),np.nan),where=total>0)
    g=bmat([[None,(mat>0).astype(np.int8)],[(mat.T>0).astype(np.int8),None]],format='csr')
    nc,component=connected_components(g,directed=False)
    counts=np.bincount(component); customers_component=np.bincount(component[:len(ids)],minlength=nc)
    out=pd.DataFrame({'client_id':ids,'g4_degree':degree,'event_weighted_idf':mean_idf,
                      'component':component[:len(ids)],'component_nodes':counts[component[:len(ids)]],
                      'component_customers':customers_component[component[:len(ids)]]})
    out.to_parquet(OUT/'geo_g4_graph_idf_customers.parquet',index=False)
    fig,axes=plt.subplots(1,3,figsize=(14,4.5))
    axes[0].hist(mean_idf[np.isfinite(mean_idf)],bins=50);axes[0].set_xlabel('IDF trung bình trọng số sự kiện')
    axes[1].hist(np.log1p(degree),bins=50);axes[1].set_xlabel('log1p bậc g4 / toàn bộ khách')
    axes[2].hist(np.log1p(customers_component[customers_component>0]),bins=50,log=True)
    axes[2].set_xlabel('log1p số khách / thành phần');axes[2].set_ylabel('Số thành phần, log')
    comps=pd.DataFrame({'component':np.arange(nc),'nodes':counts,'customers':customers_component})
    publish('geo_g4_idf_graph',fig,comps,'GEO g4: độ hiếm và thành phần đồ thị khách–ô',
        'Đồ thị hai phía có mọi khách và mọi g4; cạnh khi khách từng có sự kiện trong ô. '
        'IDF=ln((100224+1)/(số khách ô+1))+1, trung bình theo tần suất sự kiện. '
        'Connected components bằng scipy sparse; khách không GEO là nút cô lập.',
        'Mô tả độ hiếm ô và mức kết nối khách qua các ô dùng chung, không suy quan hệ doanh nghiệp.',
        f'{nc:,} thành phần, {int((degree==0).sum()):,} khách cô lập; thành phần nhiều khách nhất có {int(customers_component.max()):,} khách.',
        'Ô phổ biến có thể nối phần lớn khách; cùng thành phần không chứng minh cùng địa điểm hay tập đoàn. '
        'Chưa tính nút khớp/cặp hiếm; IDF toàn kỳ không phải thống kê có sẵn ở cutoff quá khứ.')
    tfidf=TfidfTransformer().fit_transform(mat)
    svd=TruncatedSVD(n_components=32,random_state=42)
    z=svd.fit_transform(tfidf)
    np.savez_compressed(OUT/'geo_g4_bag_svd32.npz',client_ids=ids.astype(str),embedding=z.astype(np.float32))
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    active=degree>0
    axes[0].hist2d(z[active,0],z[active,1],bins=60,norm=LogNorm());axes[0].set_xlabel('SVD1');axes[0].set_ylabel('SVD2')
    axes[1].plot(np.arange(1,33),np.cumsum(svd.explained_variance_ratio_)*100);axes[1].set_xlabel('Số chiều');axes[1].set_ylabel('% phương sai tích lũy')
    var=pd.DataFrame({'dimension':np.arange(1,33),'variance_ratio':svd.explained_variance_ratio_})
    publish('geo_g4_bag_svd',fig,var,'GEO g4: biểu diễn bag-of-cells TF-IDF và SVD',
        'Ma trận thưa toàn bộ khách × g4, trọng số số sự kiện; TF-IDF mặc định sklearn (smooth IDF, L2), SVD32 seed42. '
        'Hình mật độ giữ mọi khách có GEO; khách không GEO có vector zero và mask qua bậc đồ thị.',
        'Quan sát cấu trúc đồng xuất hiện ô giữa khách; không coi trục SVD là tọa độ địa lý.',
        f'32 chiều giữ {svd.explained_variance_ratio_.sum()*100:.2f}% phương sai theo định nghĩa TruncatedSVD; '
        f'{int(active.sum()):,} khách có GEO và {int((~active).sum()):,} khách vector zero vì không nguồn.',
        'Fit hồi cứu toàn kỳ, chưa đánh giá ổn định theo thời gian hoặc quan hệ nhãn. Không thay thế cell2vec chuỗi.')
    old=pd.read_csv(OUT/'figure_index.csv');idx=pd.concat([old,pd.DataFrame(INDEX)]).drop_duplicates('figure',keep='last');idx.to_csv(OUT/'figure_index.csv',index=False)
    status=json.loads((OUT/'execution_status.json').read_text(encoding='utf-8'))
    status['done']+=['geo_all_level_entropy_hhi','geo_g4_idf_sparse_components','geo_g4_bag_tfidf_svd32']
    status['figures']=len(idx)
    (OUT/'execution_status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
