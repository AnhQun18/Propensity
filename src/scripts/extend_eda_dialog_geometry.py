"""Exhaustive dialog geometry; no sampled vectors or semantic topic labels."""
import json
import numpy as np
import pandas as pd
import duckdb
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
from complete_eda_figures_moscow import ROOT, OUT, BASE, publish, INDEX


def mean_pair_cosine(x):
    # Exact mean over all DISTINCT unordered pairs, no quadratic matrix.
    z = normalize(x)
    n = len(z)
    sums = z.sum(axis=0, dtype=np.float64)
    return float((sums @ sums - n)/(n*(n-1))) if n > 1 else np.nan


def main():
    c = duckdb.connect(); c.execute('SET threads=2')
    d = c.execute(f"SELECT client_id,event_time,fold,embedding FROM read_parquet('{ROOT.as_posix()}/data/MBD-mini/detail/dialog/**/*.parquet', hive_partitioning=true)").df()
    c.close()
    x = np.stack(d.pop('embedding').to_numpy()).astype(np.float32)
    assert x.shape == (217205, 768) and np.isfinite(x).all()
    d['month'] = (pd.to_datetime(d.event_time)+pd.Timedelta(hours=3)).dt.to_period('M').dt.to_timestamp()
    norms = np.linalg.norm(x, axis=1)
    assert (norms > 0).all()
    variance = x.var(axis=0, dtype=np.float64)
    dim = pd.DataFrame({'dimension': np.arange(768), 'mean': x.mean(axis=0), 'variance': variance})
    order = np.argsort(variance)[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(np.arange(1, 769), variance[order]); axes[0].set_yscale('log')
    axes[0].set_xlabel('Thứ hạng chiều theo phương sai'); axes[0].set_ylabel('Phương sai')
    axes[1].plot(np.arange(1, 769), np.cumsum(variance[order])/variance.sum()*100)
    axes[1].set_xlabel('Số chiều theo thứ hạng'); axes[1].set_ylabel('% tổng phương sai theo trục gốc')
    top10 = float(variance[order[:10]].sum()/variance.sum()*100)
    publish('dialog_dimension_variance', fig, dim, 'Dialog: phương sai 768 chiều gốc',
        'Đọc toàn bộ 217.205 embedding float32, tính phương sai từng chiều bằng tích lũy float64; thứ hạng không dùng nhãn.',
        'Đo mức phân bố phương sai giữa các trục gốc; chưa phải số chiều độc lập hoặc PCA.',
        f'10 chiều phương sai cao nhất chiếm {top10:.2f}% tổng phương sai theo trục gốc. '
        'Không tự kết luận các chiều đó vô ích hoặc phải loại.',
        'Phương sai lớn không đồng nghĩa thông tin ngữ nghĩa hoặc quan hệ mua mạnh; toàn kỳ hồi cứu, không phải fit trước mua.')
    # Exact duplicate byte vectors, retaining all original records.
    keys = np.ascontiguousarray(x).view(np.dtype((np.void, x.dtype.itemsize*x.shape[1]))).ravel()
    _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    duplicate = pd.DataFrame({'vector_group': inverse, 'client_id': d.client_id})
    groups = duplicate.groupby('vector_group').agg(events=('client_id', 'size'), customers=('client_id', 'nunique')).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(groups.events, bins=40, log=True); axes[0].set_xlabel('Số dòng / vector chính xác'); axes[0].set_ylabel('Số vector, log')
    axes[1].hist(groups.customers, bins=40, log=True); axes[1].set_xlabel('Số khách / vector chính xác')
    repeated = int(counts[counts>1].sum())
    publish('dialog_exact_vector_duplicates', fig, groups, 'Dialog: vector trùng chính xác giữa các dòng và khách',
        'So toàn bộ byte của 768 float32; không làm tròn, không gộp các dòng nguồn. Thống kê số dòng và khách riêng biệt theo vector.',
        'Phân biệt trùng embedding với trùng timestamp hoặc trùng toàn dòng.',
        f'{len(counts):,} vector byte khác nhau; {repeated:,}/{len(x):,} dòng thuộc nhóm vector lặp; '
        f'nhóm lớn nhất có {int(counts.max()):,} dòng và tối đa {int(groups.customers.max()):,} khách.',
        'Trùng byte không xác nhận cùng văn bản, người hỗ trợ hoặc tin tự động. Chưa đánh giá gần trùng theo cosine.')
    del keys, inverse, duplicate
    centered = x-x.mean(axis=0)
    pca = PCA(n_components=64, svd_solver='randomized', random_state=42)
    scores = pca.fit_transform(centered)
    explained = np.cumsum(pca.explained_variance_ratio_)
    dim90 = int(np.searchsorted(explained, .9)+1) if explained[-1]>=.9 else None
    geo = pd.DataFrame({'components': np.arange(1, 65), 'cumulative_variance': explained})
    cosine_raw = mean_pair_cosine(x)
    cosine_center = mean_pair_cosine(centered)
    white = scores/np.sqrt(pca.explained_variance_)
    cosine_white = mean_pair_cosine(white)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(geo.components, explained*100); axes[0].axhline(90, color='gray', ls='--')
    axes[0].set_xlabel('Số thành phần PCA'); axes[0].set_ylabel('% phương sai tích lũy')
    axes[1].bar(['raw-L2', 'centered-L2', 'PCA64 whiten-L2'], [cosine_raw, cosine_center, cosine_white])
    axes[1].set_ylabel('Cosine trung bình tất cả cặp khác dòng'); axes[1].tick_params(axis='x', labelsize=8)
    publish('dialog_pca_anisotropy', fig, geo, 'Dialog: PCA và anisotropy trên toàn bộ embedding',
        'PCA randomized 64 chiều, seed 42, fit hồi cứu toàn bộ vector. Cosine trung bình mọi cặp khác dòng '
        'tính chính xác bằng (||sum(vector_L2)||² − n)/(n(n−1)), không lấy mẫu cặp. Whitening chia theo căn phương sai PCA.',
        'Đo mức hướng chung và độ tập trung phương sai; so sánh hình học sau các phép biến đổi.',
        f'Cosine trung bình raw={cosine_raw:.4f}, centered={cosine_center:.4f}, whitened={cosine_white:.4f}. '
        f'64 PCA giữ {explained[-1]*100:.2f}% phương sai; số chiều đạt 90% trong giới hạn 64: {dim90}.',
        'Centered cosine gần zero một phần là hệ quả toán học của trừ tâm, không chứng minh chất lượng tốt hơn. '
        'Whitening có thể khuếch đại hướng nhiễu. Toàn kỳ hồi cứu; không dùng biểu diễn này làm feature trước mua.')
    # Fixed past-only reference for monthly drift; no future fit.
    train = (d.month < pd.Timestamp('2022-01-01')).to_numpy()
    temporal = PCA(n_components=32, svd_solver='randomized', random_state=42)
    temporal.fit(x[train]); z = temporal.transform(x)
    np.save(OUT/'dialog_past_pca32_scores.npy', z.astype(np.float32))
    d[['client_id','event_time','fold','month']].to_parquet(OUT/'dialog_embedding_metadata.parquet',index=False)
    reference = z[train].mean(axis=0); scale = np.sqrt(temporal.explained_variance_)
    drift = []
    for (month, fold), g in d.groupby(['month', 'fold']):
        ix = g.index.to_numpy()
        drift.append(dict(month=str(month.date()), fold=int(fold), rows=len(ix), customers=g.client_id.nunique(),
            mean_norm=float(norms[ix].mean()), centroid_distance=float(np.linalg.norm((z[ix].mean(axis=0)-reference)/scale))))
    drift = pd.DataFrame(drift)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for fold, g in drift.groupby('fold'):
        for ax, col in zip(axes, ['mean_norm', 'centroid_distance']): ax.plot(pd.to_datetime(g.month), g[col], 'o-', ms=3, label=f'fold {fold}')
    axes[0].set_ylabel('Norm trung bình'); axes[1].set_ylabel('Khoảng cách tâm / SD PCA train')
    axes[0].legend(ncol=5, fontsize=8)
    publish('dialog_fixed_reference_drift', fig, drift, 'Dialog: độ trôi theo tháng trên hệ PCA quá khứ cố định',
        'PCA32 fit chỉ sự kiện trước 01/01/2022 00:00 Moscow; mọi tháng transform cùng hệ trục. '
        'Khoảng cách Euclid của tâm tháng–fold tới tâm train, chia theo SD PCA train. Không sử dụng nhãn.',
        'So dịch chuyển hình học và norm giữa thời gian/fold mà không đổi hệ trục.',
        f'Tập tham chiếu gồm {int(train.sum()):,} sự kiện trước năm 2022; CSV giữ số dòng/khách của từng tháng–fold. '
        f'Khoảng cách tâm cao nhất {drift.centroid_distance.max():.3f} đơn vị chuẩn hóa.',
        'Tháng đầu thiếu ngày; tâm có thể đổi do cơ cấu khách, mức ghi nhận hoặc biểu diễn. Không chứng minh đổi nội dung hay hệ thống. '
        'Chưa kiểm định MMD/PSI, chưa kiểm soát trọng số khách.')
    old = pd.read_csv(OUT/'figure_index.csv')
    idx = pd.concat([old, pd.DataFrame(INDEX)]).drop_duplicates('figure', keep='last')
    idx.to_csv(OUT/'figure_index.csv', index=False)
    status = json.loads((OUT/'execution_status.json').read_text(encoding='utf-8'))
    status['done'] += ['dialog_all_vector_variance_exact_duplicates_pca_anisotropy', 'dialog_past_only_fixed_pca_drift']
    status['figures'] = len(idx)
    (OUT/'execution_status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__': main()
