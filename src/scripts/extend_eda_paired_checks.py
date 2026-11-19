"""Paired before/after, client-cluster bootstrap, monthly placebo and sensitivity."""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from complete_eda_figures_moscow import load, OUT, BASE, publish, INDEX


def main():
    ids, months, arrays, y = load()
    inventory = pd.read_csv(BASE/'source_inventory.csv')
    for r in inventory.itertuples():
        if r.first_month_partial: arrays[r.source][:, months.get_loc(pd.Timestamp(r.first_month))] = np.nan
    anchors = np.where(np.isfinite(y).any(axis=0))[0]
    # Each column stores cluster sufficient statistics; all observations of a client travel together.
    stats = []; specs = []; rows = []
    for metric, a in arrays.items():
        for b in [0, 1]:
            sums = np.zeros(len(ids)); counts = np.zeros(len(ids))
            all_delta = []; by_anchor = []
            for t in anchors:
                valid = (y[:, t] == b)
                if t == 0 or t+1 >= a.shape[1]:
                    rows.append(dict(metric=metric, purchase=b, anchor=str(months[t].date()), population=int(valid.sum()), paired=0,
                                     mean_delta=np.nan, median_delta=np.nan, increase=np.nan, decrease=np.nan, unchanged=np.nan))
                    continue
                valid &= np.isfinite(a[:, t-1]) & np.isfinite(a[:, t+1])
                delta = a[:, t+1]-a[:, t-1]
                sums[valid] += delta[valid]; counts[valid] += 1
                v = delta[valid]; all_delta.append(v)
                by_anchor.append(float(v.mean()) if len(v) else np.nan)
                tol = 1e-6*np.maximum(1, np.abs(a[valid, t-1]))
                rows.append(dict(metric=metric, purchase=b, anchor=str(months[t].date()),
                    population=int((y[:, t]==b).sum()), paired=int(valid.sum()),
                    mean_delta=float(v.mean()) if len(v) else np.nan, median_delta=float(np.median(v)) if len(v) else np.nan,
                    increase=float((v>tol).mean()) if len(v) else np.nan,
                    decrease=float((v < -tol).mean()) if len(v) else np.nan,
                    unchanged=float((np.abs(v)<=tol).mean()) if len(v) else np.nan))
            stats.extend([sums, counts]); specs.append((metric, b))
    matrix = np.column_stack(stats)
    rng = np.random.default_rng(42)
    boots = []
    for start in range(0, 2000, 20):
        weights = rng.multinomial(len(ids), np.full(len(ids), 1/len(ids)), size=min(20,2000-start))
        totals = weights @ matrix
        boots.append(np.divide(totals[:, ::2], totals[:, 1::2], out=np.full((len(weights),len(specs)),np.nan), where=totals[:, 1::2]>0))
        if start % 400 == 0: print('Bootstrap', start, flush=True)
    boot = np.vstack(boots)
    summary = []
    for k, (metric, b) in enumerate(specs):
        sums, counts = matrix[:, 2*k], matrix[:, 2*k+1]
        per_client = np.divide(sums, counts, out=np.full(len(ids),np.nan), where=counts>0)
        summary.append(dict(metric=metric, purchase=b, paired_customer_months=int(counts.sum()),
            unique_paired_customers=int((counts>0).sum()),
            paired_mean=float(sums.sum()/counts.sum()),
            low=float(np.nanquantile(boot[:,k], .025)), high=float(np.nanquantile(boot[:,k], .975)),
            customer_equal_mean=float(np.nanmean(per_client))))
    d = pd.DataFrame(rows); s = pd.DataFrame(summary)
    d.to_csv(OUT/'paired_by_anchor.csv', index=False)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, metric in zip(axes.flat, arrays):
        q = s[s.metric==metric]
        ax.errorbar(q.purchase, q.paired_mean, yerr=[q.paired_mean-q.low,q.high-q.paired_mean], fmt='o', capsize=5, label='Trọng số khách–tháng')
        ax.scatter(q.purchase+.07, q.customer_equal_mean, marker='x', label='Mỗi khách cùng trọng số')
        ax.axhline(0,color='gray',ls=':'); ax.set_xticks([0,1], ['Không mua', 'Mua']); ax.set_title(metric)
        ax.set_ylabel('Tháng +1 trừ tháng −1'); ax.legend(fontsize=7)
    insight = '; '.join(f"{r.metric}, mua={r.purchase}: {r.paired_mean:.4g} [{r.low:.4g}; {r.high:.4g}]" for r in s.itertuples())
    publish('paired_before_after_bootstrap', fig, s, 'So trước/sau trên cùng khách và kiểm tra trọng số',
        'Tại mỗi tháng neo, ghép cùng khách có tháng −1 và +1 đủ nguồn; giữ đủ 12 tháng neo trong paired_by_anchor.csv, '
        'không đủ hai phía có paired=0. Ước lượng chính gộp khách–tháng đủ cặp. Bootstrap multinomial 2.000 lượt '
        'trên 100.224 khách, seed42; mọi tháng của cùng khách mang cùng trọng số. CI percentile 95%. '
        'Dấu x cho trung bình chênh lệch mỗi khách, sau đó cân bằng trọng số giữa khách.',
        'Kiểm tra chênh lệch có còn khi giữ cùng tập khách ở hai phía và khi khách mua lặp không có trọng số lớn hơn.',
        insight,
        'Đây chưa phải tác động mua; nhóm mua/không mua còn khác thành phần. CI từng ước lượng chưa hiệu chỉnh đa kiểm định. '
        'Tháng 0 không nằm trong chênh lệch. Không đủ hai phía không bị gán zero.')
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, metric in zip(axes.flat, arrays):
        for b, color in [(0,'#2463a6'),(1,'#d4772b')]:
            q=d[(d.metric==metric)&(d.purchase==b)]
            ax.plot(pd.to_datetime(q.anchor),q.mean_delta,'o-',color=color,label=f'Mua={b}, mean')
            ax.plot(pd.to_datetime(q.anchor),q.median_delta,':',color=color,label=f'Mua={b}, median')
        ax.set_title(metric); ax.axhline(0,color='gray',ls=':'); ax.tick_params(axis='x',rotation=30); ax.legend(fontsize=7)
    publish('paired_placebo_calendar_sensitivity', fig, d, 'Đối chứng tháng neo không mua và độ nhạy mean / median',
        'Mỗi ô giữ mọi tháng nhãn. Tháng neo không mua là nhóm đối chứng lịch cùng kỳ; không chọn ngẫu nhiên vài tháng. '
        'Cùng khách được ghép −1/+1; CSV có n đủ cặp, mean, median và tỷ lệ tăng/giảm/không đổi.',
        'Đối chiếu thay đổi quanh tháng mua với biến động cùng lịch của nhóm không mua, tránh gộp mọi mùa vụ vào mua.',
        'Hình cho phép kiểm tra tính nhất quán dấu theo từng tháng neo. Đọc các đoạn thiếu dữ liệu cùng paired=0; '
        'không khái quát đường trung bình khi mean/median hoặc các tháng khác dấu.',
        'Đối chứng này chưa ghép tầng TRX/độ dài lịch sử; không phải placebo nhân quả. Chưa có CI riêng từng tháng.')
    old=pd.read_csv(OUT/'figure_index.csv')
    idx=pd.concat([old,pd.DataFrame(INDEX)]).drop_duplicates('figure',keep='last');idx.to_csv(OUT/'figure_index.csv',index=False)
    status=json.loads((OUT/'execution_status.json').read_text(encoding='utf-8'))
    status['done']+=['paired_month_minus1_plus1','client_cluster_bootstrap_2000_monthly_metrics','calendar_nonbuyer_mean_median_weight_sensitivity']
    status['figures']=len(idx)
    status['pending']=[x for x in status['pending'] if x != 'client_bootstrap_2000']
    (OUT/'execution_status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__': main()
