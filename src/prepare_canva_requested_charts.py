"""Evidence and presentation assets for the requested in-place Canva revision."""
from pathlib import Path
import base64
import csv
import html
import json
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter
from matplotlib import dates as mdates

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/behavior_coverage/canva_requested_revision"
COLORS = ["#267969", "#db8e6e", "#8c7cb0", "#76a583"]
PAPER = "#f6f4ec"
INK = "#173d38"


def vi(n): return f"{int(n):,}".replace(",", ".")


def read(section, name):
    p = ROOT / "docs/mbd_mini_eda/tables" / section / f"{name}.csv"
    return list(csv.DictReader(p.open(encoding="utf-8-sig")))


def clean_axes(ax):
    ax.set_facecolor(PAPER)
    for name in ["top", "right"]: ax.spines[name].set_visible(False)
    ax.spines["left"].set_color("#bac9be"); ax.spines["bottom"].set_color("#bac9be")
    ax.tick_params(colors="#526963", labelsize=11)
    ax.grid(axis="y", color="#d6ded4", alpha=.6, linewidth=.7)
    ax.set_axisbelow(True)


def scan_year():
    cache = OUT / "daily_and_customer_2022.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            return {k: data[k] for k in data.files}
    started = time.perf_counter()
    base = ROOT / "data/MBD-mini"
    clients = pq.read_table(base / "client_split", columns=["client_id"])["client_id"].combine_chunks()
    clients = pc.take(clients, pc.sort_indices(clients))
    counts = np.zeros((3, 365), dtype=np.uint64)
    customer_totals = np.zeros((len(clients), 3), dtype=np.uint64)
    amount_parts = []
    scan_rows = []
    source_totals = []
    for source_index, source in enumerate(["trx", "geo", "dialog"]):
        files = sorted((base / "detail" / source).rglob("*.parquet"))
        rows = 0
        cols = ["client_id", "event_time"] + (["amount"] if source == "trx" else [])
        for fno, path in enumerate(files, 1):
            reader = pq.ParquetFile(path)
            if reader.schema_arrow.field("event_time").type.tz is not None:
                raise ValueError("Source timezone changed")
            for batch in reader.iter_batches(batch_size=150_000, columns=cols):
                rows += len(batch)
                pos = pc.fill_null(pc.index_in(batch.column(0), value_set=clients), -1).to_numpy()
                stamps = batch.column(1).to_numpy(zero_copy_only=False).astype("datetime64[ns]")
                keep = ((pos >= 0) & ~np.isnat(stamps) & (stamps >= np.datetime64("2022-01-01"))
                        & (stamps < np.datetime64("2023-01-01")))
                days = (stamps[keep].astype("datetime64[D]") - np.datetime64("2022-01-01")).astype(int)
                counts[source_index] += np.bincount(days, minlength=365).astype(np.uint64)
                np.add.at(customer_totals[:, source_index], pos[keep], 1)
                if source == "trx":
                    amount_parts.append(batch.column(2).to_numpy(zero_copy_only=False)[keep].copy())
            if fno % 100 == 0 or fno == len(files):
                print(f"{source}: {fno}/{len(files)} files; {rows:,} raw rows", flush=True)
        scan_rows.append(rows)
        source_totals.append(int(counts[source_index].sum()))
    amounts = np.concatenate(amount_parts)
    assert scan_rows == [38764930, 66295724, 217205]
    assert source_totals == [19376661, 65564488, 118745]
    assert int(counts.sum()) == 85059894 == int(customer_totals.sum())
    assert len(amounts) == source_totals[0] and np.isfinite(amounts).all()
    result = dict(daily_counts=counts, customer_totals=customer_totals, amounts=amounts,
                  scanned_rows=np.array(scan_rows), elapsed_seconds=np.array(time.perf_counter()-started))
    np.savez_compressed(cache, **result)
    dates = np.arange("2022-01-01", "2023-01-01", dtype="datetime64[D]")
    with (OUT / "daily_events_2022.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["date", "trx_events", "geo_events", "dialog_events", "total_events"])
        for i, day in enumerate(dates):
            writer.writerow([str(day), *counts[:, i].tolist(), int(counts[:, i].sum())])
    print(f"Daily scan completed in {time.perf_counter()-started:.1f}s", flush=True)
    return result


def matrix_chart():
    with np.load(ROOT / "outputs/behavior_coverage/matrix.npz", allow_pickle=False) as data:
        months = data["months"]; counts = data["counts"]
        ids = data["client_ids"]
    keep = (months >= "2022-01") & (months < "2023-01")
    selected = np.random.default_rng(42).choice(len(counts), 80, replace=False)
    values = counts[selected][:, keep].T
    assert values.shape == (12, 80)
    # Fixed ratio is an explicit requirement: one source cell is one square.
    fig = Figure(figsize=(17.92, 6.8), dpi=180, facecolor=PAPER)
    ax = fig.add_axes([.065, .27, .855, .63])
    image = ax.imshow(values, aspect="equal", cmap="Greens", vmin=0, vmax=max(1,int(values.max())), interpolation="nearest")
    ax.set_yticks(np.arange(12), [f"{m[5:]}/2022" for m in months[keep]], fontsize=10)
    ax.set_xticks(np.arange(0, 80, 5), [str(i+1) for i in range(0,80,5)], fontsize=11)
    ax.set_xlabel("Khách hàng trong mẫu (1–80)", fontsize=13, labelpad=14, color=INK)
    ax.set_ylabel("Tháng", fontsize=13, labelpad=14, color=INK)
    ax.set_xticks(np.arange(81)-.5, minor=True); ax.set_yticks(np.arange(13)-.5, minor=True)
    ax.grid(which="minor", color="#ffffff", linewidth=.3, alpha=.5)
    ax.tick_params(which="both", length=0, pad=6)
    for spine in ax.spines.values(): spine.set_visible(False)
    # Match the actual aspect-constrained plot position for a non-stretched colorbar.
    pos = ax.get_position()
    cax = fig.add_axes([.94, pos.y0, .012, pos.height])
    cbar = fig.colorbar(image, cax=cax)
    cbar.set_label("Tổng log / ô", fontsize=11); cbar.ax.tick_params(labelsize=10)
    fig.text(.49, .18, "80 khách ngẫu nhiên, không lặp · seed=42 · mỗi ô là một khách–tháng", ha="center", fontsize=12, color="#526963")
    fig.text(.49, .12, "Ô vuông được giữ đúng tỷ lệ; 0 là không ghi nhận log, không phải null.", ha="center", fontsize=12, color="#526963")
    fig.savefig(OUT / "matrix_month_by_customer.png", dpi=180, facecolor=PAPER)
    with (OUT / "matrix_sample_80.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer=csv.writer(handle);writer.writerow(["sample_column", "client_id", *months[keep]])
        for col, index in enumerate(selected):writer.writerow([col+1, ids[index], *values[:,col].tolist()])
    (OUT / "matrix_geometry.json").write_text(json.dumps({"months":12,"customers":80,
        "aspect":ax.get_aspect(),"plot_width_inches":pos.width*fig.get_figwidth(),
        "plot_height_inches":pos.height*fig.get_figheight(),"color_max":int(values.max()),
        "sample_seed":42,"policy":"known ID and valid timestamp, missing other attributes retained"},indent=2),encoding="utf-8")


def target_chart():
    records = read("descriptive.targets", "target_by_month_fold")
    months = sorted({r["mon"][:7] for r in records})
    fig = Figure(figsize=(16.96, 5.65), dpi=180, facecolor=PAPER)
    ax = fig.subplots();clean_axes(ax)
    output = []
    for target_i in range(1,5):
        target=f"target_{target_i}"; rates=[]
        for month in months:
            rows=[r for r in records if r["mon"].startswith(month) and r["target"]==target]
            n=sum(int(r["labelled_n"]) for r in rows);p=sum(int(r["positive_n"]) for r in rows)
            assert n==100224
            rates.append(p/n*100);output.append([month,target,p,n,p/n*100])
        ax.plot(range(12),rates,label=target,color=COLORS[target_i-1],linewidth=2.6,marker="o",markersize=6)
    ax.set_xticks(range(12), [m[5:]+("/2023" if m.startswith("2023") else "/2022") for m in months],fontsize=11)
    ax.set_ylim(bottom=0);ax.yaxis.set_major_formatter(FuncFormatter(lambda x,_:f"{x:.1f}%"))
    ax.set_ylabel("Tỷ lệ khách hàng có nhãn dương",fontsize=13,color=INK,labelpad=15)
    ax.set_xlabel("Tháng nhãn",fontsize=13,color=INK,labelpad=12)
    ax.legend(ncol=4,frameon=False,fontsize=13,loc="upper center",bbox_to_anchor=(.5,1.12))
    fig.subplots_adjust(left=.075,right=.98,bottom=.18,top=.85)
    fig.savefig(OUT/"target_four_lines.png",dpi=180,facecolor=PAPER)
    with (OUT/"target_rates_monthly.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.writer(handle);writer.writerow(["month","target","positive_n","labelled_n","positive_percent"]);writer.writerows(output)


def daily_chart(result):
    counts=result["daily_counts"]
    dates=np.arange("2022-01-01","2023-01-01",dtype="datetime64[D]")
    fig=Figure(figsize=(16.9,5.6),dpi=180,facecolor=PAPER)
    ax=fig.subplots();clean_axes(ax)
    for i,name in enumerate(["Giao dịch","Vị trí","Hội thoại"]):
        # Keep zero separate from log scale; no synthetic log is added to raw data.
        ax.plot(dates,np.where(counts[i]>0,counts[i],np.nan),label=name,color=COLORS[i],linewidth=1.3,alpha=.9)
    ax.plot(dates,counts.sum(axis=0),label="Tổng ba nguồn",color="#173d38",linewidth=1.2,linestyle="--",alpha=.65)
    ax.set_yscale("log");ax.yaxis.set_major_formatter(FuncFormatter(lambda x,_:f"{x:,.0f}".replace(",",".")))
    ax.xaxis.set_major_locator(mdates.MonthLocator());ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
    ax.set_xlim(np.datetime64("2022-01-01"),np.datetime64("2022-12-31"))
    ax.set_ylabel("Số log / ngày · trục logarit",fontsize=13,color=INK,labelpad=12)
    ax.legend(ncol=4,frameon=False,fontsize=12,loc="upper center",bbox_to_anchor=(.5,1.12))
    fig.subplots_adjust(left=.085,right=.98,bottom=.16,top=.85)
    fig.savefig(OUT/"daily_events_2022.png",dpi=180,facecolor=PAPER)


def stats_pages(result):
    totals=result["customer_totals"]
    names=["Giao dịch","Vị trí","Hội thoại"]
    customer_stats=[]
    for i,name in enumerate(names):
        v=totals[:,i]
        customer_stats.append({"source":name,"mean":float(v.mean()),"p25":float(np.quantile(v,.25)),
             "median":float(np.median(v)),"p75":float(np.quantile(v,.75)),"p95":float(np.quantile(v,.95)),
             "max":int(v.max()),"customers_with_logs":int((v>0).sum()),"population":len(v)})
    amounts=result["amounts"]
    q=np.quantile(amounts,[.25,.5,.75,.95,.99])
    amount_stats={"n":len(amounts),"mean":float(amounts.mean(dtype=np.float64)),"min":float(amounts.min()),
                  "max":float(amounts.max()),**dict(zip(["p25","median","p75","p95","p99"],map(float,q)))}
    fig=Figure(figsize=(10.5,4.6),dpi=180,facecolor=PAPER)
    ax=fig.subplots();clean_axes(ax)
    ax.hist(np.log10(1+amounts.astype(np.float64)),bins=45,color=COLORS[0],edgecolor=PAPER,linewidth=.6)
    ax.set_xlabel("log10(1 + amount gốc)",fontsize=12,color=INK)
    ax.set_ylabel("Số giao dịch",fontsize=12,color=INK)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x,_:f"{x/1e6:g} triệu"))
    fig.subplots_adjust(left=.13,right=.98,bottom=.19,top=.94)
    fig.savefig(OUT/"amount_distribution_2022.png",dpi=180,facecolor=PAPER)
    (OUT/"descriptive_stats_2022.json").write_text(json.dumps({"customer_counts":customer_stats,"amount":amount_stats},indent=2,ensure_ascii=False),encoding="utf-8")
    return customer_stats,amount_stats


def build_new_pages(result,customer_stats,amount_stats):
    def img(name,cls=""):
        data=base64.b64encode((OUT/name).read_bytes()).decode("ascii")
        return f'<img class="{cls}" src="data:image/png;base64,{data}"/>'
    pages=[]
    def page(title,subtitle,body,notes):
        pages.append(f'<section data-document-role="page" data-label="{html.escape(title)}" data-speaker-notes="{html.escape(notes,quote=True)}"><p class="brand">MBD / PROPENSITY LAB</p><h1>{title}</h1><p class="sub">{subtitle}</p>{body}<footer>Nguồn: tổng hợp detail, lịch sử [01/01/2022, 01/01/2023) · timestamp theo lịch lưu trong dữ liệu</footer></section>')
    counts=result["daily_counts"]
    dates=np.arange("2022-01-01","2023-01-01",dtype="datetime64[D]")
    total=counts.sum(axis=0);peak=int(np.argmax(total))
    page("Hành vi ghi nhận theo ngày trong năm", "365 ngày của năm 2022 · số sự kiện, không phải số khách hàng duy nhất",
         img("daily_events_2022.png","daily")+f'<p class="note">85.059.894 log trong năm · cao nhất {str(dates[peak])}: {vi(total[peak])} log</p>',
         "Đọc toàn bộ client_id và event_time của ba nguồn detail; chỉ giữ ID thuộc client_split, timestamp hợp lệ, trong năm 2022. "
         "Số log là tổng giao dịch, vị trí và hội thoại, không cộng ptls, không yêu cầu đầy đủ categorical. "
         "Trục Y logarit để ba nguồn chênh lệch quy mô vẫn nhìn thấy; đường đứt là tổng ba nguồn. Giá trị 0 không đặt trên trục log. "
         f"Số ngày có zero theo nguồn: {[(counts[i]==0).sum().item() for i in range(3)]}. "
         "Nguồn có thể ghi một phần ngày cuối 31/12/2022; không suy diễn thu thập liên tục của từng khách. "
         "Tổng theo ngày đối chiếu đúng 85.059.894, và tổng theo nguồn khớp monthly_activity.csv. File số liệu daily_events_2022.csv.")
    body='<table><thead><tr><th>Số log / khách</th><th>Trung bình</th><th>P25</th><th>Median</th><th>P75</th><th>P95</th></tr></thead><tbody>'
    for r in customer_stats:
        mean_label=f'{r["mean"]:.1f}'.replace('.',',')
        body+=f'<tr><td>{r["source"]}</td><td>{mean_label}</td><td>{vi(r["p25"])}</td><td>{vi(r["median"])}</td><td>{vi(r["p75"])}</td><td>{vi(r["p95"])}</td></tr>'
    body+='</tbody></table><div class="mini-grid">'
    for r in customer_stats:
        body+=f'<div><strong>{r["customers_with_logs"]/r["population"]*100:.1f}%</strong><p>khách có log {r["source"].lower()}</p></div>'
    body+='</div><p class="note">Cùng mẫu số 100.224 khách, gồm cả khách không có log.</p>'
    page("Thống kê mô tả hoạt động khách hàng", "Tổng log trên mỗi khách trong 12 tháng 2022 · P25 / P75 là tứ phân vị",body,
         "Tính tổng sự kiện mỗi khách trong năm 2022 riêng theo nguồn, rồi lấy mean và các phân vị trên toàn bộ 100.224 khách gồm số 0. "
         "Các phân vị dùng numpy.quantile mặc định tuyến tính. Không cộng khách giữa các nguồn. Đây là số log năm 2022, khác với EDA toàn lịch sử. "
         "Nguồn: daily_and_customer_2022.npz và descriptive_stats_2022.json, đối chiếu tổng nguồn với monthly_activity.csv.")
    a=amount_stats
    body='<div class="amount-layout">'+img("amount_distribution_2022.png")+'<div class="numbers">'
    for label,key in [("Trung bình","mean"),("Median","median"),("P95","p95"),("P99","p99")]:
        body+=f'<div><span>{label}</span><strong>{vi(round(a[key]))}</strong></div>'
    body+='</div></div><p class="note">Amount là giá trị gốc, trộn mã currency chưa quy đổi; không gắn đơn vị VND.</p>'
    page("Thống kê mô tả giá trị giao dịch",f'{vi(a["n"])} amount hữu hạn trong năm 2022 · histogram trên thang log',body,
         "Tính chính xác các phân vị trên toàn bộ amount hữu hạn của các giao dịch có ID/timestamp hợp lệ trong 2022; không lấy mẫu. "
         "Histogram hiển thị log10(1+amount); số thống kê bên phải ở thang amount gốc. Các thống kê trên slide làm tròn số nguyên. "
         "Chưa diễn giải hoặc quy đổi currency, không xem là VND hoặc tổng sức mua quy đổi. Không biến đổi dữ liệu nguồn. "
         "Nguồn: detail/trx đọc amount trong cùng lần quét, descriptive_stats_2022.json.")
    css='''*{box-sizing:border-box}html,body{margin:0;width:1920px;background:#f6f4ec;font-family:Arial,sans-serif;color:#173d38}section{width:1920px;height:1080px;position:relative;padding:76px 112px}.brand{font-size:18px;letter-spacing:3px;color:#526963;margin:0 0 40px}h1{font-size:60px;letter-spacing:-1.8px;line-height:1.15;margin:0 0 22px;font-weight:600}.sub{font-size:27px;color:#526963;margin:0 0 26px;line-height:1.5}.daily{display:block;width:1696px;height:562px;object-fit:contain}.note{font-size:25px;line-height:1.5;padding:22px 28px;border-radius:22px;background:#e6eee6;margin:25px 0 0}footer{position:absolute;bottom:42px;left:112px;right:112px;border-top:1px solid #d6ded4;padding-top:18px;font-size:17px;color:#526963}table{border-collapse:collapse;width:100%;font-size:29px;margin-top:40px}th{color:#526963;font-size:23px;font-weight:500}td,th{padding:25px 16px;border-bottom:1px solid #d6ded4;text-align:right}td:first-child,th:first-child{text-align:left}.mini-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:40px;margin-top:40px}.mini-grid strong{font-size:64px;color:#267969;font-weight:500}.mini-grid p{font-size:25px;color:#526963;margin-top:12px}.amount-layout{display:grid;grid-template-columns:1.6fr 1fr;gap:50px;align-items:center;margin-top:45px}.amount-layout img{width:100%;height:460px;object-fit:contain}.numbers{display:grid;grid-template-columns:1fr 1fr;gap:42px}.numbers span{font-size:25px;color:#526963}.numbers strong{display:block;font-size:44px;font-weight:500;color:#267969;margin-top:16px}'''
    head='<!doctype html><html lang="vi"><head><meta charset="utf-8"><style>'+css+'</style></head>'
    (OUT/"three_new_pages.html").write_text(head+'<body>'+''.join(pages)+'</body></html>',encoding="utf-8")
    for i,p in enumerate(pages,1):(OUT/f"new_page_{i}.html").write_text(head+'<body>'+p+'</body></html>',encoding="utf-8")


def build_five_review_pages():
    """Replacement pages preserve slide dimensions/styles and replace stale notes."""
    import re
    source=(OUT/"three_new_pages.html").read_text(encoding="utf-8")
    head=source.split('</head>')[0]
    css='''
    .replace-matrix{padding:62px 64px}.replace-matrix h1{font-size:48px;line-height:1.15;margin:0 0 18px;letter-spacing:-1.1px}.replace-matrix .sub{font-size:24px;margin:0;line-height:1.4}.replace-matrix .matrix-hero{position:absolute;left:64px;top:190px;width:1792px;height:680px;object-fit:contain}.replace-matrix footer{left:64px;right:64px;bottom:50px;font-size:20px;line-height:1.6;padding-top:16px;display:block}.replace-matrix .page-n{position:absolute;right:0;bottom:-8px;font-size:17px}
    .replace-target{padding:90px 112px}.replace-target .brandline{display:flex;justify-content:space-between;font-size:17px;color:#687b74;letter-spacing:2px;margin-bottom:32px}.replace-target .kicker{font-size:19px;color:#206b5e;letter-spacing:2px;margin:0 0 22px}.replace-target h1{font-size:64px;line-height:1.17;letter-spacing:-2.8px;max-width:1500px;margin:0 0 28px}.replace-target img{position:absolute;left:112px;top:375px;width:1696px;height:565px;object-fit:contain}footer .page-n{float:right;color:#526963}
    '''
    head+='<style>'+css+'</style></head>'
    def picture(name,cls=''):
        encoded=base64.b64encode((OUT/name).read_bytes()).decode('ascii')
        return f'<img class="{cls}" src="data:image/png;base64,{encoded}"/>'
    matrix_notes=("Lịch sử [2022-01-01, 2023-01-01); 12 tháng trên trục Y, 80 khách không lặp trên trục X. "
        "Chọn ngẫu nhiên seed=42 từ 100.224 ID; chỉ hiển thị mẫu, không khẳng định đại diện cho population. "
        "Một ô là tổng log detail/trx + geo + dialog của một khách–tháng. Ô vuông với aspect equal, không kéo giãn; "
        "số khách nhiều làm dải biểu đồ thấp hơn để bảo toàn tỷ lệ. ID/timestamp hợp lệ; vẫn giữ log thiếu thuộc tính khác. "
        "Không có log không đồng nghĩa null hoặc thiếu thu thập. 0 không ghi nhận sự kiện; timestamp naive; cutoff 01/01/2023. "
        "Dữ liệu matrix_sample_80.csv và matrix_geometry.json trong outputs/behavior_coverage/canva_requested_revision.")
    matrix=f'<section class="replace-matrix" data-document-role="page" data-label="Độ phủ hành vi trong lịch sử 12 tháng" data-speaker-notes="{html.escape(matrix_notes,quote=True)}"><h1>Độ phủ hành vi trong lịch sử 12 tháng</h1><p class="sub">01–12/2022 · Cutoff 01/01/2023 · trục Y: tháng · trục X: 80 khách hàng</p>'+picture('matrix_month_by_customer.png','matrix-hero')+'<footer>Màu = tổng log giao dịch + vị trí + hội thoại; không cộng ptls.<br>Hiển thị mẫu 80/100.224 khách · seed=42 · ô vuông đúng tỷ lệ<span class="page-n">07 / 25</span></footer></section>'
    target_notes=("Biểu đồ chung bốn đường target_1–target_4, cùng một trục tỷ lệ phần trăm từ 0. "
        "Tháng nhãn 02/2022–01/2023; mỗi tháng 100.224 mẫu trước lọc cohort. Tổng positive_n / tổng labelled_n "
        "qua năm fold, không lấy trung bình tỷ lệ không trọng số. Mốc nhãn khác lịch sử input 01–12/2022. "
        "Nguồn: docs/mbd_mini_eda/tables/descriptive.targets/target_by_month_fold.csv; số liệu target_rates_monthly.csv.")
    target=f'<section class="replace-target" data-document-role="page" data-label="Tỷ lệ toàn kỳ che khuất biến động giữa các tháng" data-speaker-notes="{html.escape(target_notes,quote=True)}"><div class="brandline"><span>MBD / PROPENSITY LAB</span><span>BÁO CÁO SƠ BỘ · 06.11.2026</span></div><p class="kicker">NHÃN QUA THỜI GIAN</p><h1>Tỷ lệ toàn kỳ che khuất<br>biến động giữa các tháng.</h1>'+picture('target_four_lines.png')+'<footer>Nguồn: target_by_month_fold.csv · gộp 5 fold · 02/2022–01/2023 · cùng trục Y (%)<span class="page-n">20 / 25</span></footer></section>'
    additions=re.findall(r'<section\b.*?</section>',source,re.S)
    assert len(additions)==3
    for i,number in enumerate([15,17,18]):
        additions[i]=additions[i].replace('</footer>',f'<span class="page-n">{number:02d} / 25</span></footer>')
    pages=[matrix,target,*additions]
    (OUT/'five_review_pages.html').write_text(head+'<body>'+''.join(pages)+'</body></html>',encoding='utf-8')
    for i,p in enumerate(pages,1):(OUT/f'review_page_{i}.html').write_text(head+'<body>'+p+'</body></html>',encoding='utf-8')
    plan={"target_design_id":"DAHaGBuj6pg","target_page_count_before":22,"target_page_count_after":25,
          "replacement_original_pages":[7,17],"source_review_order":["matrix","target_four_lines","daily_activity","customer_descriptive","amount_descriptive"],
          "insert_daily_after_original_page":14,"insert_descriptives_after_original_page":15}
    (OUT/'edit_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    matrix_chart(); target_chart()
    result=scan_year();daily_chart(result)
    customer_stats,amount_stats=stats_pages(result)
    build_new_pages(result,customer_stats,amount_stats)
    build_five_review_pages()
    print(json.dumps({"daily_total":int(result["daily_counts"].sum()),"amount_n":len(result["amounts"]),"new_pages":3}),flush=True)


if __name__=="__main__":main()
