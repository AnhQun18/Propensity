"""Four comparable, disjoint customer samples on one Canva-ready page."""
from pathlib import Path
import base64
import csv
import html
import json

import numpy as np
from matplotlib.figure import Figure
from matplotlib.colors import Normalize

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/behavior_coverage/quartet_12months"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with np.load(ROOT / "outputs/behavior_coverage/matrix.npz", allow_pickle=False) as data:
        counts, months, ids = (data[k] for k in ("counts", "months", "client_ids"))
    # Twelve whole calendar months immediately preceding the January 2023 cutoff.
    keep = (months >= "2022-01") & (months < "2023-01")
    shown_months = months[keep]
    assert np.array_equal(shown_months, np.arange("2022-01", "2023-01", dtype="datetime64[M]").astype(str))
    selected = np.random.default_rng(42).choice(len(counts), 40, replace=False)
    assert len(np.unique(selected)) == 40
    values = counts[selected][:, keep]
    maximum = int(values.max())
    norm = Normalize(vmin=0, vmax=max(1, maximum))
    panels = []
    rows = []
    for group, indices, part in zip("ABCD", np.split(selected, 4), np.split(values, 4)):
        fig = Figure(figsize=(9, 3.7), facecolor="#f6f4ec")
        ax = fig.subplots()
        image = ax.imshow(part, cmap="Greens", norm=norm, aspect="auto", interpolation="nearest")
        labels = [f"KH-{int(i)+1:06d}" for i in indices]
        ax.set_yticks(np.arange(10), labels, fontsize=8.1)
        ax.set_xticks(np.arange(12), [m[5:] for m in shown_months], fontsize=9)
        ax.set_xticks(np.arange(13)-.5, minor=True)
        ax.set_yticks(np.arange(11)-.5, minor=True)
        ax.grid(which="minor", color="white", linewidth=.7)
        ax.tick_params(which="both", length=0, pad=4)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for r in range(10):
            for c in range(12):
                v = int(part[r, c])
                rgba = image.cmap(norm(v))
                light = .2126*rgba[0]+.7152*rgba[1]+.0722*rgba[2]
                ax.text(c, r, str(v), ha="center", va="center", fontsize=9.1,
                        color="white" if light < .48 else "#173d38")
        ax.set_title(f"Nhóm {group} · 10 khách hàng", fontsize=14, weight="bold", loc="left", pad=12, color="#173d38")
        ax.set_xlabel("Tháng trong năm 2022", fontsize=9, color="#526963", labelpad=6)
        cbar = fig.colorbar(image, ax=ax, fraction=.026, pad=.025)
        cbar.set_label("Số sự kiện", fontsize=8)
        cbar.ax.tick_params(labelsize=8, length=0)
        fig.subplots_adjust(left=.125, right=.93, top=.85, bottom=.16)
        path = OUT / f"matrix_{group}.png"
        fig.savefig(path, dpi=220, facecolor=fig.get_facecolor())
        panels.append(path)
        for label, index, row in zip(labels, indices, part):
            rows.append([group, label, ids[index], *row.tolist()])
    with (OUT / "samples.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group", "display_label", "client_id", *shown_months.tolist()])
        writer.writerows(rows)
    notes = ("Lịch sử [2022-01-01, 2023-01-01), gồm đúng 12 tháng; cutoff 01/01/2023, "
             "phù hợp dự đoán nhãn tháng 01/2023 (mon=2023-01-31). "
             "Chọn ngẫu nhiên 40 khách không lặp trong 100.224 khách, seed=42, chia bốn nhóm 10 khách. "
             "Mỗi ô là tổng số log detail/trx + geo + dialog, không phải số ngày và không cộng ptls. "
             f"Cả bốn hình cùng thang màu tuyến tính 0–{maximum}, không clipping. "
             "Giữ sự kiện có ID/timestamp hợp lệ, kể cả thiếu thuộc tính khác. "
             "Mẫu minh họa không phải toàn bộ population. 0 là không ghi nhận log, không chứng minh thiếu thu thập. "
             "Mã KH ánh xạ về ID trong CSV lưu trên máy, không gửi CSV lên Canva.")
    images = []
    for path in panels:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        images.append(f'<img src="data:image/png;base64,{encoded}" alt="Ma trận nhóm {path.stem[-1]}"/>')
    doc = f'''<!doctype html><html lang="vi"><head><meta charset="utf-8"><style>
    *{{box-sizing:border-box}}html,body{{margin:0;padding:0;width:1920px;background:#f6f4ec;color:#173d38;font-family:Arial,sans-serif}}
    section{{width:1920px;height:1080px;position:relative;padding:62px 64px}}
    h1{{font-size:48px;line-height:1.15;font-weight:600;margin:0 0 18px;letter-spacing:-1.1px}}
    .sub{{font-size:24px;color:#526963;margin:0}}
    .charts{{position:absolute;left:45px;right:45px;top:190px;display:grid;grid-template-columns:1fr 1fr;gap:10px 18px}}
    img{{display:block;width:906px;height:350px;object-fit:contain}}
    footer{{position:absolute;left:64px;right:64px;bottom:50px;font-size:20px;color:#526963;line-height:1.6;border-top:1px solid #d6ded4;padding-top:16px}}
    </style></head><body><section data-document-role="page" data-label="Độ phủ hành vi — lịch sử 12 tháng" data-speaker-notes="{html.escape(notes, quote=True)}">
    <h1>Độ phủ hành vi trong lịch sử 12 tháng</h1>
    <p class="sub">01–12/2022 · Cutoff 01/01/2023 · 4 nhóm × 10 khách hàng không trùng nhau</p>
    <div class="charts">{''.join(images)}</div>
    <footer>Số trong ô = tổng log giao dịch + vị trí + hội thoại · Thang màu chung: 0–{maximum}<br>Mẫu ngẫu nhiên 40/100.224 khách · seed=42 · 0 = không ghi nhận log</footer>
    </section></body></html>'''
    (OUT / "canva_page.html").write_text(doc, encoding="utf-8")
    meta = {"history_start_inclusive":"2022-01-01", "cutoff_exclusive":"2023-01-01", "months":shown_months.tolist(),
            "population":len(counts), "sample_size":40, "group_size":10, "seed":42, "disjoint_groups":True,
            "common_vmin":0, "common_vmax":maximum, "notes":notes}
    (OUT / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k:meta[k] for k in ["months","sample_size","common_vmax"]}))
    print(OUT)


if __name__ == "__main__":
    main()
