"""Render the saved window-horizon evaluation and record a selection rationale."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; O=ROOT/'outputs/window_horizon_evaluation'
d=pd.read_csv(O/'metrics_by_cutoff.csv'); s=pd.read_csv(O/'window_summary.csv')
labels={'recent':'Tháng lịch trước','30d':'30 ngày','84d':'84 ngày','all':'Toàn lịch sử'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'text.color':'#193454'})
f,axs=plt.subplots(1,3,figsize=(16,5.8),dpi=200); f.subplots_adjust(left=.06,right=.98,bottom=.23,top=.73,wspace=.3)
for ax,(metric,title) in zip(axs,[('roc_auc','ROC-AUC ↑'),('average_precision','Average Precision (AP) ↑'),('brier','Brier ↓')]):
 for tag,col in zip(['recent','30d','84d','all'],['#7e98b6','#db985c','#71b099','#326fc2']):
  g=d[d.window==tag]; ax.plot(g.cutoff.str[:7],g[metric],marker='o',color=col,label=labels[tag])
 ax.set_title(title,fontweight='bold'); ax.grid(alpha=.2); ax.tick_params(axis='x',rotation=20)
f.suptitle('So sánh cửa sổ lịch sử theo cutoff',fontsize=22,fontweight='bold',y=.95)
f.legend(*axs[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.87),ncol=4,frameon=False)
f.text(.06,.075,'Rolling validation 09–11/2022 · Cùng 31 đặc trưng · Xác suất mô hình gốc, chưa hiệu chỉnh calibration.',fontsize=11)
f.savefig(O/'window_metrics_by_cutoff.png')
def markdown(df):
 def val(v): return f'{v:.6f}' if isinstance(v,float) else str(v)
 return '\n'.join(['| '+' | '.join(df.columns)+' |','| '+' | '.join(['---']*len(df.columns))+' |',*['| '+' | '.join(val(v) for v in row)+' |' for row in df.itertuples(index=False,name=None)]])
lines=['# Đánh giá cửa sổ lịch sử trước cutoff','', 'Kỳ dự báo giữ nguyên một tháng. Chỉ thay cửa sổ hành vi; 15 đặc trưng lịch sử mua giữ nguyên.', '', '## Kết quả theo cutoff','',markdown(d),'','## Trung bình các cutoff','',markdown(s),'','## Lựa chọn','',f"Cửa sổ đứng đầu theo AP trung bình: {labels[s.iloc[0]['window']]}. Đây là lựa chọn baseline theo các validation đã thử, không phải bằng chứng tối ưu tuyệt đối.",'','AP dùng average_precision_score; PR-AUC hình thang được xuất riêng. ROC-AUC/AP/PR-AUC càng cao càng tốt; Brier càng thấp càng tốt. Brier skill so với hằng số tỷ lệ dương của tập train cùng cutoff. Không fit calibration hoặc mở test.','', 'So sánh mô hình cùng siêu tham số, cùng 16 tóm tắt hành vi và 15 đặc trưng lịch sử mua. Tháng lịch trước khác cửa sổ cố định 30 ngày. Toàn lịch sử là từ đầu log sẵn có tới trước cutoff, không phải thời gian cố định.','', 'Chỉ có ba cutoff; chưa kiểm định ý nghĩa thống kê của chênh lệch. Brier còn phụ thuộc tỷ lệ dương; so các cửa sổ trong cùng cutoff trước khi so qua tháng. Kết quả lịch sử mua dùng giả định nhãn có sẵn sau cuối tháng, chưa xác minh SLA nhà phát hành.']
(O/'report.md').write_text('\n'.join(lines),encoding='utf-8')
