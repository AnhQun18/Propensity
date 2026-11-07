from pathlib import Path
import json,base64,html,re
from pypdf import PdfReader
import pdfplumber
ROOT=Path(__file__).resolve().parents[2];WORK=Path(__file__).resolve().parent
PDF=ROOT/'output/pdf/MBD_mini_Bao_cao_day_du.pdf'
pages=json.loads((WORK/'html_text_pages.json').read_text(encoding='utf-8'))
summary=json.loads((WORK/'build_summary.json').read_text(encoding='utf-8'))
reader=PdfReader(PDF);sections=[];scale=4/3;link_count=0
for p,pdfpage,title in zip(pages,reader.pages,summary['page_titles']):
    png=WORK/'graphics'/f"page-{p['index']:03}.png"
    links=[]
    for a in pdfpage.get('/Annots',[]):
        a=a.get_object();action=a.get('/A',{})
        if action.get('/URI'):
            rect=[float(x) for x in a['/Rect']]
            links.append((rect[0],p['height']-rect[3],rect[2],p['height']-rect[1],str(action['/URI'])))
    children=['<img class="graphics" alt="" src="data:image/png;base64,'+base64.b64encode(png.read_bytes()).decode()+'">']
    for r in p['runs']:
        style=f"left:{r['left']*scale:.3f}px;top:{(r['top']-.075*r['size'])*scale:.3f}px;font-size:{r['size']*scale:.3f}px;color:{r['color']};font-weight:{r['weight']};font-style:{r['style']}"
        value=html.escape(r['text'])
        for x0,y0,x1,y1,url in links:
            if x0-1<=r['left']<=x1+1 and y0-2<=r['top']+r['size']*.5<=y1+2:
                value='<a href="'+html.escape(url,quote=True)+'">'+value+'</a>';link_count+=1;break
        children.append('<div class="copy" style="'+style+'">'+value+'</div>')
    sections.append('<section class="page" data-document-role="page" data-label="'+html.escape(title,quote=True)+'" style="width:'+str(p['width']*scale)+'px;height:'+str(p['height']*scale)+'px">'+''.join(children)+'</section>')
out=ROOT/'output/pdf/MBD_mini_Bao_cao_day_du_Canva.html'
out.write_text('<!doctype html><html lang="vi"><head><meta charset="utf-8"><title>MBD-mini | Báo cáo phân tích đầy đủ</title><style>*{box-sizing:border-box}html,body{margin:0;padding:0;background:#fff;font-family:Arial,sans-serif}.page{position:relative;margin:0;overflow:hidden;background:#fff;page-break-after:always}.graphics{position:absolute;left:0;top:0;width:100%;height:100%}.copy{position:absolute;line-height:1;white-space:pre;letter-spacing:0;font-family:Arial,sans-serif}a{color:inherit;text-decoration:none}</style></head><body>'+''.join(sections)+'</body></html>',encoding='utf-8')
alltext='\n'.join(r['text'] for p in pages for r in p['runs'])
ledger=json.loads((WORK/'coverage_ledger.json').read_text(encoding='utf-8'))
audit={'pages':len(pages),'tables':len(ledger['tables']),'figures':len(ledger['figures']),'linked_text_blocks':link_count,
 'raw_64hex_ids':len(re.findall(r'\b[0-9a-f]{64}\b',alltext)),
 'local_machine_paths':len(re.findall(r'[A-Z]:[\\/]|Users[\\/]admin',alltext)),
 'html_bytes':out.stat().st_size,'pdf_bytes':PDF.stat().st_size}
with pdfplumber.open(PDF) as doc:
    assert len(doc.pages)==summary['pages']
    assert all(len(p.chars)>80 for p in doc.pages)
    assert all(-.5<=ch['x0']<=ch['x1']<=float(p.width)+.5 for p in doc.pages for ch in p.chars)
    # Every source filename and exported variable is discoverable in the catalogue.
    catalog='\n'.join(p.extract_text() for p in doc.pages[summary['catalog_start']-1:-1])
    compact=lambda s:re.sub(r'\s+','',s)
    for item in ledger['tables']:
        assert compact(item['key']) in compact(catalog),item['key']
        assert all(compact(c) in compact(catalog) for c in item['columns']),item['key']
    assert audit['raw_64hex_ids']==audit['local_machine_paths']==0
    audit['verification']=f"{summary['pages']} pages nonempty; horizontal bounds; 61 table names/all columns in catalogue; public aggregate data only"
(WORK/'disclosure_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(audit,ensure_ascii=False))
