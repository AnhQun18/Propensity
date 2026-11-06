"""Separate PDF vector evidence from editable text for faithful HTML import."""
from pathlib import Path
import base64
import html
import json
from collections import defaultdict
from pypdf import PdfReader,PdfWriter
from pypdf.generic import ContentStream,NameObject
import pdfplumber

ROOT=Path(__file__).resolve().parents[2]
WORK=Path(__file__).resolve().parent
PDF=ROOT/'output/pdf/MBD_mini_Bao_cao_day_du.pdf'
reader=PdfReader(PDF);writer=PdfWriter()
for page in reader.pages:
    stream=ContentStream(page.get_contents(),reader)
    ops=[];inside=False
    for operands,operator in stream.operations:
        if operator==b'BT':inside=True;continue
        if operator==b'ET':inside=False;continue
        if not inside:ops.append((operands,operator))
    stream.operations=ops
    page[NameObject('/Contents')]=writer._add_object(stream)
    writer.add_page(page)
with (WORK/'graphics_only.pdf').open('wb') as handle:writer.write(handle)

def rgb(color):
    if color is None:return '#233747'
    if isinstance(color,(float,int)):color=(color,color,color)
    if len(color)==4:
        c,m,y,k=color;color=(1-min(1,c+k),1-min(1,m+k),1-min(1,y+k))
    return '#'+''.join(f'{round(float(v)*255):02x}' for v in color[:3])

pages=[];manifest=[]
with pdfplumber.open(PDF) as doc:
    for index,page in enumerate(doc.pages,1):
        groups=defaultdict(list)
        for ch in page.chars:
            groups[(round(ch['top'],1),ch['fontname'],round(ch['size'],2),rgb(ch.get('non_stroking_color')))].append(ch)
        runs=[]
        for (_,font,size,color),chars in groups.items():
            chars.sort(key=lambda x:x['x0']);blocks=[];current=[]
            for ch in chars:
                if current and ch['x0']-current[-1]['x1']>size*.9:
                    blocks.append(current);current=[]
                current.append(ch)
            if current:blocks.append(current)
            for block in blocks:
                value='';prior=None
                for ch in block:
                    if prior and ch['x0']-prior['x1']>size*.18 and not value.endswith(' ') and ch['text']!=' ':value+=' '
                    value+=ch['text'];prior=ch
                if not value.strip():continue
                runs.append(dict(left=block[0]['x0'],top=min(ch['top'] for ch in block),
                    width=block[-1]['x1']-block[0]['x0'],text=value,size=size,color=color,
                    weight='700' if 'Bold' in font else '400',style='italic' if 'Italic' in font else 'normal'))
        pages.append(dict(index=index,width=float(page.width),height=float(page.height),runs=runs))
        manifest.append(dict(page=index,editable_text_blocks=len(runs)))
(WORK/'html_text_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
(WORK/'html_text_pages.json').write_text(json.dumps(pages,ensure_ascii=False),encoding='utf-8')
print(json.dumps({'pages':len(pages),'text_blocks':sum(len(p['runs']) for p in pages)}))
