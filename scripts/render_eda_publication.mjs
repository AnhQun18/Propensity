// Render only the local publication artifacts. No personal browser profile is used.
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const packages=process.argv[2];
if(!packages) throw new Error('Pass the directory containing the playwright package');
const { chromium }=await import(pathToFileURL(path.join(packages,'playwright/index.mjs')).href);
const out=path.join(root,'outputs/eda_publication_review');
await fs.mkdir(out,{recursive:true});
const browser=await chromium.launch({channel:'msedge',headless:true});
const page=await browser.newPage({viewport:{width:1280,height:720},deviceScaleFactor:1});
page.on('pageerror',error=>console.error('Page JavaScript:',String(error)));
const sections=[];
try {
  await page.goto(pathToFileURL(path.join(root,'deliverables/eda/presentation/canva_import.html')).href);
  await page.evaluate(()=>document.fonts.ready);
  const slides=page.locator('section.slide');
  const count=await slides.count();
  if(count!==9) throw new Error(`Expected 9 slides, found ${count}`);
  for(let i=0;i<count;i++) {
    const slide=slides.nth(i);
    await slide.screenshot({path:path.join(out,`slide-${i+1}.png`)});
    const check=await slide.evaluate(el=>{
      const b=el.getBoundingClientRect(), footer=el.querySelector('.deck-footer').getBoundingClientRect();
      const over=[];
      for(const child of [...el.children].filter(x=>!x.matches('.deck-footer,aside.notes'))) {
        const r=child.getBoundingClientRect();
        if(r.right>b.right-20 || r.left<b.left || r.bottom>footer.top-10) over.push({tag:child.tagName,text:child.textContent.slice(0,80),bottom:r.bottom-b.top,footerTop:footer.top-b.top});
      }
      return {title:el.dataset.label,overflow:over};
    });
    sections.push({index:i+1,...check});
  }
  await page.goto(pathToFileURL(path.join(root,'deliverables/eda/presentation/index.html')).href);
  await page.waitForTimeout(400);
  if(await page.locator('.deck > .slide.is-active').count()!==1)throw new Error('Keyboard runtime did not initialize');
  await page.keyboard.press('ArrowRight');
  await page.waitForTimeout(550);
  if(await page.locator('.deck > .slide.is-active').getAttribute('data-label')!=='Phạm vi và cấu trúc dữ liệu')throw new Error('Keyboard navigation failed');
  const thumbPath=path.join(out,'canva_pages.json');
  let canvaImages=[];
  try {
    const thumbnails=JSON.parse(await fs.readFile(thumbPath,'utf8'));
    const markup='<html><meta charset="utf-8"><style>body{margin:0;padding:20px;background:#eaf0f4;font:16px Arial}main{display:grid;grid-template-columns:repeat(3,596px);gap:16px}figure{margin:0}img{display:block;width:596px;height:335px;background:white}figcaption{padding:8px}</style><main>'+thumbnails.map(t=>`<figure><img src="${t.url.replaceAll('&','&amp;')}"/><figcaption>Trang ${t.index}</figcaption></figure>`).join('')+'</main></html>';
    await page.setViewportSize({width:1860,height:1200});
    await page.setContent(markup,{waitUntil:'networkidle'});
    canvaImages=await page.locator('img').evaluateAll(imgs=>imgs.map(x=>({loaded:x.complete && x.naturalWidth>0,width:x.naturalWidth})));
    await page.screenshot({path:path.join(out,'canva-contact-sheet.png'),fullPage:true});
  } catch(err) { canvaImages=[{error:String(err)}]; }
  const report={slides:sections,keyboard_navigation:true,canva_images:canvaImages};
  await fs.writeFile(path.join(out,'layout_checks.json'),JSON.stringify(report,null,2));
  console.log(JSON.stringify(report,null,2));
} finally { await browser.close(); }
