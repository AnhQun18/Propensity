// Review local publication artifacts in a fresh headless browser profile.
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const packages=process.argv[2];
if(!packages)throw Error('Pass the directory containing playwright-core');
const {chromium}=await import(pathToFileURL(path.join(packages,'playwright-core/index.mjs')).href);
const out=path.join(root,'outputs/baseline_presentation_review');
await fs.mkdir(out,{recursive:true});
const browser=await chromium.launch({channel:'msedge',headless:true});
try{
  const page=await browser.newPage({viewport:{width:1280,height:720},deviceScaleFactor:1});
  const errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.goto(pathToFileURL(path.join(root,'deliverables/baseline-final/presentation/canva_import.html')).href);
  await page.evaluate(()=>document.fonts.ready);
  const slides=page.locator('section.slide');
  if(await slides.count()!==8)throw Error('Expected eight slides');
  const checks=[];
  for(let i=0;i<8;i++){
    const slide=slides.nth(i);
    await slide.screenshot({path:path.join(out,`slide-${i+1}.png`)});
    checks.push(await slide.evaluate(el=>{
      const box=el.getBoundingClientRect();
      const footer=el.querySelector('.deck-footer').getBoundingClientRect();
      const overflow=[];
      for(const child of el.children){
        if(child.matches('.deck-footer,.notes'))continue;
        const r=child.getBoundingClientRect();
        if(r.right>box.right-20||r.left<box.left||r.bottom>footer.top-8)
          overflow.push({text:child.textContent.slice(0,90),bottom:r.bottom-box.top,footer:footer.top-box.top});
      }
      return {title:el.dataset.title,overflow,bars:[...el.querySelectorAll('.bar-fill')].map(x=>({style:x.style.width,width:x.getBoundingClientRect().width,parent:x.parentElement.getBoundingClientRect().width}))};
    }));
  }
  await page.goto(pathToFileURL(path.join(root,'deliverables/baseline-final/presentation/index.html')).href);
  await page.waitForTimeout(400);
  if(await page.locator('.deck > .slide.is-active').count()!==1)throw Error('Runtime failed');
  await page.keyboard.press('ArrowRight');
  await page.waitForTimeout(500);
  const navigation=await page.locator('.deck > .slide.is-active').getAttribute('data-title')==='Một khách hàng, một tháng';
  const result={slides:checks,navigation,errors};
  await fs.writeFile(path.join(out,'checks.json'),JSON.stringify(result,null,2));
  console.log(JSON.stringify(result,null,2));
  if(!navigation||errors.length||checks.some(c=>c.overflow.length))process.exitCode=1;
}finally{await browser.close();}
