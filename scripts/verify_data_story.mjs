// Render and validate both evidence-backed HTML decks. Pass node_modules containing playwright-core.
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {fileURLToPath,pathToFileURL} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
if(!process.argv[2])throw Error('Pass node_modules directory containing playwright-core');
const {chromium}=await import(pathToFileURL(path.join(process.argv[2],'playwright-core/index.mjs')).href);
const out=path.join(root,'outputs/data_story_review_v2');await fs.mkdir(out,{recursive:true});
const browser=await chromium.launch({channel:'msedge',headless:true});
const results=[];
try{
 for(const [file,expectedMain] of [['index.html',19],['detailed.html',23]]){
  const context=await browser.newContext({viewport:{width:1280,height:720},deviceScaleFactor:1});
  const page=await context.newPage();const errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.goto(pathToFileURL(path.join(root,'deliverables/propensity-data-story',file)).href);
  await page.evaluate(()=>document.fonts.ready);await page.waitForTimeout(150);
  const main=await page.evaluate(()=>Number(document.body.dataset.mainCount));
  const total=await page.locator('.deck > .slide').count();const checks=[];
  const dir=path.join(out,file.replace('.html',''));await fs.mkdir(dir,{recursive:true});
  for(let i=1;i<=total;i++){
   await page.evaluate(i=>location.hash='#/'+i,i);await page.waitForTimeout(140);
   checks.push(await page.evaluate(()=>{
    const s=document.querySelector('.deck > .slide.is-active');
    const title=s.querySelector('h2').getBoundingClientRect();
    const chart=s.querySelector('.chart').getBoundingClientRect();
    const meaning=s.querySelector('.meaning').getBoundingClientRect();
    const method=s.querySelector('.method').getBoundingClientRect();
    const source=s.querySelector('.source').getBoundingClientRect();
    const overflow=[...s.querySelectorAll('svg text')].filter(e=>{const b=e.getBoundingClientRect();return b.left<48||b.right>1236||b.bottom>526||b.top<145}).map(e=>e.textContent);
    const cells=[...s.querySelectorAll('td,th')].filter(e=>{const b=e.getBoundingClientRect();return b.bottom>526||e.scrollWidth>e.clientWidth+1}).map(e=>e.textContent);
    return {title:s.querySelector('h2').textContent,visuals:s.querySelectorAll('svg,table').length,
     titleChartOverlap:title.bottom>chart.top,meaningMethodOverlap:meaning.bottom>method.top,methodSourceOverlap:method.bottom>source.top,overflow,cells,
     hasMethod:!!s.querySelector('.method').textContent,hasMeaning:s.querySelector('.meaning').textContent.startsWith('Ý nghĩa cho mô hình:')};
   }));
   await page.screenshot({path:path.join(dir,`${String(i).padStart(2,'0')}.png`)});
  }
  await page.keyboard.press('Home');await page.keyboard.press('ArrowRight');await page.waitForTimeout(150);
  const keyboard=await page.evaluate(()=>location.hash==='#/2');
  await page.getByRole('button',{name:'Phụ lục',exact:true}).click();await page.waitForTimeout(150);
  const appendix=await page.evaluate(m=>location.hash===`#/${m+1}`,main);
  await page.keyboard.press('o');await page.waitForTimeout(150);
  const overview=await page.evaluate(()=>{const e=document.querySelector('.overview');return e&&getComputedStyle(e).opacity!=='0'&&getComputedStyle(e).display!=='none'});
  await page.keyboard.press('Escape');
  const pop=page.waitForEvent('popup');await page.keyboard.press('s');const presenter=await pop;
  await presenter.waitForLoadState();await presenter.waitForTimeout(250);
  const presenterWorks=await presenter.locator('iframe').count()===2;await presenter.close();
  await page.setViewportSize({width:390,height:844});await page.waitForTimeout(150);
  const mobile=await page.evaluate(()=>{const r=document.querySelector('.deck').getBoundingClientRect();return r.width<=innerWidth+1&&r.height<=innerHeight+1&&document.documentElement.scrollWidth<=innerWidth});
  await page.screenshot({path:path.join(dir,'mobile.png')});
  const result={file,main,total,appendixCount:total-main,keyboard,appendixButton:appendix,overview,presenterWorks,mobileFits:mobile,errors,checks};
  const bad=checks.filter(c=>c.visuals!==1||c.titleChartOverlap||c.meaningMethodOverlap||c.methodSourceOverlap||c.overflow.length||c.cells.length||!c.hasMethod||!c.hasMeaning);
  result.pass=main===expectedMain&&keyboard&&appendix&&overview&&presenterWorks&&mobile&&!errors.length&&!bad.length;
  results.push(result);console.log(JSON.stringify({file,main,total,pass:result.pass,errors,bad}));
  await context.close();
 }
 const hashes={};for(const r of results)hashes[r.file]=crypto.createHash('sha256').update(await fs.readFile(path.join(root,'deliverables/propensity-data-story',r.file))).digest('hex');
 const report={checked_at_utc:new Date().toISOString(),browser:'Microsoft Edge headless / Playwright',viewport:[1280,720],mobile_viewport:[390,844],html_sha256:hashes,results};
 await fs.writeFile(path.join(out,'verification.json'),JSON.stringify(report,null,2)+'\n');
 if(results.some(r=>!r.pass))throw Error('Visual/interaction QA failed. See outputs/data_story_review_v2/verification.json');
 await fs.writeFile(path.join(root,'deliverables/propensity-data-story/verification.json'),JSON.stringify(report,null,2)+'\n');
}finally{await browser.close();}
