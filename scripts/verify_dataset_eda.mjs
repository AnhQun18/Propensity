// Validate the standalone 40-slide deck in offline Edge; export every slide.
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import crypto from 'node:crypto';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const modules=process.argv[2];
if(!modules)throw Error('Pass node_modules containing playwright-core');
const {chromium}=await import(pathToFileURL(path.join(modules,'playwright-core/index.mjs')).href);
const dir=path.join(root,'deliverables/mbd-mini-dataset-eda');
const render=path.join(root,'outputs/dataset_eda/render');
await fs.mkdir(render,{recursive:true});
const browser=await chromium.launch({channel:'msedge',headless:true});
const result={browser:'Microsoft Edge / Playwright',viewport:[1280,720],main:30,appendix:10,checks:[]};
try{
 const context=await browser.newContext({viewport:{width:1280,height:720},deviceScaleFactor:1,offline:true});
 const page=await context.newPage();const errors=[],network=[];
 page.on('pageerror',e=>errors.push(e.message));
 page.on('request',r=>{if(/^https?:/.test(r.url()))network.push(r.url())});
 await page.goto(pathToFileURL(path.join(dir,'index.html')).href);
 await page.evaluate(()=>document.fonts.ready);
 const count=await page.locator('.deck > .slide').count();result.total=count;
 for(let n=1;n<=count;n++){
  await page.evaluate(n=>location.hash='#/'+n,n);await page.waitForTimeout(130);
  const c=await page.evaluate(()=>{
   const s=document.querySelector('.deck > .slide.is-active');
   const box=e=>e.getBoundingClientRect();const title=s.querySelector('h2'),visual=s.querySelector('.visual'),insight=s.querySelector('.insight'),method=s.querySelector('.method'),source=s.querySelector('.source');
   const v=box(visual);const svgOverflow=[...s.querySelectorAll('svg text')].filter(e=>{const b=box(e);return b.left<v.left-2||b.right>v.right+2||b.top<v.top-2||b.bottom>v.bottom+2}).map(e=>e.textContent);
   const cellOverflow=[...s.querySelectorAll('th,td')].filter(e=>e.scrollWidth>e.clientWidth+2).map(e=>e.textContent);
   return {title:title.textContent,activeCount:document.querySelectorAll('.deck > .slide.is-active').length,
    titleOverlap:box(title).bottom>v.top,visualInsightOverlap:v.bottom>box(insight).top,
    insightMethodOverlap:box(insight).bottom>box(method).top,methodSourceOverlap:box(method).bottom>box(source).top,
    tableOutside:[...s.querySelectorAll('table')].some(e=>box(e).top<v.top-2||box(e).bottom>v.bottom+2),
    childOutside:[...visual.querySelectorAll('article,.flow>div,.hero-stats')].some(e=>box(e).bottom>v.bottom+2),
    svgOverflow,cellOverflow,hasNotes:s.querySelector('.notes').textContent.trim().split(/\s+/).length>100,
    hasSource:!!source.textContent.trim(),containsReplacement:s.textContent.includes('\ufffd')};
  });
  c.slide=n;c.pass=c.activeCount===1&&!c.titleOverlap&&!c.visualInsightOverlap&&!c.insightMethodOverlap&&!c.methodSourceOverlap&&!c.tableOutside&&!c.childOutside&&!c.svgOverflow.length&&!c.cellOverflow.length&&c.hasNotes&&c.hasSource&&!c.containsReplacement;
  result.checks.push(c);
  await page.screenshot({path:path.join(render,`${String(n).padStart(2,'0')}.png`)});
 }
 await page.keyboard.press('Home');await page.keyboard.press('ArrowRight');await page.waitForTimeout(120);
 result.keyboard=await page.evaluate(()=>location.hash==='#/2');
 await page.getByRole('button',{name:'Phụ lục',exact:true}).click();await page.waitForTimeout(130);
 result.appendixButton=await page.evaluate(()=>location.hash==='#/31');
 await page.keyboard.press('o');await page.waitForTimeout(150);
 result.overview=await page.evaluate(()=>{const e=document.querySelector('.overview');return e&&getComputedStyle(e).display!=='none'&&getComputedStyle(e).opacity!=='0'});
 await page.keyboard.press('Escape');
 await page.keyboard.press('n');await page.waitForTimeout(100);
 result.notes=await page.evaluate(()=>{const e=document.querySelector('.notes-overlay.open');return !!e&&e.getBoundingClientRect().top<innerHeight});
 await page.keyboard.press('Escape');
 const popupEvent=page.waitForEvent('popup');await page.keyboard.press('s');const presenter=await popupEvent;
 await presenter.waitForLoadState();await presenter.waitForTimeout(250);
 result.presenter=(await presenter.locator('iframe').count())===2;
 await presenter.keyboard.press('ArrowRight');await page.waitForTimeout(200);
 result.presenterSync=await page.evaluate(()=>location.hash==='#/32');
 await presenter.screenshot({path:path.join(render,'presenter.png')});await presenter.close();
 await page.keyboard.press('f');await page.waitForTimeout(200);
 result.fullscreen=await page.evaluate(()=>!!document.fullscreenElement);
 if(result.fullscreen)await page.evaluate(()=>document.exitFullscreen());
 await page.setViewportSize({width:390,height:844});await page.waitForTimeout(130);
 result.mobileFits=await page.evaluate(()=>{const b=document.querySelector('.deck').getBoundingClientRect();return b.left>=-1&&b.right<=innerWidth+1&&b.top>=-1&&b.bottom<=innerHeight+1&&document.documentElement.scrollWidth<=innerWidth});
 await page.screenshot({path:path.join(render,'mobile.png')});
 // Main file alone keeps all aggregate source tables embedded and requires no fetch.
 result.embeddedData=await page.evaluate(()=>{const d=JSON.parse(document.getElementById('deck-data').textContent);return d.manifest.checks.y_any_positive===12854&&Object.keys(d.tables).length>=20});
 const downloadEvent=page.waitForEvent('download');await page.getByRole('button',{name:'Tải số liệu',exact:true}).click();const download=await downloadEvent;
 result.download=download.suggestedFilename()==='MBD-mini_EDA_aggregate_data.json';
 await page.setViewportSize({width:1280,height:720});
 await page.emulateMedia({media:'print'});
 result.printVisible=await page.locator('.deck > .slide:visible').count()===40;
 result.errors=errors;result.networkRequests=network;
 result.html_sha256=crypto.createHash('sha256').update(await fs.readFile(path.join(dir,'index.html'))).digest('hex');
 result.pass=count===40&&result.checks.every(c=>c.pass)&&result.keyboard&&result.appendixButton&&result.overview&&result.notes&&result.presenter&&result.presenterSync&&result.fullscreen&&result.mobileFits&&result.embeddedData&&result.download&&result.printVisible&&!errors.length&&!network.length;
 result.checked_utc=new Date().toISOString();
 await fs.writeFile(path.join(dir,'verification.json'),JSON.stringify(result,null,2)+'\n');
 console.log(JSON.stringify({...result,checks:result.checks.filter(c=>!c.pass)},null,2));
 if(!result.pass)throw Error('Deck QA failed; see verification.json');
 await context.close();
}finally{await browser.close()}
