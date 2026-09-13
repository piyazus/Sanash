// UI smoke test against a locally served synthetic intake; no passenger data.
// Usage: node verify_browser.cjs <playwright-core-path> <output-dir> [URL]
const { chromium } = require(process.argv[2]);
const path = require('node:path');
const fs = require('node:fs');
const assert = require('node:assert/strict');
const out = path.resolve(process.argv[3]);
const url = process.argv[4] || 'http://127.0.0.1:8769/annotate.html';
if (!['localhost', '127.0.0.1'].includes(new URL(url).hostname)) throw Error('Only a local synthetic fixture is supported.');
if (fs.existsSync(out)) throw Error('Choose a new output directory.');
fs.mkdirSync(out, {recursive:true});

(async()=>{
 const browser=await chromium.launch({channel:'chrome',headless:true});
 const errors=[];
 async function pageFor(rater){
  const context=await browser.newContext({viewport:{width:1280,height:900},acceptDownloads:true});
  const page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));
  await page.goto(url);await page.getByText('Искусственный тест. Пассажирских данных нет.',{exact:true}).waitFor();
  await page.locator('#picture').waitFor();
  await page.locator('#rater').fill(rater);await page.locator('#begin').click();return page;
 }
 async function mark(page,x,y){const box=await page.locator('#overlay').boundingBox();await page.mouse.click(box.x+box.width*x,box.y+box.height*y);}
 async function exportFile(page,name){const pending=page.waitForEvent('download');await page.locator('#export').click();const download=await pending;await download.saveAs(path.join(out,name));}
 try{
  const a=await pageFor('QA_BROWSER_A');
  await a.locator('#zone').selectOption('middle');await a.locator('#posture').selectOption('seated');await a.locator('#visibility').selectOption('partial');
  await mark(a,.25,.3);assert.equal(await a.locator('#counter').textContent(),'Отмечено: 1');
  await a.locator('#visibility').selectOption('inferred');await a.locator('#zone').selectOption('rear');
  await mark(a,.7,.6);await a.locator('#undo').click();assert.equal(await a.locator('#counter').textContent(),'Отмечено: 1');
  await mark(a,.7,.6);await a.locator('#notes').fill('SYNTHETIC UI CHECK ONLY');await a.locator('#reviewed').check();
  await a.locator('#rater').fill('DIFFERENT');await a.locator('#begin').click();assert.match(await a.locator('#message').textContent(),/Сначала скачайте/);
  await a.locator('#rater').fill('QA_BROWSER_A');await exportFile(a,'rater_a_partial.json');
  await a.reload();await a.locator('#rater').fill('QA_BROWSER_A');await a.locator('#begin').click();
  assert.equal(await a.locator('#counter').textContent(),'Отмечено: 2');assert.equal(await a.locator('#reviewed').isChecked(),true);
  await a.locator('#next').click();assert.equal(await a.locator('#reviewed').isChecked(),false);await a.locator('#reviewed').check();
  await exportFile(a,'rater_a.json');await a.locator('#previous').click();
  await a.screenshot({path:path.join(out,'desktop.png'),fullPage:true});
  await a.setViewportSize({width:390,height:844});
  assert.equal(await a.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth),true);
  await a.screenshot({path:path.join(out,'mobile.png'),fullPage:true});
  const b=await pageFor('QA_BROWSER_B');await b.locator('#zone').selectOption('middle');await mark(b,.25,.3);
  await b.locator('#reviewed').check();await b.locator('#next').click();await b.locator('#reviewed').check();await exportFile(b,'rater_b.json');
  await b.locator('#import').setInputFiles(path.join(out,'rater_a.json'));
  await b.getByText('Файл относится к другому проекту или разметчику.',{exact:true}).waitFor();
  const data=JSON.parse(fs.readFileSync(path.join(out,'rater_a.json'),'utf8'));
  assert.equal(data.origin,'synthetic');assert.equal(data.frames[0].points.length,2);
  assert.equal(data.frames[1].points.length,0);assert.equal(data.frames[1].reviewed,true);assert.equal(data.frames[2].reviewed,false);
  assert.deepEqual(errors,[]);
  fs.writeFileSync(path.join(out,'result.json'),JSON.stringify({status:'passed',origin:'synthetic',url,
   checks:['point placement','point labels','undo','rater-switch guard','JSON download','draft reload','explicit reviewed zero','unreviewed excluded','wrong-rater import rejected','desktop render','mobile no overflow','no page errors'],page_errors:errors},null,2));
  console.log('Browser checks passed; synthetic exports and screenshots: '+out);
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
