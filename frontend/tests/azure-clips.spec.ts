import {expect,test} from '@playwright/test';
import {filterEvents} from '../src/time';
import {writeFile} from 'node:fs/promises';
test('all Azure clips select, play and seek while analysis remains gated by validation',async({page})=>{
 test.skip(!process.env.TACTIQAI_AZURE_BROWSER,'Explicit live Azure integration opt-in required.');
 const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));
 const listing=await page.request.get('/api/recordings');expect(listing.status()).toBe(200);
 const rows=(await listing.json()).recordings;
 if(process.env.TACTIQAI_AZURE_REQUIRE_ANALYSIS==='1'){
  expect(rows.filter((r:{source?:string})=>r.source!=='cv_experimental')).toHaveLength(10);
  expect(rows.filter((r:{source?:string})=>r.source!=='cv_experimental').every((r:{artifact_ready:boolean,status:string})=>r.artifact_ready&&r.status==='available')).toBe(true);
 }
 const results:unknown[]=[];
 await page.goto('/');await expect(page.getByText('API connected')).toBeVisible();
 const expected=['118575_clip','128058_clip','132831_clip','117092_clip','117093_clip','118576_clip','118577_clip','118578_clip','128057_clip','132877_clip'];
 for(const id of expected){
  console.info("Verifying",id);
  await expect(page.locator(`option[value="${id}"]`)).toBeEnabled();
  await page.getByLabel('Recording',{exact:true}).selectOption(id);
  const ready=rows.find((r:{recording_id:string})=>r.recording_id===id).artifact_ready;
  await expect(page.getByText(ready?'API connected':'Video connected')).toBeVisible();
  if(await page.getByRole('dialog').isVisible())await page.getByRole('button',{name:'Use neutral labels'}).click();
  const video=page.locator('video');console.info(id,'view loaded');
  await expect(video).toHaveAttribute('src',`/api/matches/${id}/video.mp4`);
  await expect.poll(()=>video.evaluate((v:HTMLVideoElement)=>v.readyState),{timeout:45000}).toBeGreaterThanOrEqual(2);
  console.info(id,'metadata loaded');await video.evaluate((v:HTMLVideoElement)=>{v.muted=true;void v.play()});
  await expect.poll(()=>video.evaluate((v:HTMLVideoElement)=>v.currentTime),{timeout:15000}).toBeGreaterThan(.2);
  console.info(id,'time advancing');await video.evaluate((v:HTMLVideoElement)=>v.pause());
  await video.evaluate((v:HTMLVideoElement)=>{v.currentTime=20});
  await expect.poll(()=>video.evaluate((v:HTMLVideoElement)=>v.seeking),{timeout:30000}).toBe(false);
  expect(await video.evaluate((v:HTMLVideoElement)=>v.error)).toBeNull();
  await expect(page.getByRole('img',{name:'Tactical pitch at 20.00 seconds'})).toBeVisible();
  const range=await page.request.get(`/api/matches/${id}/video.mp4`,{headers:{Range:'bytes=0-15'}});expect(range.status()).toBe(206);
  const view=(await (await page.request.get(`/api/matches/${id}/view`)).json());
  await page.getByLabel('Filter timeline').selectOption('all');
  const grouped=filterEvents(view.events,'all');
  const next=grouped.findIndex(e=>e.timestamp>=20);
  const event=grouped[Math.max(0,(next<0?grouped.length:next)-2)];
  expect(event).toBeTruthy();
  await page.locator('.event-row').first().click();
  await expect.poll(()=>video.evaluate((v:HTMLVideoElement)=>v.seeking),{timeout:30000}).toBe(false);
  const eventSeek=await video.evaluate((v:HTMLVideoElement)=>v.currentTime);expect(eventSeek).toBeLessThan(event.timestamp);
  const response=await page.request.post(`/api/matches/${id}/ask`,{data:{question:'How many passes were there altogether?',provider:'local'}});
  expect(response.status()).toBe(ready?200:503);
  if(ready){const answer=await response.json();expect(answer.answer.status).toBe('ok');expect(answer.conversation.recording_id).toBe(id);expect(answer.evidence.length).toBeGreaterThan(0)}
  else await expect(page.getByText('Match analysis is not available yet')).toBeVisible();
  if(ready&&id==='132877_clip'){
   await page.getByLabel('Answer provider').selectOption('local');
   const section=page.getByRole('region',{name:'Key Match Insights'});
   await section.getByText('Fastest tracked player · inferred',{exact:true}).click();
   await section.getByRole('button',{name:'Review 03:29.00'}).click();
   await expect(page.getByTestId('current-time')).toHaveText('03:27.00');
   await expect(page.getByRole('img',{name:'Tactical pitch at 207.00 seconds'})).toBeVisible();
   await expect.poll(()=>video.evaluate((v:HTMLVideoElement)=>v.currentTime),{timeout:30000}).toBeCloseTo(207,1);
   await page.getByRole('button',{name:'Who was the fastest player?'}).click();
   await expect(page.getByText('Looking up grounded match evidence…')).not.toBeVisible();
   await page.getByRole('button',{name:'Analyst',exact:true}).click();
   const fastest=section.locator(':scope > details').filter({has:page.getByText('Fastest tracked player · inferred',{exact:true})});
   await expect(fastest.getByText('Canonical provenance')).toBeVisible();
   await fastest.getByText('Canonical provenance').click();
   await expect(fastest.locator('code').first()).toContainText('player_summary.csv');
   await section.getByText('Difficult completed transfer · heuristic',{exact:true}).click();
   await section.getByRole('button',{name:'Review 02:21.28'}).click();
   await expect(page.getByTestId('current-time')).toHaveText('02:19.28');
   await expect(page.getByRole('img',{name:'Tactical pitch at 139.28 seconds'})).toBeVisible();
   await page.getByRole('button',{name:'Beginner',exact:true}).click();
  }
  results.push({recording_id:id,artifact_ready:ready,ask_status:response.status(),byte_range_status:range.status(),event_timestamp:event.timestamp,event_seek:eventSeek,tactical_loaded:true,...await video.evaluate((v:HTMLVideoElement)=>({duration:v.duration,readyState:v.readyState,currentTime:v.currentTime,error:v.error?.message??null}))});
  await writeFile('test-results/azure-clips-results.json',JSON.stringify(results,null,2));console.info(id,'verified');
 }
 await page.screenshot({path:'test-results/azure-clips.png',fullPage:true});
 await writeFile('test-results/azure-clips-results.json',JSON.stringify(results,null,2));
 expect(errors).toEqual([]);
});
