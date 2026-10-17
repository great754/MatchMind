import {expect,test} from '@playwright/test';
import {writeFile} from 'node:fs/promises';
test('all Azure clips select, play and seek while analysis remains gated by validation',async({page})=>{
 test.skip(!process.env.TACTIQAI_AZURE_BROWSER,'Explicit live Azure integration opt-in required.');
 const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));
 const listing=await page.request.get('/api/recordings');expect(listing.status()).toBe(200);
 const rows=(await listing.json()).recordings;
 if(process.env.TACTIQAI_AZURE_REQUIRE_ANALYSIS==='1'){
  expect(rows).toHaveLength(10);
  expect(rows.every((r:{artifact_ready:boolean,status:string})=>r.artifact_ready&&r.status==='available')).toBe(true);
 }
 const results:unknown[]=[];
 await page.goto('/');await expect(page.getByText('API connected')).toBeVisible();
 const expected=['118575_clip','128058_clip','132831_clip','117092_clip','117093_clip','118576_clip','118577_clip','118578_clip','128057_clip','132877_clip'];
 for(const id of expected){
  await expect(page.locator(`option[value="${id}"]`)).toBeEnabled();
  await page.getByLabel('Recording',{exact:true}).selectOption(id);
  const ready=rows.find((r:{recording_id:string})=>r.recording_id===id).artifact_ready;
  await expect(page.getByText(ready?'API connected':'Video connected')).toBeVisible();
  if(await page.getByRole('dialog').isVisible())await page.getByRole('button',{name:'Use neutral labels'}).click();
  const video=page.locator('video');
  await expect(video).toHaveAttribute('src',`/api/matches/${id}/video.mp4`);
  await expect.poll(()=>video.evaluate((v:HTMLVideoElement)=>v.readyState),{timeout:45000}).toBeGreaterThanOrEqual(2);
  await video.evaluate((v:HTMLVideoElement)=>v.play());
  await expect.poll(()=>video.evaluate((v:HTMLVideoElement)=>v.currentTime),{timeout:15000}).toBeGreaterThan(.2);
  await video.evaluate((v:HTMLVideoElement)=>v.pause());
  await video.evaluate((v:HTMLVideoElement)=>{v.currentTime=20});
  await expect.poll(()=>video.evaluate((v:HTMLVideoElement)=>v.seeking),{timeout:30000}).toBe(false);
  const response=await page.request.post(`/api/matches/${id}/ask`,{data:{question:'How many passes were there altogether?',provider:'local'}});
  expect(response.status()).toBe(ready?200:503);
  if(ready){const answer=await response.json();expect(answer.answer.status).toBe('ok');expect(answer.conversation.recording_id).toBe(id);expect(answer.evidence.length).toBeGreaterThan(0)}
  else await expect(page.getByText('Match analysis is not available yet')).toBeVisible();
  results.push({recording_id:id,artifact_ready:ready,ask_status:response.status(),...await video.evaluate((v:HTMLVideoElement)=>({duration:v.duration,readyState:v.readyState,currentTime:v.currentTime,error:v.error?.message??null}))});
 }
 await page.screenshot({path:'test-results/azure-clips.png',fullPage:true});
 await writeFile('test-results/azure-clips-results.json',JSON.stringify(results,null,2));
 expect(errors).toEqual([]);
});
