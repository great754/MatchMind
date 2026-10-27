import {test,expect} from '@playwright/test';
import {writeFile} from 'node:fs/promises';
test('CV recording plays, seeks, steps and exposes uncertainty in the existing website',async({page})=>{
 const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto('/');await page.getByLabel('Recording',{exact:true}).selectOption('test_match');
 await expect(page.getByText('CV Experimental',{exact:true})).toBeVisible();
 const video=page.locator('video');await expect.poll(()=>video.evaluate((v:HTMLVideoElement)=>v.readyState)).toBeGreaterThanOrEqual(2);
 await video.evaluate((v:HTMLVideoElement)=>{v.muted=true;void v.play()});await expect.poll(()=>video.evaluate((v:HTMLVideoElement)=>v.currentTime)).toBeGreaterThan(.5);await expect.poll(async()=>Number((await page.getByTestId('cv-state').textContent())?.match(/Frame (\d+)/)?.[1]??0)).toBeGreaterThan(0);await video.evaluate((v:HTMLVideoElement)=>v.pause());
 const checks=[];
 for(const t of [3,9,15,21]){
  await video.evaluate((v:HTMLVideoElement,time)=>{v.currentTime=time},t);
  const frame=Math.floor(t*24000/1001);
  await expect(page.getByTestId('cv-state')).toContainText(`Frame ${frame} ·`);
  const result=await page.request.get(`/api/matches/test_match/cv/frame?timestamp=${t}`);const row=await result.json();
  expect(row.frame).toBe(frame);if(!row.camera.calibration_valid)expect(row.camera.visible_pitch_region).toBeNull();else expect(row.camera.visible_pitch_region.length).toBeGreaterThanOrEqual(3);
  checks.push({time:t,frame:row.frame,calibration:row.camera.calibration_status,visible:row.entities.filter((e:{visibility:string})=>e.visibility==='visible').length,unknown_positions:row.entities.every((e:{current_pitch:unknown})=>e.current_pitch===null)});
 }
 const metadata=await (await page.request.get('/api/matches/test_match/cv')).json();
 const acquired=metadata.transitions.find((t:{label:string})=>t.label==='Calibration valid');
 const propagated=metadata.transitions.find((t:{label:string})=>t.label==='Camera propagated');
 const lost=metadata.transitions.find((t:{label:string})=>t.label==='Calibration lost');
 expect(acquired).toBeTruthy();expect(propagated).toBeTruthy();expect(lost).toBeTruthy();
 for(const transition of [acquired,propagated,lost]){
  await video.evaluate((v:HTMLVideoElement,t)=>{v.currentTime=t+0.0001},transition.timestamp);
  await expect(page.getByTestId('cv-state')).toContainText(`Frame ${transition.frame} ·`);
  const row=await (await page.request.get(`/api/matches/test_match/cv/frame?timestamp=${transition.timestamp+0.0001}`)).json();
  await expect(page.getByTestId('cv-camera-debug')).toHaveAttribute('data-status',row.camera.motion_evidence.localization.status);
  if(transition===lost){await expect(page.getByRole('region',{name:'Camera State'})).toContainText('Localization unavailable');expect(row.camera.calibration_valid).toBe(false);await expect(page.locator('[data-testid="cv-current-player"]')).toHaveCount(0)}
  else{expect(row.camera.calibration_valid).toBe(true);await expect(page.locator('svg[aria-label="CV canonical pitch"] polygon')).toBeVisible()}
  if(transition===acquired)expect(await page.getByTestId('cv-current-player').count()).toBeGreaterThan(0);
  if(transition===propagated)await expect(page.locator('svg[aria-label="CV canonical pitch"] polygon').first()).toHaveAttribute('stroke-dasharray','2 1');
  checks.push({transition:transition.label,frame:row.frame,position_source:row.camera.calibration_source_kind,calibration_valid:row.camera.calibration_valid});
 }
 const relocalized=metadata.transitions.find((t:{transition:string})=>t.transition==='relocalization');
 expect(relocalized).toBeTruthy();
 await video.evaluate((v:HTMLVideoElement,t)=>{v.currentTime=t+0.0001},relocalized.timestamp);
 await expect(page.getByTestId('cv-state')).toContainText(`Frame ${relocalized.frame} ·`);
 await expect(page.getByTestId('cv-camera-debug')).toHaveAttribute('data-status','RELOCALIZED');
 const lastAcquired=metadata.transitions.filter((t:{label:string})=>t.label==='Calibration valid').at(-1);
 await page.getByRole('button',{name:`Calibration valid · ${lastAcquired.timestamp.toFixed(2)}s`,exact:true}).click();
 await expect.poll(()=>video.evaluate((v:HTMLVideoElement)=>v.currentTime)).toBeCloseTo(Math.max(0,lastAcquired.timestamp-2),2);
 await video.evaluate((v:HTMLVideoElement)=>{v.currentTime=21});await expect(page.getByTestId('cv-state')).toContainText('Frame 503 ·');
 const before=await page.getByTestId('cv-state').textContent();
 await page.getByLabel('Player boxes',{exact:true}).uncheck();await page.getByLabel('Pitch landmarks',{exact:true}).uncheck();await page.getByLabel('Show last-seen positions',{exact:true}).check();
 expect(await page.getByTestId('cv-state').textContent()).toBe(before);
 await page.locator('.cv-video-overlay [role="button"]').last().locator('rect').click({timeout:5000});
 await expect(page.getByRole('region',{name:'Selected CV player'})).toContainText('Unknown');
 await page.getByRole('button',{name:'+1 frame',exact:true}).click();await expect(page.getByTestId('cv-state')).toContainText('Frame 504 ·');
 await page.getByLabel('Calibration evidence',{exact:true}).check();
 await video.evaluate((v:HTMLVideoElement,t)=>{v.currentTime=t+0.0001},propagated.timestamp);
 await expect(page.getByTestId('cv-state')).toContainText(`Frame ${propagated.frame} ·`);
 await page.screenshot({path:'test-results/cv-review-m4.png',fullPage:true});
 expect(errors).toEqual([]);expect(await video.evaluate((v:HTMLVideoElement)=>v.error)).toBeNull();
 await writeFile('test-results/cv-review-results.json',JSON.stringify({checks,play:true,seek:true,step:true,overlay_presentation_only:true,person_inspection:true,page_errors:errors},null,2));
});
