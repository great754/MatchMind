import {test,expect} from '@playwright/test';
import {writeFile} from 'node:fs/promises';
test('M5 timeline, partial provenance and canonical overlay agree with validated frame evidence',async({page})=>{
 const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));await page.goto('/');await page.getByLabel('Recording',{exact:true}).selectOption('test_match');
 const video=page.locator('video');await expect.poll(()=>video.evaluate((v:HTMLVideoElement)=>v.readyState)).toBeGreaterThanOrEqual(2);
 const metadata=await(await page.request.get('/api/matches/test_match/cv')).json();expect(metadata.camera_model_version).toBe('m5');expect(metadata.camera_timeline).toHaveLength(541);
 await expect(page.getByTestId('camera-tick')).toHaveCount(541);const results=[];
 for(const frame of [35,76,94,138,150,216,300,470,490]){
  const tick=page.locator(`[data-testid="camera-tick"][data-frame="${frame}"]`);await tick.click();await expect(page.getByTestId('cv-state')).toContainText(`Frame ${frame} ·`);
  await expect.poll(()=>video.evaluate((v:HTMLVideoElement)=>v.currentTime)).toBeCloseTo(frame/metadata.fps,4);
  const row=await(await page.request.get(`/api/matches/test_match/cv/frame?timestamp=${(frame+.0001)/metadata.fps}`)).json();const loc=row.camera.motion_evidence.localization;
  await expect(page.getByTestId('cv-camera-debug')).toHaveAttribute('data-status',loc.status);await expect(page.getByTestId('camera-timeline-selection')).toContainText(loc.outcome);
  await page.getByLabel('Projected pitch geometry',{exact:true}).check();
  if(row.camera.calibration_valid){
   expect(await page.getByTestId('projected-pitch-line').count()).toBeGreaterThan(0);
   if(loc.status==='PARTIAL_CONSTRAINED'){
    await expect(page.locator('svg[aria-label="CV canonical pitch"] polygon')).toHaveAttribute('stroke-dasharray','1 1');
    const current=row.entities.find((e:any)=>e.visibility==='visible'&&e.current_pitch);expect(current).toBeTruthy();
    const marker=page.locator(`[data-testid="cv-current-player"][data-source="partial_constrained"]`).first();await expect(marker).toHaveAttribute('fill','none');
    await page.locator(`[aria-label="Inspect ${current.entity_id}"] rect`).click();await expect(page.getByTestId('player-camera-evidence')).toContainText('PARTIAL_CONSTRAINED');await expect(page.getByTestId('player-camera-evidence')).toContainText('image_foot');
    expect(current.projection_camera.image_foot).toEqual(current.image_foot);expect(current.projection_camera.current_semantic_support).toBe(true);
   }
  }else{await expect(page.getByTestId('projected-pitch-line')).toHaveCount(0);await expect(page.getByTestId('cv-current-player')).toHaveCount(0);await expect(page.getByRole('region',{name:'Camera State'})).toContainText('Current positions are Unknown')}
  if(frame===470)await page.screenshot({path:'test-results/camera-placement-m5.png',fullPage:true});
  results.push({frame,status:loc.status,reason:loc.outcome,last_direct_age_seconds:loc.last_absolute_age_seconds,projected:row.entities.filter((e:any)=>e.current_pitch).length,projected_geometry:row.camera.projected_pitch_geometry.length});
 }
 const cvAsk=await page.request.post('/api/matches/test_match/ask',{data:{question:'How many passes?',provider:'local'}});expect(cvAsk.status()).toBe(503);expect(errors).toEqual([]);
 await writeFile('test-results/camera-placement-results.json',JSON.stringify({results,exact_timeline_seeking:true,current_foot_provenance:true,unknown_has_no_overlay:true,cv_ask_status:cvAsk.status(),page_errors:errors},null,2));
});
