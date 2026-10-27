import {test,expect} from '@playwright/test';
import {writeFile} from 'node:fs/promises';
test('M6 both CV recordings expose measured camera, marker legend and persistent team evidence',async({page})=>{
 const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));const results=[];await page.goto('/');
 for(const id of ['team_goal','test_match']){
  await page.getByLabel('Recording',{exact:true}).selectOption(id);const video=page.locator('video');await expect.poll(()=>video.evaluate((v:HTMLVideoElement)=>v.readyState)).toBeGreaterThanOrEqual(2);const meta=await(await page.request.get(`/api/matches/${id}/cv`)).json();expect(meta.camera_model_version).toBe('m6');
  await expect(page.getByRole('region',{name:'Player marker legend'})).toContainText('BIDIRECTIONAL');await expect(page.getByRole('region',{name:'Player marker legend'})).toContainText('Position unknown');
  await page.getByLabel('Camera motion debug',{exact:true}).check();await page.getByLabel('Projected pitch geometry',{exact:true}).check();
  const frames=id==='team_goal'?[1,180,250,349,350,400]:[35,100,216,470];
  for(const frame of frames){
   await page.locator(`[data-testid="camera-tick"][data-frame="${frame}"]`).click();await expect(page.getByTestId('cv-state')).toContainText(`Frame ${frame} ·`);const row=await(await page.request.get(`/api/matches/${id}/cv/frame?timestamp=${(frame+.001)/meta.fps}`)).json();const loc=row.camera.motion_evidence.localization;
   await expect(page.getByTestId('cv-camera-debug')).toHaveAttribute('data-status',loc.status);await expect(page.getByTestId('temporal-diagnostics')).toBeVisible();await expect(page.getByTestId('camera-motion-overlay')).toHaveCount(1);
   if(id==='team_goal'&&frame===1){expect(loc.status).toBe('TEMPORAL_PREDICTED');await expect(page.getByTestId('temporal-diagnostics')).toContainText('future evidence');expect(row.camera.motion_evidence.temporal_camera.absolute_anchor_frame).toBe(2)}
   if(id==='team_goal'&&frame===350){expect(row.camera.calibration_valid).toBe(false);await expect(page.getByTestId('cv-current-player')).toHaveCount(0);await expect(page.getByTestId('projected-pitch-line')).toHaveCount(0)}
   results.push({id,frame,state:loc.status,reason:loc.outcome,flow_matches:row.camera.motion_evidence.temporal_camera.field_motion.matches,topline:row.camera.motion_evidence.temporal_camera.field_motion.top_touchline_tracked});
  }
  const target=meta.camera_timeline.find((t:any)=>t.state==='TEMPORAL_CORRECTED')??meta.camera_timeline.find((t:any)=>t.state==='PARTIAL_CONSTRAINED');expect(target).toBeTruthy();await page.locator(`[data-testid="camera-tick"][data-frame="${target.frame}"]`).click();await expect(page.getByTestId('cv-state')).toContainText(`Frame ${target.frame} ·`);await expect(page.getByTestId('predicted-camera-region')).toHaveCount(1);
  const frame=await(await page.request.get(`/api/matches/${id}/cv/frame?timestamp=${(target.frame+.001)/meta.fps}`)).json();const person=frame.entities.find((e:any)=>e.visibility==='visible'&&e.team_belief);expect(person).toBeTruthy();await page.locator(`[aria-label="Inspect ${person.entity_id}"] rect`).click();await page.getByLabel('Team classification debug',{exact:true}).check();await expect(page.getByTestId('team-belief-evidence')).toBeVisible();await expect(page.getByTestId('player-camera-evidence')).toContainText('Temporal direction');await page.getByRole('button',{name:'+1 frame',exact:true}).click();await expect(page.getByTestId('cv-state')).toContainText(`Frame ${target.frame+1} ·`);
  await page.screenshot({path:`test-results/${id}-m6.png`,fullPage:true});
  const ask=await page.request.post(`/api/matches/${id}/ask`,{data:{question:'Who scored?',provider:'local'}});expect(ask.status()).toBe(503);
 }
 expect(errors).toEqual([]);await writeFile('test-results/temporal-world-results.json',JSON.stringify({results,both_videos:true,debug:true,legend:true,team_evidence:true,cv_analytics_isolated:true,page_errors:errors},null,2));
});
