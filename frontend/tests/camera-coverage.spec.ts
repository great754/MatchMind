import {test,expect} from '@playwright/test';
import {writeFile} from 'node:fs/promises';
test('M7 persistent field trajectories, recovered geometry and unknown gap remain inspectable',async({page})=>{
 const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));await page.goto('/');const sampled=[];
 for(const id of ['test_match','team_goal']){
  await page.getByLabel('Recording',{exact:true}).selectOption(id);const video=page.locator('video');await expect.poll(()=>video.evaluate((v:HTMLVideoElement)=>v.readyState)).toBeGreaterThanOrEqual(2);
  const meta=await(await page.request.get(`/api/matches/${id}/cv`)).json();expect(['m7','m8']).toContain(meta.camera_model_version);
  await page.getByLabel('Camera motion debug',{exact:true}).check();await page.getByLabel('Projected pitch geometry',{exact:true}).check();
  for(const frame of id==='test_match'?[110,120,129,146,150,151,216,458]:[1,60,180,250,349,350,400,477]){
   await page.locator(`[data-testid="camera-tick"][data-frame="${frame}"]`).click();await expect(page.getByTestId('cv-state')).toContainText(`Frame ${frame} ·`);const row=await(await page.request.get(`/api/matches/${id}/cv/frame?timestamp=${(frame+.001)/meta.fps}`)).json();const c=row.camera,d=c.motion_evidence.temporal_camera;
   await expect(page.getByTestId('cv-camera-debug')).toHaveAttribute('data-status',c.motion_evidence.localization.status);await expect(page.getByTestId('camera-motion-overlay')).toHaveCount(1);
   if(d.world_field&&d.trajectory_uncertainty_px!=null){await expect(page.getByTestId('persistent-field-diagnostics')).toBeVisible();await expect(page.getByTestId('persistent-field-diagnostics')).toContainText('Trajectory uncertainty budget')}
   if(id==='test_match'&&frame===120){expect(c.calibration_valid).toBe(true);expect(c.calibration_source_kind).toBe('bidirectional_interpolated');expect(d.world_field.active_world_anchors).toBeGreaterThan(40);await expect(page.getByTestId('persistent-field-anchors')).toBeVisible();expect(await page.getByTestId('persistent-field-anchors').locator('circle').count()).toBeGreaterThan(0);await page.screenshot({path:'test-results/test_match-m7-recovered.png',fullPage:true})}
   if((id==='test_match'&&frame===216)||(id==='team_goal'&&frame===350)){expect(c.calibration_valid).toBe(false);await expect(page.getByTestId('cv-current-player')).toHaveCount(0);await expect(page.getByTestId('projected-pitch-line')).toHaveCount(0);expect(row.entities.some((e:any)=>e.visibility==='visible')).toBe(true)}
   sampled.push({id,frame,valid:c.calibration_valid,state:c.motion_evidence.localization.status,reason:c.source,world_anchors:d.world_field?.active_world_anchors,origin:d.independent_anchor_proof});
  }
  await page.screenshot({path:`test-results/${id}-m7.png`,fullPage:true});const ask=await page.request.post(`/api/matches/${id}/ask`,{data:{question:'Who scored?',provider:'local'}});expect(ask.status()).toBe(503);
 }
 expect(errors).toEqual([]);await writeFile('test-results/camera-coverage-results.json',JSON.stringify({sampled,page_errors:errors},null,2));
});
