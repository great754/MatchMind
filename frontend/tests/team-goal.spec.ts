import {test,expect} from '@playwright/test';
import {writeFile} from 'node:fs/promises';
test('new team_goal video plays and its own CV timeline and people are interactive',async({page})=>{
 const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));await page.goto('/');
 const listing=await(await page.request.get('/api/recordings')).json();const row=listing.recordings.find((r:any)=>r.recording_id==='team_goal');expect(row.status).toBe('available');expect(row.capabilities.cv_world_model_available).toBe(true);expect(row.capabilities.ask_available).toBe(false);
 await page.getByLabel('Recording',{exact:true}).selectOption('team_goal');const video=page.locator('video');await expect(video).toHaveAttribute('src','/api/matches/team_goal/video.mp4');await expect.poll(()=>video.evaluate((v:HTMLVideoElement)=>v.readyState)).toBeGreaterThanOrEqual(2);
 expect(await video.evaluate((v:HTMLVideoElement)=>[v.videoWidth,v.videoHeight])).toEqual([640,360]);
 const metadata=await(await page.request.get('/api/matches/team_goal/cv')).json();expect(metadata.recording_id).toBe('team_goal');expect(metadata.frames).toBe(500);expect(metadata.fps).toBe(25);expect(metadata.camera_timeline).toHaveLength(500);await expect(page.getByTestId('camera-tick')).toHaveCount(500);
 await video.evaluate((v:HTMLVideoElement)=>{v.muted=true;void v.play()});await expect.poll(()=>video.evaluate((v:HTMLVideoElement)=>v.currentTime)).toBeGreaterThan(.3);await video.evaluate((v:HTMLVideoElement)=>v.pause());
 const sampled=[];const frameSet=new Set<number>([0,125,250,375,499]);const supported=metadata.camera_timeline.find((t:any)=>['DIRECT','RELOCALIZED','PARTIAL_CONSTRAINED'].includes(t.state));if(supported)frameSet.add(supported.frame);
 for(const frame of frameSet){
  await page.locator(`[data-testid="camera-tick"][data-frame="${frame}"]`).click();await expect(page.getByTestId('cv-state')).toContainText(`Frame ${frame} ·`);
  const state=await(await page.request.get(`/api/matches/team_goal/cv/frame?timestamp=${(frame+.001)/25}`)).json();expect(state.recording_id).toBe('team_goal');const loc=state.camera.motion_evidence.localization;await expect(page.getByTestId('cv-camera-debug')).toHaveAttribute('data-status',loc.status);
  await page.getByLabel('Projected pitch geometry',{exact:true}).check();
  if(state.camera.calibration_valid)expect(await page.getByTestId('projected-pitch-line').count()).toBeGreaterThan(0);else {await expect(page.getByTestId('projected-pitch-line')).toHaveCount(0);await expect(page.getByTestId('cv-current-player')).toHaveCount(0)}
  const visible=state.entities.filter((e:any)=>e.visibility==='visible');expect(await page.locator('.cv-video-overlay [role="button"]').count()).toBe(visible.length+state.unassigned.length);
  sampled.push({frame,state:loc.status,reason:loc.outcome,visible_entities:visible.length,unresolved_tracks:state.unassigned.length,projected:visible.filter((e:any)=>e.current_pitch).length+state.unassigned.filter((u:any)=>u.observation.pitch).length});
 }
 const inspectFrame=supported?.frame??125;await page.locator(`[data-testid="camera-tick"][data-frame="${inspectFrame}"]`).click();await expect(page.getByTestId('cv-state')).toContainText(`Frame ${inspectFrame} ·`);
 const boxes=page.locator('.cv-video-overlay [role="button"] rect');expect(await boxes.count()).toBeGreaterThan(0);await boxes.first().click();await expect(page.getByRole('region',{name:'Selected CV player'})).toContainText('Temporary track');await expect(page.getByTestId('player-camera-evidence')).toBeVisible();
 await page.getByLabel('Team classification debug',{exact:true}).check();await expect(page.getByRole('region',{name:'Selected CV player'})).toContainText('Kit');
 await page.getByRole('button',{name:'+1 frame',exact:true}).click();await expect(page.getByTestId('cv-state')).toContainText(`Frame ${Math.min(499,inspectFrame+1)} ·`);
 await page.screenshot({path:'test-results/team-goal-review.png',fullPage:true});
 await page.getByLabel('Recording',{exact:true}).selectOption('test_match');await expect(page.locator('video')).toHaveAttribute('src','/api/matches/test_match/video.mp4');await expect(page.getByTestId('camera-tick')).toHaveCount(541);await expect(page.getByTestId('cv-state')).toContainText('Frame 0 ·');
 await page.getByLabel('Recording',{exact:true}).selectOption('team_goal');await expect(page.getByTestId('cv-state')).toContainText('Frame 0 ·');await expect(page.getByTestId('camera-tick')).toHaveCount(500);
 const range=await page.request.get('/api/matches/team_goal/video.mp4',{headers:{Range:'bytes=0-15'}});expect(range.status()).toBe(206);expect((await range.body()).length).toBe(16);expect(errors).toEqual([]);expect(await page.locator('video').evaluate((v:HTMLVideoElement)=>v.error)).toBeNull();
 await writeFile('test-results/team-goal-results.json',JSON.stringify({metadata:{frames:metadata.frames,fps:metadata.fps,width:metadata.width,height:metadata.height},sampled,playback:true,timeline:true,step:true,person_inspection:true,recording_isolation:true,range_status:range.status(),page_errors:errors},null,2));
});
