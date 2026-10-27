import {test,expect} from '@playwright/test';
import {writeFile} from 'node:fs/promises';
test('persistent player, team and projection evidence survives camera loss and recording changes',async({page})=>{
 const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));await page.goto('/');await page.getByLabel('Recording',{exact:true}).selectOption('test_match');
 const video=page.locator('video');await expect.poll(()=>video.evaluate((v:HTMLVideoElement)=>v.readyState)).toBeGreaterThanOrEqual(2);
 const metadata=await(await page.request.get('/api/matches/test_match/cv')).json();const fps=metadata.fps;
 async function seek(frame:number){await video.evaluate((v:HTMLVideoElement,t)=>{v.pause();v.currentTime=t},(frame+.001)/fps);await expect.poll(()=>video.evaluate((v:HTMLVideoElement)=>v.seeking)).toBe(false);await expect(page.getByTestId('cv-state')).toContainText(`Frame ${frame} ·`);return (await(await page.request.get(`/api/matches/test_match/cv/frame?timestamp=${(frame+.001)/fps}`)).json())}
 const row=await seek(100);const visible=row.entities.filter((e:any)=>e.visibility==='visible');
 await expect(page.getByRole('region',{name:'World Model'})).toContainText(`Visible: ${visible.length}`);
 expect(await page.locator('.cv-video-overlay [role="button"]').count()).toBe(visible.length+row.unassigned.length);
 const green=visible.find((e:any)=>e.team==='team_b'&&e.image_bbox[0]<800);expect(green).toBeTruthy();
 await page.getByLabel('Team classification debug',{exact:true}).check();await page.locator(`[aria-label="Inspect ${green.entity_id}"] rect`).click();
 const inspection=page.getByRole('region',{name:'Selected CV player'});await expect(inspection).toContainText('team_b');await expect(inspection).toContainText('Track history:');await expect(inspection).toContainText('Kit evidence count:');await expect(page.getByLabel('Current torso crop',{exact:true})).toBeVisible();
 expect(await page.getByLabel('Current torso crop',{exact:true}).evaluate((c:HTMLCanvasElement)=>c.toDataURL().length)).toBeGreaterThan(1000);
 const center=await seek(216);expect(center.camera.calibration_valid).toBe(false);expect(center.camera.pitch_evidence.curves.length).toBeGreaterThan(0);
 expect(await page.locator('.cv-video-overlay [role="button"]').count()).toBeGreaterThan(0);await expect(page.getByTestId('cv-current-player')).toHaveCount(0);
 await page.getByText('Visible projection diagnostics',{exact:true}).click();await expect(page.getByText(/NO_CAMERA_GEOMETRY/).first()).toBeVisible();
 const historical=center.entities.find((e:any)=>e.visibility==='off_camera'&&e.team);expect(historical).toBeTruthy();expect(historical.current_pitch).toBeNull();expect(historical.team_memory.current_observation).toBeNull();
 const panel=page.getByRole('region',{name:'World Model'});await panel.getByText('Inspect persistent entities',{exact:true}).click();await panel.getByRole('button',{name:new RegExp('^'+historical.entity_id+' ·')}).click();
 await expect(inspection).toContainText(historical.team);await expect(inspection).toContainText('off_camera');await expect(inspection).toContainText('Position: Unknown');await expect(inspection).toContainText('No current torso crop');
 await panel.getByText('Inspect persistent entities',{exact:true}).click();await page.getByText('Visible projection diagnostics',{exact:true}).click();
 await seek(100);await page.locator(`[aria-label="Inspect ${green.entity_id}"] rect`).click();await page.screenshot({path:'test-results/player-world-m4.png',fullPage:true});
 await page.getByLabel('Recording',{exact:true}).selectOption('118575_clip');await expect(page.locator('video')).toHaveAttribute('src','/api/matches/118575_clip/video.mp4');await page.getByLabel('Recording',{exact:true}).selectOption('test_match');await expect(page.getByTestId('cv-state')).toContainText('Frame 0 ·');
 expect(errors).toEqual([]);await writeFile('test-results/player-world-results.json',JSON.stringify({frame100:{visible_entities:visible.length,unresolved_tracks:row.unassigned.length,selected_team:green.team,team_evidence_count:green.team_memory.evidence_count},center216:{image_people:center.entities.filter((e:any)=>e.visibility==='visible').length+center.unassigned.length,camera_valid:false,current_positions:0},historical_entity:historical.entity_id,historical_team:historical.team,team_crop:true,recording_isolation:true,page_errors:errors},null,2));
});
