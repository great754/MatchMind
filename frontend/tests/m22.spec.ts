import {test,expect} from '@playwright/test';
import {writeFile} from 'node:fs/promises';

test('trained M22 evidence, heatmaps, playback and grounded Ask stay isolated on all approved clips',async({page})=>{
 const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));let body:any,turn:any;
 await page.route('**/api/matches/*/assistant',async route=>{
  body={...route.request().postDataJSON(),provider:'local'};
  const response=await route.fetch({postData:JSON.stringify(body)});expect(response.status()).toBe(200);
  turn=await response.json();await route.fulfill({response,json:turn});
 });
 await expect.poll(async()=>{try{return(await page.request.get('/api/ready',{timeout:4000})).status()}catch{return 0}},{timeout:300000,intervals:[1000,2000,3000]}).toBe(200);
 const original=await(await page.request.get('/api/cv/review/export')).json();
 const withdrawn=await(await page.request.get('/api/cv/m21/cv/review/export')).json();const results=[];
 for(const name of ['test_match','team_goal','southampton']){
  await page.goto(`/?recording=${name}`);const picker=page.getByLabel('Ball review',{exact:true});
  await expect(picker).toHaveValue('m14',{timeout:120000});
  await expect(picker.locator('option[value=m22]')).toBeEnabled({timeout:120000});await picker.selectOption('m22');
  const panel=page.getByRole('region',{name:'M22 Temporal Football Detector'});
  await expect(panel.getByTestId('m22-state')).toContainText('Frame',{timeout:120000});
  const response=await page.request.get(`/api/matches/${name}/cv?ball_pipeline=m22`);expect(response.status()).toBe(200);const meta=await response.json();
  const metrics=await(await page.request.get(`/api/matches/${name}/cv/football`)).json();expect(metrics.promotion.status).toBe('not_promoted');expect(metrics.independent_test_status).toBe('not_available');
  expect(metrics.model.m22_annotation_snapshot_sha256).toMatch(/^[a-f0-9]{64}$/);
  expect(metrics.model.split.train_sources).not.toContain(name);expect(metrics.model.split.validation_sources).not.toContain(name);
  expect(metrics.model.split.shared_context_frames).toBe(0);expect(metrics.completed_review.frames).toBe(150);
  expect(meta.m22_checkpoint_sha256).toBe(metrics.checkpoint_sha256);
  const video=page.locator('.cv-video-stage video');await expect.poll(()=>video.evaluate((v:HTMLVideoElement)=>v.readyState),{timeout:45000}).toBeGreaterThanOrEqual(2);
  const target=meta.ball_examples.find((e:any)=>e.directly_observed)?.frame??Math.floor(meta.frames/2);
  await video.evaluate((v:HTMLVideoElement,t:number)=>{v.currentTime=t},(target+.001)/meta.fps);
  await expect(panel.getByTestId('m22-state')).toContainText(`Frame ${target} ·`);
  const detail=await(await page.request.get(`/api/matches/${name}/cv/football/frame?timestamp=${(target+.001)/meta.fps}`)).json();
  expect(detail.context_frames[2]).toBe(target);expect(detail.context_frames.every((f:number)=>f<=target)).toBe(true);expect(detail.ball_revision).toBe(meta.m22_ball_revision);
  if(!detail.directly_observed)await expect(page.getByTestId('ball-image-observed')).toHaveCount(0);
  await panel.getByRole('checkbox',{name:'Compare M20 and M22 side by side'}).check();
  await expect(panel.getByLabel('M20 comparison video')).toBeVisible();await expect(panel.getByLabel('M22 comparison video')).toBeVisible();
  await panel.getByRole('checkbox',{name:'Inspect current-frame activation map'}).check();await expect(panel.getByRole('img',{name:'M22 source-aligned heatmap'})).toBeVisible();
  await video.evaluate((v:HTMLVideoElement)=>{v.muted=true;return v.play()});await page.waitForTimeout(400);await video.evaluate((v:HTMLVideoElement)=>v.pause());
  await panel.getByText('Compare reviewed results',{exact:true}).click();await expect(panel.getByRole('table')).toContainText('M22 trained temporal detector');
  await panel.getByText('Results from your completed M22 review',{exact:true}).click();
  await expect(panel.getByRole('table',{name:'Completed M22 review comparison'})).toContainText('Retrained M22');
  await expect(panel.getByText(/not an independent final test/)).toBeVisible();
  await page.screenshot({path:`test-results/m22-${name}.jpg`,type:'jpeg',quality:70,fullPage:true});
  await page.getByRole('button',{name:'How many passes were detected?',exact:true}).click();await expect(page.getByTestId('assistant-answer')).toBeVisible({timeout:90000});
  expect(body.cv_pipeline).toBe('m22');expect(body.cv_experiment??null).toBeNull();expect(turn.cv_pipeline).toBe('m22');expect(turn.cv_model_revision).toContain(meta.m22_ball_revision);
  expect(turn.events.every((e:any)=>e.safe_for_automatic)).toBe(true);
  await picker.selectOption('m20');await expect(page.getByTestId('assistant-answer')).toHaveCount(0);await expect(panel).toHaveCount(0);
  results.push({name,source_playback:true,source_seeking:true,comparison:true,heatmap:true,checkpoint_isolated:true,grounded_local_ask:true,original_default:'m14'});
 }
 expect(await(await page.request.get('/api/cv/review/export')).json()).toEqual(original);
 expect(await(await page.request.get('/api/cv/m21/cv/review/export')).json()).toEqual(withdrawn);expect(errors).toEqual([]);
 await writeFile('test-results/m22-results.json',JSON.stringify({results,page_errors:errors,original_and_withdrawn_labels_preserved:true},null,2));
});
