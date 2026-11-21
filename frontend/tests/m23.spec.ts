import {test,expect} from '@playwright/test';
import {writeFile} from 'node:fs/promises';

test('M23 source, primary comparison, diagnostics and grounded Ask are isolated on all approved clips',async({page})=>{
 const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));let body:any,turn:any;
 await page.route('**/api/matches/*/assistant',async route=>{
  body={...route.request().postDataJSON(),provider:'local'};
  const response=await route.fetch({postData:JSON.stringify(body)});expect(response.status()).toBe(200);
  turn=await response.json();await route.fulfill({response,json:turn});
 });
 const results=[];
 for(const name of ['test_match','team_goal','southampton']){
  await page.goto(`/?recording=${name}`);
  const picker=page.getByLabel('Ball review',{exact:true});
  await expect(picker).toHaveValue('m14',{timeout:120000});
  await expect(picker.locator('option[value=m23]')).toBeEnabled({timeout:120000});
  await picker.selectOption('m23');
  const panel=page.getByRole('region',{name:'M23 intelligent ball tracking'});
  await expect(panel.getByTestId('m23-state')).toContainText('Frame',{timeout:120000});
  await expect(panel.getByRole('table',{name:'M23 controlled comparisons'})).toBeVisible();
  await expect(panel.getByLabel('M21 comparison video')).toBeVisible();
  await expect(panel.getByLabel('M23 comparison video')).toBeVisible();
  const meta=await(await page.request.get(`/api/matches/${name}/cv?ball_pipeline=m23`)).json();
  const report=await(await page.request.get(`/api/matches/${name}/cv/intelligent`)).json();
  expect(report.promotion.status).toBe('not_promoted');expect(report.reference_role).toBe('development_only');
  expect(report.model_set_sha256).toBe(meta.m23_model_set_sha256);
  const video=page.locator('.cv-video-stage video');
  await expect.poll(()=>video.evaluate((v:HTMLVideoElement)=>v.readyState),{timeout:45000}).toBeGreaterThanOrEqual(2);
  const diagnostic=report.metrics.navigation.find((n:any)=>n.kind==='improved_reviewed_frame')??report.metrics.navigation[0];
  expect(diagnostic).toBeTruthy();
  const example=panel.getByRole('button',{name:`${diagnostic.kind.replaceAll('_',' ')} · frame ${diagnostic.frame}`,exact:true}).first();
  await example.click();
  await expect(panel.getByTestId('m23-state')).toContainText(`Frame ${diagnostic.frame} ·`);
  await expect(page.getByTestId('m23-overlay')).toHaveAttribute('data-frame',String(diagnostic.frame));
  const detail=await(await page.request.get(`/api/matches/${name}/cv/intelligent/frame?timestamp=${(diagnostic.frame+.001)/meta.fps}`)).json();
  const reference=await(await page.request.get(`/api/matches/${name}/cv/intelligent/reference?timestamp=${(diagnostic.frame+.001)/meta.fps}`)).json();
  expect(detail.frame).toBe(reference.frame);expect(detail.model_set_sha256).toBe(meta.m23_model_set_sha256);expect(detail.ball_revision).toBe(meta.m23_ball_revision);
  expect(detail.future_visual_frames_used).toBe(false);expect(detail.m22_context_frames.every((f:number)=>f<=detail.frame)).toBe(true);
  expect(detail.candidates.filter((c:any)=>c.selected).every((c:any)=>c.observation_eligible&&c.confidence>=.35)).toBe(true);
  await panel.getByLabel('M23 evaluation sample').selectOption('original');
  await expect(panel.getByRole('table')).toContainText('Known passes');
  await panel.getByLabel('M23 evaluation sample').selectOption('new_150');
  await video.evaluate((v:HTMLVideoElement)=>{v.muted=true;return v.play()});
  await expect.poll(()=>video.evaluate((v:HTMLVideoElement)=>v.currentTime)).toBeGreaterThan(diagnostic.frame/meta.fps+.2);
  await video.evaluate((v:HTMLVideoElement)=>v.pause());
  await page.screenshot({path:`test-results/m23-${name}.jpg`,type:'jpeg',quality:70,fullPage:true});
  await page.getByRole('button',{name:'How many passes were detected?',exact:true}).click();
  await expect(page.getByTestId('assistant-answer')).toBeVisible({timeout:90000});
  expect(body.cv_pipeline).toBe('m23');expect(turn.cv_pipeline).toBe('m23');
  expect(turn.cv_model_revision).toContain(meta.m23_ball_revision);
  expect(turn.events.every((e:any)=>e.safe_for_automatic)).toBe(true);
  await picker.selectOption('m20');
  await expect(page.getByTestId('assistant-answer')).toHaveCount(0);await expect(panel).toHaveCount(0);
  results.push({recording:name,source_playback:true,synchronized_comparison:true,diagnostic_navigation:true,
   model_revision_isolated:true,grounded_local_ask:true,ordinary_default:'m14',metrics:report.metrics.new_150.ball});
 }
 expect(errors).toEqual([]);
 await writeFile('test-results/m23-results.json',JSON.stringify({results,page_errors:errors},null,2));
});
