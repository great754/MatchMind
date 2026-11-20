import {test,expect} from '@playwright/test';
import {writeFile} from 'node:fs/promises';

test('M21 compares frozen source frames and isolates experiment answers on all three clips',async({page})=>{
 const errors:string[]=[];page.on('pageerror',error=>errors.push(error.message));let lastBody:any,lastTurn:any;
 await page.route('**/api/matches/*/assistant',async route=>{
  lastBody={...route.request().postDataJSON(),provider:'local'};
  const response=await route.fetch({postData:JSON.stringify(lastBody)});expect(response.status()).toBe(200);
  lastTurn=await response.json();await route.fulfill({response,json:lastTurn});
 });
 await expect.poll(async()=>{try{return(await page.request.get('/api/ready',{timeout:4000})).status()}catch{return 0}},{timeout:300000,intervals:[1000,2000,3000]}).toBe(200);
 const original=await(await page.request.get('/api/cv/review/export')).json();
 const results:any[]=[];
 await page.goto('/?recording=test_match');
 for(const [name,count] of [['test_match',1],['team_goal',0],['southampton',0]] as const){
  if(name!=='test_match')await page.getByLabel('Recording',{exact:true}).selectOption(name);
  const picker=page.getByLabel('Ball review',{exact:true});await expect(picker.locator('option[value=m21]')).toBeEnabled({timeout:120000});await picker.selectOption('m21');
  const panel=page.getByRole('region',{name:'M21 Advanced Perception'});
  await expect(panel.getByTestId('m21-state')).toContainText('Frame',{timeout:120000});await expect(page.getByTestId('cv-pass-count')).toHaveText(String(count));
  await expect(panel.getByText(/new independent test labels are pending/i)).toBeVisible();
  const meta=await(await page.request.get(`/api/matches/${name}/cv?ball_pipeline=m21`)).json();
  const source=page.locator('.cv-video-stage video');await expect.poll(()=>source.evaluate((v:HTMLVideoElement)=>v.readyState),{timeout:45000}).toBeGreaterThanOrEqual(2);
  const example=meta.ball_examples.find((e:any)=>e.directly_observed);if(example){
   await source.evaluate((v:HTMLVideoElement,t:number)=>{v.currentTime=t},(example.frame+.001)/meta.fps);
   await expect(panel.getByTestId('m21-state')).toContainText(`Frame ${example.frame} ·`);
  }
  await panel.getByRole('checkbox',{name:'Compare M20 and M21 side by side'}).check();
  await expect(panel.getByLabel('M20 comparison video')).toBeVisible();await expect(panel.getByLabel('M21 comparison video')).toBeVisible();
  await source.evaluate((v:HTMLVideoElement)=>{v.muted=true;return v.play()});await page.waitForTimeout(500);await source.evaluate((v:HTMLVideoElement)=>v.pause());
  await page.screenshot({path:`test-results/m21-${name}.png`,fullPage:true});
  await page.getByRole('button',{name:'How many passes were detected?',exact:true}).click();
  await expect(page.getByTestId('assistant-answer')).toBeVisible({timeout:90000});expect(lastBody.cv_pipeline).toBe('m21');expect(lastTurn.cv_pipeline).toBe('m21');
  await panel.getByLabel('M21 experiment',{exact:true}).selectOption('temporal_offline');
  await expect(page.getByTestId('assistant-answer')).toHaveCount(0);
  await expect(panel.getByTestId('m21-state')).toContainText('Offline',{timeout:120000});
  await expect(page.getByText('M21 ball evidence',{exact:true})).toBeVisible();
  await page.getByRole('button',{name:'How many passes were detected?',exact:true}).click();
  await expect(page.getByTestId('assistant-answer')).toBeVisible({timeout:90000});expect(lastBody.cv_experiment).toBe('temporal_offline');expect(lastBody.previous_turn).toBeNull();expect(lastTurn.cv_experiment).toBe('temporal_offline');
  expect(lastTurn.events.every((e:any)=>e.safe_for_automatic)).toBe(true);
  await picker.selectOption('m20');await expect(page.getByTestId('assistant-answer')).toHaveCount(0);await expect(panel).toHaveCount(0);
  results.push({name,source_playback:true,comparison_canvas:true,qualified_control_passes:count,experiment_context_isolated:true});
 }
 expect(await(await page.request.get('/api/cv/review/export')).json()).toEqual(original);expect(errors).toEqual([]);
 await writeFile('test-results/m21-results.json',JSON.stringify({results,page_errors:errors,original_labels_preserved:true},null,2));
});

test('withdrawn M21 review shows a notice and never writes or alters original labels',async({page})=>{
 const original=await(await page.request.get('/api/cv/review/export')).json();
 const before=await(await page.request.get('/api/cv/m21/cv/review/export')).json();let writes=0;
 page.on('request',request=>{if(request.method()==='POST'&&request.url().includes('/cv/review/labels'))writes++});
 await page.goto('/?review=m21&recording=m21_holdout_1');
 await expect(page.getByRole('heading',{name:'M21 review withdrawn'})).toBeVisible({timeout:90000});
 await expect(page.getByRole('link',{name:'Open M22 football training review'})).toBeVisible();
 await expect(page.getByRole('button',{name:'Save human reference'})).toHaveCount(0);
 await expect(page.getByRole('region',{name:'M21 Advanced Perception'})).toHaveCount(0);
 expect(writes).toBe(0);expect(await(await page.request.get('/api/cv/m21/cv/review/export')).json()).toEqual(before);
 expect(await(await page.request.get('/api/cv/review/export')).json()).toEqual(original);
 // No held-out source screenshot is inspected or used for model selection.
});
