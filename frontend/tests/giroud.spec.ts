import {test,expect} from '@playwright/test';

test('Giroud plays with source-specific M23 evidence and unlabelled results',async({page})=>{
 const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));
 let answer:any;
 await page.route('**/api/matches/giroud/assistant',async route=>{
  const payload={...route.request().postDataJSON(),provider:'local'};
  const response=await route.fetch({postData:JSON.stringify(payload)});
  expect(response.status()).toBe(200);answer=await response.json();
  await route.fulfill({response,json:answer});
 });
 await page.goto('/?recording=giroud&ball_pipeline=m23');
 const picker=page.getByLabel('Ball review',{exact:true});
 await expect(picker.locator('option[value=m23]')).toBeEnabled({timeout:90000});
 await expect(picker).toHaveValue('m23');
 const panel=page.getByRole('region',{name:'M23 intelligent ball tracking'});
 const report=await(await page.request.get('/api/matches/giroud/cv/intelligent')).json();
 if(report.completed_review)await expect(panel.getByRole('table',{name:'Completed ball review comparison'})).toBeVisible({timeout:90000});
 else{await expect(panel.getByTestId('m23-unlabelled')).toBeVisible({timeout:90000});await expect(panel.getByRole('table')).toHaveCount(0)}
 const meta=await(await page.request.get('/api/matches/giroud/cv?ball_pipeline=m23')).json();
 expect(meta.frames).toBe(450);expect(meta.fps).toBe(25);expect(meta.recording_id).toBe('giroud');
 const video=page.locator('.cv-video-stage video');
 await video.evaluate((v:HTMLVideoElement)=>{v.muted=true;return v.play()});
 await expect.poll(()=>video.evaluate((v:HTMLVideoElement)=>v.currentTime),{timeout:30000}).toBeGreaterThan(.2);
 await video.evaluate((v:HTMLVideoElement)=>v.pause());
 await expect(panel.getByTestId('m23-state')).toContainText('Frame');
 await expect(panel.getByLabel('M21 comparison video')).toBeVisible();
 await expect(panel.getByLabel('M23 comparison video')).toBeVisible();
 const diagnostic=report.metrics.navigation[0];expect(diagnostic).toBeTruthy();
 await panel.getByRole('button',{name:`${diagnostic.kind.replaceAll('_',' ')} · frame ${diagnostic.frame}`,exact:true}).click();
 await expect(panel.getByTestId('m23-state')).toContainText(`Frame ${diagnostic.frame} ·`);
 await expect(page.getByTestId('m23-overlay')).toHaveAttribute('data-frame',String(diagnostic.frame));
 await page.getByRole('button',{name:'How many passes were detected?',exact:true}).click();
 await expect(page.getByTestId('assistant-answer')).toBeVisible({timeout:90000});
 expect(answer.cv_pipeline).toBe('m23');expect(answer.cv_model_revision).toContain(meta.m23_ball_revision);
 expect(answer.events.every((e:any)=>e.safe_for_automatic)).toBe(true);
 expect(errors).toEqual([]);
 await page.screenshot({path:'test-results/giroud-m23.jpg',type:'jpeg',quality:75,fullPage:true});
});
