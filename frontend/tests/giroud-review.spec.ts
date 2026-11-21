import {test,expect} from '@playwright/test';
import {writeFile} from 'node:fs/promises';
const overlap=(a:number[],b:number[])=>{const x=Math.max(0,Math.min(a[0]+a[2],b[0]+b[2])-Math.max(a[0],b[0]))*Math.max(0,Math.min(a[1]+a[3],b[1]+b[3])-Math.max(a[1],b[1]));return x/(a[2]*a[3]+b[2]*b[3]-x)};

test('Giroud review auto-saves, advances, corrects, resumes and completes with test writes mocked',async({page})=>{
 const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));
 const endpoint='/api/cv/m23/matches/giroud/review';
 const before=await(await page.request.get('/api/cv/m23/review/export')).json();
 const original=await(await page.request.get('/api/cv/review/export')).json();
 const m22=await(await page.request.get('/api/cv/m22/review/export')).json();
 const live=await(await page.request.get(endpoint+'/tasks')).json();
 expect(live.tasks).toHaveLength(40);expect(live.labels).toHaveLength(0);
 expect(live.tasks.every((t:any)=>t.proposals.length<=2)).toBe(true);
 let labels:any[]=[],posts=0,failNext=false;
 await page.addInitScript(()=>localStorage.setItem('tactiq.m23.reviewer','Automated UI check; not human reference'));
 await page.route('**/api/cv/m23/matches/giroud/review/tasks',async route=>{
  await route.fulfill({status:200,json:{...live,labels,sources:live.sources.map((s:any)=>({...s,saved_tasks:labels.length}))}});
 });
 await page.route('**/api/cv/m23/matches/giroud/review/labels',async route=>{
  posts++;const body=route.request().postDataJSON();
  const task=live.tasks.find((t:any)=>t.task_id===body.task_id);expect(task).toBeTruthy();
  expect(new Set(body.reviewed_proposals)).toEqual(new Set(task.proposals.map((p:any)=>p.proposal_id)));
  expect(body.model_assisted).toBe(true);expect(body.reviewed_from_source).toBe(true);
  if(failNext){failNext=false;await route.fulfill({status:503,json:{detail:'Test save failure; server labels unchanged'}});return}
  labels=[...labels.filter(l=>l.task_id!==body.task_id),body];
  await route.fulfill({status:200,json:{saved:true,task_id:body.task_id}});
 });
 await page.goto('/?review=m23&recording=giroud');
 await expect(page.getByRole('heading',{name:'Giroud ball review'})).toBeVisible();
 const region=page.getByRole('region',{name:'M23 ball annotation'});
 await expect(page.getByTestId('m23-review-progress')).toContainText('0/40');
 await expect(region.getByRole('button',{name:'Accept proposal 1'})).toBeEnabled();
 expect(posts).toBe(0);await expect(page.getByRole('button',{name:/Save human|Save verified/})).toHaveCount(0);
 await page.screenshot({path:'test-results/giroud-review-desktop.jpg',type:'jpeg',quality:75,fullPage:true});
 await region.getByText('More context & optional notes',{exact:true}).click();
 const video=page.getByLabel('Source video context');await video.evaluate((v:HTMLVideoElement)=>{v.muted=true;return v.play()});
 await expect.poll(()=>video.evaluate((v:HTMLVideoElement)=>v.currentTime)).toBeGreaterThan(.15);
 await video.evaluate((v:HTMLVideoElement)=>v.pause());
 await region.getByRole('button',{name:'Accept proposal 1',exact:true}).click();
 for(let i=1;i<live.tasks[0].proposals.length;i++){
  await region.getByRole('button',{name:overlap(live.tasks[0].proposals[0].bbox,live.tasks[0].proposals[i].bbox)>.1?`Accept proposal ${i+1}`:`Not the ball ${i+1}`,exact:true}).click();
 }
 await expect(region.getByRole('heading',{level:2})).toContainText(`Frame ${live.tasks[1].frame} ·`);expect(posts).toBe(1);
 await region.getByRole('button',{name:'Previous frame',exact:true}).click();
 await expect(region.getByRole('button',{name:'Accept proposal 1'})).toHaveAttribute('aria-pressed','true');
 await region.getByRole('button',{name:"Can't tell",exact:true}).click();
 await expect(region.getByRole('heading',{level:2})).toContainText(`Frame ${live.tasks[1].frame} ·`);expect(posts).toBe(2);
 failNext=true;await region.getByRole('button',{name:"Can't tell",exact:true}).click();
 await expect(page.getByRole('alert')).toContainText('Test save failure');const attempts=posts;
 await page.waitForTimeout(900);expect(posts).toBe(attempts);
 await page.getByRole('button',{name:'Retry automatic save'}).click();
 await expect(region.getByRole('heading',{level:2})).toContainText(`Frame ${live.tasks[2].frame} ·`);
 await page.reload();await expect(region.getByRole('heading',{level:2})).toContainText(`Frame ${live.tasks[2].frame} ·`);
 await expect(page.getByTestId('m23-review-progress')).toContainText('2/40');
 await page.setViewportSize({width:390,height:844});expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
 await page.screenshot({path:'test-results/giroud-review-mobile.jpg',type:'jpeg',quality:75,fullPage:true});
 await page.setViewportSize({width:1440,height:1100});
 for(let i=2;i<40;i++){
  await expect(region.getByRole('heading',{level:2})).toContainText(`Frame ${live.tasks[i].frame} ·`);
  await expect(region.getByRole('button',{name:"Can't tell",exact:true})).toBeEnabled();
  await region.getByRole('button',{name:"Can't tell",exact:true}).click();
  if(i<39)await expect(region.getByRole('heading',{level:2})).toContainText(`Frame ${live.tasks[i+1].frame} ·`);
 }
 await expect(page.getByText('Saved. This clip is complete. You can go back to make corrections.')).toBeVisible();
 await expect(page.getByTestId('m23-review-progress')).toContainText('40/40');
 await expect(region.getByRole('button',{name:'Next frame',exact:true})).toBeDisabled();
 expect(await(await page.request.get('/api/cv/m23/review/export')).json()).toEqual(before);
 expect(await(await page.request.get('/api/cv/review/export')).json()).toEqual(original);
 expect(await(await page.request.get('/api/cv/m22/review/export')).json()).toEqual(m22);
 expect(errors).toEqual([]);
 await writeFile('test-results/giroud-review-results.json',JSON.stringify({tasks:40,actual_source_video_playback:true,
  automatic_save_next:true,back_correction:true,failed_save_retry:true,resume:true,last_question_completion:true,
  mobile_no_overflow:true,posts_mocked:posts,no_real_annotations_written:true,page_errors:errors},null,2));
});

test('Giroud analysis opens its own ball review without saving annotations',async({page})=>{
 const before=await(await page.request.get('/api/cv/m23/review/export')).json();let posts=0;
 await page.route('**/api/cv/m23/matches/giroud/review/labels',async route=>{posts++;await route.fulfill({status:503,json:{detail:'Unexpected test write blocked'}})});
 await page.goto('/?recording=giroud&ball_pipeline=m23');
 const link=page.getByRole('link',{name:'Review the ball',exact:true});
 await expect(link).toHaveAttribute('href','/?review=m23&recording=giroud');await link.click();
 await expect(page.getByRole('heading',{name:'Giroud ball review'})).toBeVisible();
 await expect(page.getByRole('button',{name:'Accept proposal 1'})).toBeEnabled();
 expect(posts).toBe(0);expect(await(await page.request.get('/api/cv/m23/review/export')).json()).toEqual(before);
});
