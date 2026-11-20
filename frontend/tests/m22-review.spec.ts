import {test,expect} from '@playwright/test';
import {writeFile} from 'node:fs/promises';
function iou(a:number[],b:number[]){const v=Math.max(0,Math.min(a[0]+a[2],b[0]+b[2])-Math.max(a[0],b[0]))*Math.max(0,Math.min(a[1]+a[3],b[1]+b[3])-Math.max(a[1],b[1]));return v/(a[2]*a[3]+b[2]*b[3]-v)}

test('M22 automatically saves fully reviewed proposals, advances and restores corrections without writing test labels',async({page})=>{
 const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));const bodies:any[]=[];
 await page.addInitScript(()=>localStorage.setItem('tactiq.m22.reviewer','Automated UI check, not a human label'));
 await expect.poll(async()=>{try{return(await page.request.get('/api/ready',{timeout:4000})).status()}catch{return 0}},{timeout:300000,intervals:[1000,2000,3000]}).toBe(200);
 const original=await(await page.request.get('/api/cv/review/export')).json(),before=await(await page.request.get('/api/cv/m22/review/export')).json();
 await page.route('**/api/cv/m22/matches/*/review/labels',async route=>{
  const body=route.request().postDataJSON();bodies.push(body);expect(body.model_assisted).toBe(true);expect(body.reviewed_from_source).toBe(true);
  await new Promise(resolve=>setTimeout(resolve,350));await route.fulfill({status:200,json:{saved:true,task_id:body.task_id}});
 });
 const results=[];
 for(const name of ['team_goal','test_match','southampton']){
  await page.goto(`/?review=m22&recording=${name}`);await expect(page.getByRole('heading',{name:'M22 football training review'})).toBeVisible({timeout:90000});
  const data=await(await page.request.get(`/api/cv/m22/matches/${name}/review/tasks`)).json();expect(data.tasks).toHaveLength(50);expect(data.supports_reviewed_proposals).toBe(true);
  const current=data.tasks[0],next=data.tasks[1],region=page.getByRole('region',{name:'M22 ball annotation'}),start=bodies.length;
  await expect(region.getByRole('heading',{level:2})).toContainText(`Frame ${current.frame} ·`);await expect(page.getByLabel('Notes',{exact:true})).toBeHidden();await expect(page.getByRole('button',{name:/Save verified/})).toHaveCount(0);
  await expect(region.getByRole('button',{name:'Accept proposal 1'})).toBeEnabled();await expect(page.getByRole('img',{name:'Zoomed proposal 1',exact:true})).toBeVisible();expect(bodies.length).toBe(start);
  await page.screenshot({path:`test-results/m22-review-simple-${name}.jpg`,type:'jpeg',quality:75,fullPage:true});
  await region.getByRole('button',{name:'Accept proposal 1'}).click();
  for(let i=1;i<current.proposals.length;i++){
   if(iou(current.proposals[0].bbox,current.proposals[i].bbox)>.1){await region.getByRole('button',{name:`Accept proposal ${i+1}`,exact:true}).click();await region.getByRole('button',{name:'Accept proposal 1',exact:true}).click()}
   else await region.getByRole('button',{name:`Not the ball ${i+1}`,exact:true}).click();
  }
  await expect(region.getByRole('heading',{level:2})).toContainText(`Frame ${next.frame} ·`);expect(bodies.length).toBe(start+1);expect(new Set(bodies[start].reviewed_proposals)).toEqual(new Set(current.proposals.map((p:any)=>p.proposal_id)));
  await page.getByRole('button',{name:'Previous frame',exact:true}).click();await expect(region.getByRole('button',{name:'Accept proposal 1'})).toHaveAttribute('aria-pressed','true');expect(bodies.length).toBe(start+1);
  await region.getByRole('button',{name:"Can't tell",exact:true}).click();await expect(region.getByRole('heading',{level:2})).toContainText(`Frame ${next.frame} ·`);expect(bodies.length).toBe(start+2);expect(bodies[start+1].state).toBe('uncertain');
  await page.setViewportSize({width:390,height:844});expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBeTruthy();await page.screenshot({path:`test-results/m22-review-simple-${name}-mobile.jpg`,type:'jpeg',quality:70,fullPage:true});await page.setViewportSize({width:1440,height:1100});
  results.push({name,tasks:50,zoomed_proposals:true,automatic_save_and_next:true,back_restores_choices:true,automatic_correction:true,mobile_no_overflow:true});
 }
 expect(await(await page.request.get('/api/cv/review/export')).json()).toEqual(original);expect(await(await page.request.get('/api/cv/m22/review/export')).json()).toEqual(before);expect(errors).toEqual([]);
 await writeFile('test-results/m22-review-results.json',JSON.stringify({results,page_errors:errors,no_real_annotations_written:true,posts_mocked:bodies.length},null,2));
 await writeFile('test-results/m22-review-auto-payloads.json',JSON.stringify(bodies,null,2));
});

test('M22 failed automatic saves keep the current frame and require a deliberate retry',async({page})=>{
 await page.addInitScript(()=>localStorage.setItem('tactiq.m22.reviewer','Automated failed-save check'));
 let attempts=0;await page.route('**/api/cv/m22/matches/*/review/labels',async route=>{attempts++;const body=route.request().postDataJSON();await route.fulfill({status:attempts===1?503:200,json:attempts===1?{detail:'Temporary test save failure'}:{saved:true,task_id:body.task_id}})});
 await page.goto('/?review=m22&recording=team_goal');const data=await(await page.request.get('/api/cv/m22/matches/team_goal/review/tasks')).json();const region=page.getByRole('region',{name:'M22 ball annotation'});
 await region.getByRole('button',{name:"Can't tell",exact:true}).click();await expect(page.getByRole('alert')).toContainText('Temporary test save failure');await expect(region.getByRole('heading',{level:2})).toContainText(`Frame ${data.tasks[0].frame} ·`);
 await page.waitForTimeout(900);expect(attempts).toBe(1);await page.getByRole('button',{name:'Retry automatic save'}).click();await expect(region.getByRole('heading',{level:2})).toContainText(`Frame ${data.tasks[1].frame} ·`);expect(attempts).toBe(2);
});
