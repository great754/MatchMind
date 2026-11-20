import {test,expect} from '@playwright/test';
import {writeFile} from 'node:fs/promises';

test('M22 review uses the approved active-play clips, explicit saves and persistent source provenance',async({page})=>{
 const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));let posts=0;
 const original=await(await page.request.get('/api/cv/review/export')).json();
 const before=await(await page.request.get('/api/cv/m22/review/export')).json();
 await page.route('**/api/cv/m22/matches/*/review/labels',async route=>{posts++;const body=route.request().postDataJSON();expect(body.model_assisted).toBe(true);expect(body.reviewed_from_source).toBe(true);await route.fulfill({status:200,json:{saved:true,task_id:body.task_id}})});
 const results=[];
 for(const name of ['team_goal','test_match','southampton']){
  await page.goto(`/?review=m22&recording=${name}`);await expect(page.getByRole('heading',{name:'M22 football training review'})).toBeVisible({timeout:90000});
  const data=await(await page.request.get(`/api/cv/m22/matches/${name}/review/tasks`)).json();expect(data.tasks).toHaveLength(50);expect(data.reference_role).toBe('development_only');expect(data.model_proposals_are_truth).toBe(false);
  const region=page.getByRole('region',{name:'M22 ball annotation'});await expect(region).toBeVisible();expect(await page.getByRole('img',{name:'Current source frame: verify the football'}).locator('image').getAttribute('href')).toContain(`/matches/${name}/review/frame.jpg?frame=`);
  await page.getByRole('button',{name:'Next selected frame'}).click();await expect(region.getByRole('heading')).toContainText(`Frame ${data.tasks[1].frame}`);
  await page.getByRole('button',{name:'Previous selected frame'}).click();
  await page.getByText('See the selected source footage and selection reasons',{exact:true}).click();
  const preview=page.getByRole('img',{name:`${name} selected source frames, page 1`});await expect(preview).toBeVisible();await expect.poll(()=>preview.evaluate((i:HTMLImageElement)=>i.naturalWidth)).toBeGreaterThan(0);
  await page.getByText('See the selected source footage and selection reasons',{exact:true}).click();
  await page.getByLabel('Reviewer ID',{exact:true}).fill('UI test, not a human label');await page.getByLabel('Notes',{exact:true}).fill('Unsaved browser check');
  expect(posts).toBe(results.length);await expect(page.getByRole('button',{name:'Save verified development label'})).toBeDisabled();
  await page.getByLabel('Ball visibility',{exact:true}).selectOption('not_visible');await page.getByLabel('I checked the actual source frame and corrected any proposal errors.',{exact:true}).check();
  await page.getByRole('button',{name:'Save verified development label'}).click();await expect(page.getByRole('status')).toContainText('Saved human-verified development label.');
  await page.screenshot({path:`test-results/m22-review-${name}.jpg`,type:'jpeg',quality:70,fullPage:true});results.push({name,tasks:50,source_preview_loaded:true,explicit_save_mock_only:true});
 }
 await page.setViewportSize({width:390,height:844});expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBeTruthy();
 expect(await(await page.request.get('/api/cv/review/export')).json()).toEqual(original);expect(await(await page.request.get('/api/cv/m22/review/export')).json()).toEqual(before);expect(errors).toEqual([]);
 await writeFile('test-results/m22-review-results.json',JSON.stringify({results,page_errors:errors,no_real_annotations_written:true,mobile_no_overflow:true},null,2));
});
