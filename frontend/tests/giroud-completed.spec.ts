import {test,expect} from '@playwright/test';
import {writeFile} from 'node:fs/promises';

test('actual completed Giroud reviews show validated frozen metrics without writing labels',async({page})=>{
 const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));let writes=0;
 await page.route('**/api/cv/m23/matches/giroud/review/labels',async route=>{writes++;await route.fulfill({status:503,json:{detail:'Read-only browser verification blocks annotation writes'}})});
 const before=await(await page.request.get('/api/cv/m23/review/export')).json();expect(before.annotations).toHaveLength(40);
 await page.goto('/?review=m23&recording=giroud');
 await expect(page.getByTestId('m23-review-progress')).toContainText('40/40');
 await expect(page.getByRole('button',{name:'Next frame',exact:true})).toBeDisabled();
 await page.getByRole('link',{name:'Back to analysis'}).click();
 const panel=page.getByRole('region',{name:'M23 intelligent ball tracking'});
 const table=panel.getByRole('table',{name:'Completed ball review comparison'});
 await expect(table).toBeVisible({timeout:90000});
 const row=table.getByRole('row').filter({has:page.getByRole('rowheader',{name:'M23',exact:true})});
 await expect(row).toContainText('1 / 6 / 30');await expect(row).toContainText('14.3%');await expect(row).toContainText('3.2%');await expect(row).toContainText('0.053');
 await expect(panel.getByText(/40 answers saved · 31 visible balls · 9 uncertain frames excluded · 49 rejected/)).toBeVisible();
 await expect(panel.getByTestId('m23-unlabelled')).toHaveCount(0);
 const metadata=await(await page.request.get('/api/matches/giroud/cv/intelligent')).json();
 expect(metadata.reference_role).toBe('reviewed_development');expect(metadata.completed_review.independent_final_test).toBe(false);
 expect(metadata.completed_review.labels_updated_since_snapshot).toBe(false);
 const video=page.locator('.cv-video-stage video');await video.evaluate((v:HTMLVideoElement)=>{v.muted=true;return v.play()});
 await expect.poll(()=>video.evaluate((v:HTMLVideoElement)=>v.currentTime)).toBeGreaterThan(.2);await video.evaluate((v:HTMLVideoElement)=>v.pause());
 await panel.screenshot({path:'test-results/giroud-completed-results.jpg',type:'jpeg',quality:80});
 expect(writes).toBe(0);expect(await(await page.request.get('/api/cv/m23/review/export')).json()).toEqual(before);
 expect(errors).toEqual([]);
 await writeFile('test-results/giroud-completed-results.json',JSON.stringify({human_answers:40,completed_review_resumes:true,
  measured_table:true,source_playback:true,annotation_writes:0,labels_unchanged:true,independent_test_claimed:false,page_errors:errors},null,2));
});
