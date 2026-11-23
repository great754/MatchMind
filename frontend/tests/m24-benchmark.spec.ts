import {test,expect} from '@playwright/test';
import {writeFile,mkdir} from 'node:fs/promises';

test('actual M24 source frames, heatmaps, case navigation and video remain read-only',async({page})=>{
 const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));let writes=0;
 page.on('request',r=>{if(r.method()!=='GET'&&r.method()!=='HEAD')writes++});
 const before=await(await page.request.get('/api/cv/m23/review/export')).json();
 await page.goto('/?benchmark=m24&recording=giroud');await expect(page.getByRole('heading',{name:'M24 — FootAndBall Detector Benchmark'})).toBeVisible();
 await expect(page.getByText('Uncertain human reference · excluded from scoring')).toBeVisible();
 await page.getByRole('button',{name:'Next case'}).click();await expect(page.getByLabel('Benchmark frame')).toHaveValue('60');await expect(page.getByText('Missed ball',{exact:true})).toBeVisible();
 await page.getByLabel('Heatmap',{exact:true}).check();await expect(page.getByAltText('Ball confidence heatmap')).toBeVisible();
 await expect.poll(()=>page.getByAltText('Ball confidence heatmap').evaluate((v:HTMLImageElement)=>v.complete&&v.naturalWidth===640)).toBe(true);
 await page.getByLabel('Find benchmark cases').selectOption('m23_wins');await page.getByRole('button',{name:'Next case'}).click();
 await expect(page.getByLabel('Benchmark frame')).not.toHaveValue('60');
 await page.getByText('Play the original video',{exact:true}).click();const video=page.locator('video');
 await video.evaluate((v:HTMLVideoElement)=>{v.muted=true;return v.play()});await expect.poll(()=>video.evaluate((v:HTMLVideoElement)=>v.currentTime)).toBeGreaterThan(.2);await video.evaluate((v:HTMLVideoElement)=>v.pause());
 const recordings=[];
 for(const recording of ['test_match','team_goal','southampton']){
  await page.getByLabel('Benchmark recording').selectOption(recording);await expect(page.getByAltText(`${recording} source frame 0`,{exact:true})).toBeVisible();
  await page.getByLabel('Find benchmark cases').selectOption('reviewed');await expect(page.getByRole('button',{name:'Next case'})).toBeEnabled();await page.getByRole('button',{name:'Next case'}).click();
  await expect.poll(()=>page.getByAltText(new RegExp(`^${recording} source frame [0-9]+$`)).evaluate((v:HTMLImageElement)=>v.complete&&v.naturalWidth>0)).toBe(true);
  recordings.push(recording);
 }
 await page.getByLabel('Benchmark recording').selectOption('test_match');await page.getByLabel('Reference set').selectOption('completed_m22');await page.getByLabel('Find benchmark cases').selectOption('correct');
 await page.getByRole('button',{name:'Next case'}).click();await expect(page.getByText('Correct ball detection',{exact:true})).toBeVisible();
 await mkdir('/tmp/tactiq-m24-browser-artifacts',{recursive:true});await page.screenshot({path:'/tmp/tactiq-m24-browser-artifacts/benchmark.png',fullPage:true});
 expect(await(await page.request.get('/api/cv/m23/review/export')).json()).toEqual(before);expect(writes).toBe(0);expect(errors).toEqual([]);
 await writeFile('/tmp/tactiq-m24-browser-artifacts/result.json',JSON.stringify({giroud:true,recordings,source_images:true,heatmap:true,case_navigation:true,original_video_playback:true,annotation_writes:writes,labels_unchanged:true,page_errors:errors},null,2));
});
