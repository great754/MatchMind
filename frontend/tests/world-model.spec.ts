import {test,expect} from '@playwright/test';
test('local real-video world debug preserves unknown position and synchronized evidence',async({page})=>{
 const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto('/');
 await expect(page.getByRole('heading',{name:'TactiqAI — persistent player world model'})).toBeVisible();
 const video=page.locator('video');await expect.poll(()=>video.evaluate((v:HTMLVideoElement)=>v.readyState)).toBeGreaterThanOrEqual(2);
 expect(await page.evaluate('frames.length')).toBe(541);
 expect(await page.evaluate('frames.every(s=>s.players.every(e=>e.pitch===null)&&s.unknown.every(e=>e.pitch===null))')).toBe(true);
 expect(await page.evaluate("frames.some(s=>s.players.some(e=>e.visibility==='off_camera'))")).toBe(true);
 expect(await page.evaluate("frames.some(s=>s.unknown.some(e=>e.status==='ambiguous'))")).toBe(true);
 for(const time of [3,9,15,21]){
  await video.evaluate((v:HTMLVideoElement,t)=>{v.currentTime=t},time);
  await expect.poll(()=>video.evaluate((v:HTMLVideoElement)=>v.seeking)).toBe(false);
  await expect(page.locator('#status')).toContainText('calibration UNAVAILABLE');
  await expect.poll(async()=>Math.abs(Number((await page.locator('#status').textContent())!.split('s')[0])-time)).toBeLessThan(.05);
  await page.screenshot({path:`test-results/world-model-${time}s.png`,fullPage:true});
 }
 await video.evaluate((v:HTMLVideoElement)=>{v.currentTime=3;return v.play()});
 await expect.poll(()=>video.evaluate((v:HTMLVideoElement)=>v.currentTime)).toBeGreaterThan(3.5);
 await video.evaluate((v:HTMLVideoElement)=>v.pause());expect(errors).toEqual([]);
});
