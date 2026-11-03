import {test,expect} from '@playwright/test';
import {writeFile} from 'node:fs/promises';
test('CV overlays follow actual video frames rather than low-frequency timeupdate',async({page})=>{
 const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));await page.goto('/');const results=[];
 for(const id of ['test_match','team_goal','southampton']){
  await page.getByLabel('Recording',{exact:true}).selectOption(id);const meta=await(await page.request.get(`/api/matches/${id}/cv`)).json();const video=page.locator('video');await expect.poll(()=>video.evaluate((v:HTMLVideoElement)=>v.readyState)).toBeGreaterThanOrEqual(2);
  await video.evaluate((v:HTMLVideoElement)=>{v.currentTime=3;v.muted=true});await expect(page.getByTestId('cv-state')).not.toContainText('Frame Unknown');await video.evaluate((v:HTMLVideoElement)=>void v.play());await expect.poll(()=>video.evaluate((v:HTMLVideoElement)=>v.currentTime)).toBeGreaterThan(4);
  const sample=await page.evaluate(async(fps)=>{
   const video=document.querySelector('video')!;const changes:{frame:number,time:number}[]=[];const lags:number[]=[];const start=performance.now();let missing=0,ticks=0;
   await new Promise<void>(resolve=>{const tick=()=>{const overlay=document.querySelector('[aria-label="CV source evidence"]');const raw=overlay?.getAttribute('data-frame');ticks++;if(raw==null)missing++;else{const frame=Number(raw);if(changes.at(-1)?.frame!==frame)changes.push({frame,time:performance.now()-start});lags.push(Math.floor(video.currentTime*fps+1e-8)-frame)}if(performance.now()-start<2500)requestAnimationFrame(tick);else resolve()};requestAnimationFrame(tick)});
   video.pause();const elapsed=(performance.now()-start)/1000;return {elapsed,changes:changes.length,updates_per_second:(changes.length-1)/elapsed,missing_fraction:missing/ticks,max_lag:Math.max(...lags),max_gap_ms:Math.max(...changes.slice(1).map((p,i)=>p.time-changes[i].time))};
  },meta.fps);
  results.push({id,source_fps:meta.fps,...sample});expect(sample.updates_per_second).toBeGreaterThan(meta.fps*.65);expect(sample.missing_fraction).toBeLessThan(.1);expect(sample.max_gap_ms).toBeLessThan(250);expect(sample.max_lag).toBeLessThanOrEqual(3);
  // Exact diagnostic seeks still use the source frame; no interpolated positions.
  await page.locator('[data-testid="camera-tick"][data-frame="216"]').click();await expect(page.getByTestId('cv-state')).toContainText('Frame 216 ·');
 }
 expect(errors).toEqual([]);await writeFile('test-results/cv-smooth-results.json',JSON.stringify({results,page_errors:errors},null,2));
});
