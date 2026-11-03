import {test,expect} from '@playwright/test';
import {writeFile} from 'node:fs/promises';
test('M15 motion search and ball provenance remain synchronized across three source videos',async({page})=>{
 const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));await page.goto('/');const results=[];
 for(const id of ['test_match','team_goal','southampton']){
  await page.getByLabel('Recording',{exact:true}).selectOption(id);await expect(page.getByLabel('Ball review',{exact:true}).locator('option[value="m15"]')).toBeEnabled();await page.getByLabel('Ball review',{exact:true}).selectOption('m15');const meta=await(await page.request.get(`/api/matches/${id}/cv?ball_pipeline=m15`)).json();expect(meta.camera_model_version).toBe('m8');expect(meta.player_model_version).toBe('m9');expect(meta.ball_model_version).toBe('m15');const inspector=page.getByRole('region',{name:'Ball evidence'});await expect(inspector).toBeVisible();await expect(inspector).toContainText('Ball evidence · M15');
  for(const status of ['TRACKED','PREDICTED','LOST']){
   const example=meta.ball_examples.find((e:any)=>e.status===status);if(!example){expect(meta.ball_statistics[status]??0).toBe(0);continue}
   await inspector.getByRole('button',{name:`Ball ${status} · frame ${example.frame}`,exact:true}).click();await expect(inspector.getByTestId('ball-state')).toContainText(`Frame ${example.frame} · ${status}`);await expect(page.getByTestId('ball-video-overlay')).toHaveAttribute('data-frame',String(example.frame));const row=await(await page.request.get(`/api/matches/${id}/cv/frame?timestamp=${(example.frame+.001)/meta.fps}&ball_pipeline=m15`)).json();expect(row.ball_world.frame).toBe(example.frame);expect(row.player_world.version).toBe('m9');await expect(inspector.getByTestId('ball-motion-search')).toContainText(row.ball_world.search_stage);
   if(status==='TRACKED'){expect(row.ball_world.directly_observed).toBe(true);await expect(page.getByTestId('ball-image-observed')).toBeVisible()}
   if(status==='PREDICTED'){expect(row.ball_world.image_position).toBeNull();await expect(page.getByTestId('ball-image-predicted')).toBeVisible();await expect(inspector.getByTestId('ball-current-image')).toHaveText('UNKNOWN')}
   if(status==='LOST'){expect(row.ball_world.predicted_image_position).toBeNull();expect(row.ball_world.image_position).toBeNull();await expect(page.getByTestId('ball-image-observed')).toHaveCount(0)}
   if(row.ball_world.coordinate_state!=='GROUND_SUPPORTED'){expect(row.ball_world.pitch_position).toBeNull();await expect(page.getByTestId('ball-pitch-observed')).toHaveCount(0);await expect(inspector).toContainText('Pitch position unavailable')}
  }
  await page.screenshot({path:`test-results/${id}-m15-ball.png`,fullPage:true});results.push({id,statistics:meta.ball_statistics});
 }
 expect(errors).toEqual([]);await writeFile('test-results/ball-motion-results.json',JSON.stringify({results,page_errors:errors},null,2));
});
