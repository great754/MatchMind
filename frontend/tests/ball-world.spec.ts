import {test,expect} from '@playwright/test';
import {writeFile} from 'node:fs/promises';
test('M14 synchronized ball evidence separates observed, predicted and unknown pitch positions',async({page})=>{
 const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));await page.goto('/');const results=[];
 for(const id of ['test_match','team_goal','southampton']){
  await page.getByLabel('Recording',{exact:true}).selectOption(id);const meta=await(await page.request.get(`/api/matches/${id}/cv`)).json();expect(meta.camera_model_version).toBe('m8');expect(meta.player_model_version).toBe('m9');expect(meta.ball_model_version).toBe('m14');const inspector=page.getByRole('region',{name:'Ball evidence'});await expect(inspector).toBeVisible();
  for(const status of ['TRACKED','PREDICTED','LOST']){
   const example=meta.ball_examples.find((e:any)=>e.status===status);expect(example).toBeTruthy();await inspector.getByRole('button',{name:`Ball ${status} · frame ${example.frame}`,exact:true}).click();await expect(inspector.getByTestId('ball-state')).toContainText(`Frame ${example.frame} · ${status}`);await expect(page.getByTestId('ball-video-overlay')).toHaveAttribute('data-frame',String(example.frame));const row=await(await page.request.get(`/api/matches/${id}/cv/frame?timestamp=${(example.frame+.001)/meta.fps}`)).json();expect(row.ball_world.frame).toBe(example.frame);expect(row.player_world.version).toBe('m9');
   if(status==='TRACKED'){await expect(page.getByTestId('ball-image-observed')).toBeVisible();expect(row.ball_world.directly_observed).toBe(true)}
   if(status==='PREDICTED'){expect(row.ball_world.image_position).toBeNull();await expect(page.getByTestId('ball-image-predicted')).toBeVisible();await expect(inspector.getByTestId('ball-current-image')).toHaveText('UNKNOWN')}
   if(status==='LOST'){expect(row.ball_world.image_position).toBeNull();expect(row.ball_world.predicted_image_position).toBeNull();await expect(page.getByTestId('ball-image-observed')).toHaveCount(0)}
   if(row.ball_world.coordinate_state!=='GROUND_SUPPORTED'){expect(row.ball_world.pitch_position).toBeNull();await expect(page.getByTestId('ball-pitch-observed')).toHaveCount(0)}
  }
  await page.screenshot({path:`test-results/${id}-m14-ball.png`,fullPage:true});results.push({id,frames:meta.frames,ball:meta.ball_statistics,camera:meta.camera_model_version,player:meta.player_model_version});
 }
 expect(errors).toEqual([]);await writeFile('test-results/ball-world-results.json',JSON.stringify({results,page_errors:errors},null,2));
});
