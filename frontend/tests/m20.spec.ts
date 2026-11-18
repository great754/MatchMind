import {test,expect} from '@playwright/test';
import {writeFile} from 'node:fs/promises';
test('M20 synchronizes real crops, motion uncertainty, source seeking, player focus and assistant evidence on three clips',async({page})=>{
 const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));let lastBody:any,lastTurn:any;
 await page.route('**/api/matches/*/assistant',async route=>{lastBody={...route.request().postDataJSON(),provider:'local'};const response=await route.fetch({postData:JSON.stringify(lastBody)});expect(response.status()).toBe(200);lastTurn=await response.json();await route.fulfill({response,json:lastTurn})});
 await expect.poll(async()=>{try{return(await page.request.get('/api/ready',{timeout:4000})).status()}catch{return 0}},{timeout:240000,intervals:[1000,2000,3000]}).toBe(200);
 const originalLabels=await(await page.request.get('/api/cv/review/export')).json();const results:any[]=[];
 await page.goto('/?recording=test_match');
 for(const [name,count,observed] of [['test_match',1,410],['team_goal',0,30],['southampton',0,147]] as const){
  if(name!=='test_match')await page.getByLabel('Recording',{exact:true}).selectOption(name);
  const picker=page.getByLabel('Ball review',{exact:true});await expect(picker.locator('option[value=m20]')).toBeEnabled({timeout:120000});await picker.selectOption('m20');
  const panel=page.getByRole('region',{name:'M20 trajectory tracking'});await expect(panel.getByTestId('m20-state')).toContainText('Frame',{timeout:90000});await expect(page.getByTestId('cv-pass-count')).toHaveText(String(count));await expect(panel.getByRole('table')).toContainText(`${observed}/`);
  const metadata=await(await page.request.get(`/api/matches/${name}/cv?ball_pipeline=m20`)).json();expect(metadata.ball_statistics.direct_observed_frames).toBe(observed);const source=page.locator('.cv-video-stage video');await expect.poll(()=>source.evaluate((v:HTMLVideoElement)=>v.readyState),{timeout:45000}).toBeGreaterThanOrEqual(2);
  const examples=[metadata.ball_examples.find((e:any)=>e.directly_observed),metadata.ball_examples.find((e:any)=>['PREDICTED','OCCLUDED'].includes(e.status)),metadata.ball_examples.find((e:any)=>e.status==='AMBIGUOUS'),metadata.ball_examples.find((e:any)=>e.status==='LOST')].filter(Boolean);const sampled=[];
  for(const example of examples){
   await page.locator(`[data-testid="camera-tick"][data-frame="${example.frame}"]`).click();await expect(panel.getByTestId('m20-state')).toContainText(`Frame ${example.frame} ·`);await expect(page.getByTestId('m20-overlay')).toHaveAttribute('data-frame',String(example.frame));
   const debug=await(await page.request.get(`/api/matches/${name}/cv/trajectory?timestamp=${(example.frame+.001)/metadata.fps}`)).json();expect(debug.independent_accuracy).toBe('pending_human_review');await expect(page.getByTestId('m20-overlay')).toHaveAttribute('data-state',debug.state);
   if(debug.state==='OBSERVED')await expect(page.getByTestId('ball-image-observed')).toHaveCount(1);else await expect(page.getByTestId('ball-image-observed')).toHaveCount(0);
   if(debug.state==='PREDICTED')expect(await page.getByTestId('m20-uncertainty').count()).toBeGreaterThan(0);
   sampled.push({frame:example.frame,state:debug.state,regions:debug.search_regions.length});
  }
  await panel.getByLabel('Show rejected and competing candidates').check();await panel.getByText('Motion hypotheses and visual checks').click();await page.screenshot({path:`test-results/m20-${name}.png`,fullPage:true});
  const current=Number((await page.getByTestId('m20-overlay').getAttribute('data-frame')));if(current<metadata.frames-1){await page.getByRole('button',{name:'+1 frame',exact:true}).click();await expect(panel.getByTestId('m20-state')).toContainText(`Frame ${current+1} ·`)}
  await source.evaluate((v:HTMLVideoElement)=>{v.muted=true;return v.play()});await expect.poll(()=>source.evaluate((v:HTMLVideoElement)=>v.paused)).toBe(false);await source.evaluate((v:HTMLVideoElement)=>v.pause());
  const events=await(await page.request.get(`/api/matches/${name}/cv/events?ball_pipeline=m20`)).json();expect(events.statistics.goal_count).toBeNull();expect(events.statistics.pass_accuracy).toBeNull();
  const ref=(events.events.find((e:any)=>e.status==='estimated')?.sender_ref)??events.players[0].player_ref;await page.getByLabel('CV player focus').selectOption(ref);await expect(page.locator('.player-panel').getByTestId('player-profile')).toBeVisible({timeout:90000});
  await page.locator('.player-panel').getByRole('button',{name:'Ask about this player'}).click();await page.getByRole('button',{name:'Ask TactiqAI',exact:true}).click();await expect(page.getByTestId('assistant-answer')).toBeVisible({timeout:90000});expect(lastBody.cv_pipeline).toBe('m20');expect(lastBody.selected_player).toBe(ref);expect(lastTurn.cv_pipeline).toBe('m20');expect(lastTurn.player_profile.stats.find((s:any)=>s.key==='goals').value).toBeNull();
  await picker.selectOption('m19');await expect(page.getByTestId('assistant-answer')).toHaveCount(0);await expect(page.getByTestId('m20-overlay')).toHaveCount(0);await expect(page.getByText('M19 ball evidence',{exact:true})).toBeVisible({timeout:90000});
  results.push({name,observed,qualified_estimates:count,sampled,frame_step:true,source_playback:true,player_focus:true,assistant_version_isolated:true});
 }
 const finalLabels=await(await page.request.get('/api/cv/review/export')).json();expect(finalLabels).toEqual(originalLabels);expect(errors).toEqual([]);await writeFile('test-results/m20-results.json',JSON.stringify({results,page_errors:errors,human_annotations_unchanged:true,human_labels_written:0},null,2));
});
