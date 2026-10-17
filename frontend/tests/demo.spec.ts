import {expect,test} from '@playwright/test';
test('actual backend: ask → evidence → jump, modes, pressure, event and player',async({page})=>{
 const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>{if(m.type()==='error')errors.push(m.text())});
 await page.goto('/');await expect(page.getByText('API connected')).toBeVisible();if(await page.getByRole('dialog').isVisible())await page.getByRole('button',{name:'Use neutral labels'}).click();await expect(page.getByText('API connected')).toBeVisible();await expect(page.getByRole('img',{name:'Tactical pitch at 0.00 seconds'})).toBeVisible();
 await expect.poll(()=>page.locator('video').evaluate((v:HTMLVideoElement)=>v.readyState)).toBeGreaterThanOrEqual(2);
 await page.getByRole('button',{name:'Play video',exact:true}).click();
 await expect.poll(()=>page.locator('video').evaluate((v:HTMLVideoElement)=>v.currentTime)).toBeGreaterThan(.2);
 await page.getByRole('button',{name:'Pause video',exact:true}).click();
 const played=await page.locator('video').evaluate((v:HTMLVideoElement)=>v.currentTime);
 await expect(page.getByRole('img',{name:`Tactical pitch at ${played.toFixed(2)} seconds`})).toBeVisible();
 await page.screenshot({path:'test-results/overview.png',fullPage:true});
 await page.getByRole('button',{name:'Who was the fastest player?',exact:true}).click();await expect(page.getByRole('button',{name:'Jump to moment'})).toBeVisible();
 await page.getByRole('button',{name:'Jump to moment'}).click();await expect(page.getByTestId('current-time')).toHaveText('02:51.28');await expect(page.getByRole('heading',{name:/Player \d+/})).toBeVisible();
 await expect.poll(()=>page.locator('video').evaluate((v:HTMLVideoElement)=>v.currentTime)).toBeCloseTo(171.28,1);
 await page.getByRole('button',{name:'Analyst',exact:true}).click();await expect(page.getByRole('button',{name:'Jump to moment'})).toBeVisible();await expect(page.locator('.fact').first()).toBeVisible();
 await page.getByRole('button',{name:'Beginner',exact:true}).click();await expect(page.getByRole('button',{name:'Jump to moment'})).toBeVisible();
 await page.getByRole('button',{name:'When was Home side under the most pressure?',exact:true}).click();await expect(page.getByRole('button',{name:'Jump to moment'})).toBeVisible();await page.getByRole('button',{name:'Jump to moment'}).click();await expect(page.getByTestId('current-time')).toHaveText('00:21.00');
 await page.locator('.event-row').first().click();await page.getByRole('button',{name:'Explain this event'}).click();await expect(page.getByText('Event explanation',{exact:true})).toBeVisible();await expect(page.getByText('What was observed',{exact:false})).toBeVisible();
 await page.getByLabel('Ask about this match').fill('What did the coach tell Team 1?');await page.getByRole('button',{name:'Ask TactiqAI',exact:true}).click();await expect(page.getByText('I don’t have enough evidence from this match to answer that reliably.')).toBeVisible();
 await page.screenshot({path:'test-results/match-intelligence.png',fullPage:true});expect(errors).toEqual([]);
});
test('missing video and unknown tactical state remain explicit',async({page})=>{
 await page.route('**/api/matches/118575_clip/view',async route=>{const response=await route.fetch();const data=await response.json();data.media_url=null;data.tactical_states=[];data.replay=null;data.status='unknown';await route.fulfill({json:data})});
 await page.goto('/');await expect(page.getByText('API connected')).toBeVisible();if(await page.getByRole('dialog').isVisible())await page.getByRole('button',{name:'Use neutral labels'}).click();await expect(page.getByText('Match video unavailable')).toBeVisible();await expect(page.getByText('Tactical positions unavailable for this recording.')).toBeVisible();
});
test('fallback is visible and remains grounded',async({page})=>{
 await page.route('**/api/matches/118575_clip/ask',async route=>{const response=await route.fetch();const data=await response.json();data.fallback=true;data.provider='local';await route.fulfill({json:data})});
 await page.goto('/');await expect(page.getByText('API connected')).toBeVisible();if(await page.getByRole('dialog').isVisible())await page.getByRole('button',{name:'Use neutral labels'}).click();await expect(page.getByText('API connected')).toBeVisible();await page.getByRole('button',{name:'Who was the fastest player?',exact:true}).click();await expect(page.getByText('Local fallback',{exact:true})).toBeVisible();await expect(page.getByRole('button',{name:'Jump to moment'})).toBeVisible();
});
test('laptop and mobile layouts have no horizontal overflow',async({page})=>{
 for(const width of [1024,390]){await page.setViewportSize({width,height:900});await page.goto('/');await expect(page.getByText('API connected')).toBeVisible();if(await page.getByRole('dialog').isVisible())await page.getByRole('button',{name:'Use neutral labels'}).click();await expect(page.getByText('API connected')).toBeVisible();expect(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth)).toBeTruthy();}
});
test('match setup persists identity and colors across review and Ask',async({page})=>{
 await page.goto('/');await expect(page.getByRole('dialog')).toBeVisible();
 const names=page.getByLabel('Team name',{exact:true});await names.nth(0).fill('Berry FC');await names.nth(1).fill('Oak FC');
 await page.getByLabel('Team color',{exact:true}).nth(0).fill('#11aa33');
 await page.getByText('Name your players',{exact:true}).click();
 await page.getByLabel('Name',{exact:true}).first().fill('Marcus');await page.getByLabel('Jersey',{exact:true}).first().fill('8');
 await page.getByRole('button',{name:'Save match setup'}).click();await expect(page.getByRole('dialog')).toBeHidden();
 await expect(page.getByRole('button',{name:'Select Marcus #8',exact:true})).toBeVisible();
 await expect(page.locator('circle[fill="#11aa33"]').first()).toBeVisible();
 await page.getByRole('button',{name:'Select Marcus #8',exact:true}).click();await expect(page.getByRole('heading',{name:'Marcus #8',exact:true})).toBeVisible();
 await page.getByLabel('Ask about this match').fill('How far did Marcus run?');await page.getByRole('button',{name:'Ask TactiqAI',exact:true}).click();await expect(page.locator('.answer-text')).toContainText('Marcus #8');
 await page.reload();await expect(page.getByText('API connected')).toBeVisible();await expect(page.getByRole('dialog')).toBeHidden();await expect(page.getByRole('button',{name:'Select Marcus #8',exact:true})).toBeVisible();
 const text=await page.locator('main').innerText();expect(text).not.toMatch(/Actor \d{4,}|confirmed_pass_share|bas_time_actor_match|start_x_m/);
 await page.screenshot({path:'test-results/identity-setup.png',fullPage:true});
});
test('typed pass paraphrases resolve from real canonical evidence',async({page})=>{
 await page.goto('/');await expect(page.getByText('API connected')).toBeVisible();if(await page.getByRole('dialog').isVisible())await page.getByRole('button',{name:'Use neutral labels'}).click();
 const cases=[['How many passes did both teams make?','Home side completed 16'],['How many completed passes did each side have?','Away side completed 13'],['Who passed more?','Home side completed more'],['Which team completed the most passes?','Home side completed more'],['How many passes were there altogether?','29 observable transfers altogether']];
 for(const [question,expected] of cases){
  await page.getByLabel('Ask about this match').fill(question);const response=page.waitForResponse(r=>r.url().endsWith('/ask')&&r.request().method()==='POST');await page.getByRole('button',{name:'Ask TactiqAI',exact:true}).click();const data=await (await response).json();expect(data.answer.status).toBe('ok');expect(data.answer.facts.find((f:{field:string})=>f.field==='total_confirmed_passes').value).toBe(29);await expect(page.locator('.answer-text')).toContainText(expected);
 }
});
test('unprocessed second recording is visible but cannot be selected',async({page})=>{
 await page.goto('/');await expect(page.getByText('API connected')).toBeVisible();
 await expect(page.getByRole('option',{name:/128058_clip.*awaiting processing/})).toHaveAttribute('disabled','');
 await expect(page.getByLabel('Recording',{exact:true})).toHaveValue('118575_clip');
});
