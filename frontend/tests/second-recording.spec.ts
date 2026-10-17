import {expect,test} from '@playwright/test';
test('actual second recording selection playback and grounded Ask',async({page})=>{
 test.skip(!process.env.TACTIQAI_BROWSER_REGISTRY,'Opt in with the two-recording registry after processing.');
 const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>{if(m.type()==='error')errors.push(m.text())});
 await page.goto('/');await expect(page.getByText('API connected')).toBeVisible();if(await page.getByRole('dialog').isVisible())await page.getByRole('button',{name:'Use neutral labels'}).click();
 const viewResponse=page.waitForResponse(r=>r.url().endsWith('/128058_clip/view'));
 await page.getByLabel('Recording',{exact:true}).selectOption('128058_clip');const view=await (await viewResponse).json();expect(view.recording_id).toBe('128058_clip');expect(view.replay.frames).toBe(5981);
 await expect(page.getByText('API connected')).toBeVisible();if(await page.getByRole('dialog').isVisible())await page.getByRole('button',{name:'Use neutral labels'}).click();
 await expect.poll(()=>page.locator('video').evaluate((v:HTMLVideoElement)=>v.readyState)).toBeGreaterThanOrEqual(2);
 await expect(page.locator('video')).toHaveAttribute('src','/api/matches/128058_clip/video.mp4');
 const mediaDuration=await page.locator('video').evaluate((v:HTMLVideoElement)=>v.duration);expect(mediaDuration).toBeGreaterThanOrEqual(239.24);expect(mediaDuration).toBeLessThan(240.25); // Container audio can outlast the decoded video frames.
 await page.getByRole('button',{name:'Play video',exact:true}).click();await expect.poll(()=>page.locator('video').evaluate((v:HTMLVideoElement)=>v.currentTime)).toBeGreaterThan(.2);await page.getByRole('button',{name:'Pause video',exact:true}).click();
 await page.getByLabel('Ask about this match').fill('How many passes were there altogether?');const answerResponse=page.waitForResponse(r=>r.url().endsWith('/128058_clip/ask'));await page.getByRole('button',{name:'Ask TactiqAI',exact:true}).click();const answer=await (await answerResponse).json();expect(answer.answer.status).toBe('ok');expect(answer.conversation.recording_id).toBe('128058_clip');await expect(page.locator('.answer-text')).toContainText('altogether');
 await page.screenshot({path:'test-results/128058-processed.png',fullPage:true});
 await page.getByLabel('Recording',{exact:true}).selectOption('118575_clip');await expect(page.getByText('API connected')).toBeVisible();await expect(page.getByTestId('current-time')).toHaveText('00:00.00');expect(errors).toEqual([]);
});
