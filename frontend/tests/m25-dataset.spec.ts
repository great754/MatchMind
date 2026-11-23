import {test,expect} from '@playwright/test';
test('real dataset includes long-video frames, keeps trusted labels and saves only explicit review actions',async({page})=>{
 let writes=0;await page.route('**/api/cv/m25/frames/*/shot',async route=>{await route.fulfill({json:{saved:true,shot_type:route.request().postDataJSON().shot_type}})});await page.route('**/api/cv/m25/frames/*/labels',async route=>{writes++;const body=route.request().postDataJSON();await route.fulfill({json:{saved:true,task_id:body.task_id}})});
 await page.goto('/?review=m25');await expect(page.getByRole('heading',{name:'M25 — Football dataset review'})).toBeVisible();await expect(page.getByText(/8 verified labels reused/)).toBeVisible();
 await expect(page.getByRole('alert')).toHaveCount(0);await expect(page.getByText(/12 Gemini frames · 12 Gemini boxes/)).toBeVisible();
 await expect.poll(()=>page.locator('svg[aria-label="Current source frame: verify the football"] image').evaluate((e:SVGImageElement)=>Boolean(e.href.baseVal))).toBe(true);
 expect(writes).toBe(0);await page.getByLabel('Dataset source').selectOption('tottenham_arsenal_long');await expect(page.getByText(/1280×720/)).toBeVisible();
 await page.getByLabel('Frame type').selectOption('replay');await expect(page.getByText(/Excluded from gameplay annotation/)).toBeVisible();await page.getByLabel('Frame type').selectOption('broadcast_gameplay');await expect(page.getByLabel('Gemini full-frame proposal')).toBeChecked();await page.getByLabel('Gemini full-frame proposal').uncheck();expect(writes).toBe(0);await page.getByLabel('Gemini full-frame proposal').check();await page.getByRole('button',{name:"Can't tell",exact:true}).click();await expect.poll(()=>writes).toBe(1);await expect(page.getByRole('status').filter({hasText:'Saved with revision history'})).toBeVisible();
 await page.getByRole('button',{name:'Previous frame',exact:true}).click();await expect(page.getByRole('heading',{name:/Frame/})).toBeVisible();
 await page.screenshot({path:'/tmp/tactiq-m25-browser/workspace.png',fullPage:true});
});
