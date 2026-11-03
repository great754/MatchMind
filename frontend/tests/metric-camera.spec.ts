import {test,expect} from '@playwright/test';
test('M8 metric trajectory, landmark recovery and safe unknown remain inspectable',async({page})=>{
 const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));await page.goto('/');
 for(const [id,frame] of [['test_match',250],['team_goal',250],['southampton',400]] as const){
  await page.getByLabel('Recording',{exact:true}).selectOption(id);
  const meta=await(await page.request.get(`/api/matches/${id}/cv`)).json();expect(meta.camera_model_version).toBe('m8');
  await page.getByLabel('Camera motion debug',{exact:true}).check();
  await page.locator(`[data-testid="camera-tick"][data-frame="${frame}"]`).click();await expect(page.getByTestId('cv-state')).toContainText(`Frame ${frame} ·`);
  await expect(page.getByTestId('metric-camera-debug')).toBeVisible();await expect(page.getByTestId('metric-camera-debug')).toContainText('not the physical 3D camera position');
  if(id!=='team_goal'){await expect(page.getByTestId('metric-landmark-decision')).toContainText('Validated');await expect(page.getByTestId('observed-metric-conic')).toHaveCount(1)}
  await page.screenshot({path:`test-results/${id}-m8.png`,fullPage:true});
 }
 await page.getByLabel('Recording',{exact:true}).selectOption('test_match');await page.locator('[data-testid="camera-tick"][data-frame="216"]').click();await expect(page.getByTestId('cv-state')).toContainText('Frame 216 ·');await expect(page.getByTestId('cv-current-player')).toHaveCount(0);await expect(page.getByTestId('metric-camera-debug')).toContainText('Current camera: Unknown');expect(errors).toEqual([]);
});
