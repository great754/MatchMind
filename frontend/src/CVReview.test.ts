import {describe,it,expect} from 'vitest';
import {precedingFrame,cameraFrameSeekTime} from './CVReview';
describe('CV frame sampling',()=>{
 it('uses the sample immediately preceding video time, with bounds',()=>{
  expect(precedingFrame(.079,25,100)).toBe(1);expect(precedingFrame(.08,25,100)).toBe(2);
  expect(precedingFrame(200,25,100)).toBe(99);expect(precedingFrame(0,25,100)).toBe(0);
  expect(precedingFrame(21,24000/1001,541)).toBe(503);
 });
});

it('diagnostic timeline tolerates media microsecond truncation at first and last frames',()=>{const fps=24000/1001;for(const frame of [0,35,216,540]){const t=cameraFrameSeekTime(frame/fps,fps,541);expect(precedingFrame(Math.floor(t*1e6)/1e6,fps,541)).toBe(frame);expect(t).toBeLessThan(541/fps)}});
