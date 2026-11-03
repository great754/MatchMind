import {describe,it,expect,vi} from 'vitest';
import {observeVideoFrames,VideoFrameBuffer,decodedFrameTime} from './videoFrames';
const flush=async()=>{for(let i=0;i<10;i++)await Promise.resolve()};
describe('frame-aligned tracking playback',()=>{
 it('uses decoded media timestamps and cancels the frame observer',()=>{
  let tick:VideoFrameRequestCallback=()=>{};const cancel=vi.fn();const video={requestVideoFrameCallback:vi.fn((cb:VideoFrameRequestCallback)=>{tick=cb;return 7}),cancelVideoFrameCallback:cancel} as unknown as HTMLVideoElement;const sample=vi.fn();const close=observeVideoFrames(video,sample);tick(0,{mediaTime:.04} as VideoFrameCallbackMetadata);expect(sample).toHaveBeenCalledWith(.04,true);expect(video.requestVideoFrameCallback).toHaveBeenCalledTimes(2);close();tick(0,{mediaTime:.08} as VideoFrameCallbackMetadata);expect(sample).toHaveBeenCalledTimes(1);expect(cancel).toHaveBeenCalledWith(7);
 });
 it('prefetches upcoming observations within the concurrency bound',async()=>{
  let active=0,max=0;const load=vi.fn(async(i:number)=>{max=Math.max(max,++active);await Promise.resolve();active--;return i});const b=new VideoFrameBuffer(load,100,12,vi.fn(),20,3);const show=vi.fn();b.watch(0,show);await flush();expect(max).toBeLessThanOrEqual(3);expect(b.peek(8)).toBe(8);b.watch(8,show);expect(show).toHaveBeenLastCalledWith(8);b.close();
 });
 it('never displays an obsolete response after seeking',async()=>{
  const resolve=new Map<number,(v:number)=>void>();const b=new VideoFrameBuffer<number>(i=>new Promise(r=>resolve.set(i,r)),100,0,vi.fn(),10,2);const show=vi.fn();b.watch(1,show);b.watch(80,show);resolve.get(1)!(1);await flush();expect(show).not.toHaveBeenCalled();resolve.get(80)!(80);await flush();expect(show).toHaveBeenLastCalledWith(80);b.close();
 });
 it('bounds cached rows and aborts work when the recording unmounts',async()=>{
  let signal:AbortSignal|undefined;const b=new VideoFrameBuffer(async(i,s)=>{signal=s;return i},100,15,vi.fn(),4,2);b.watch(0,vi.fn());await flush();expect(Array.from({length:100},(_,i)=>b.peek(i)).filter(x=>x!==undefined).length).toBeLessThanOrEqual(4);b.close();expect(signal?.aborted).toBe(true);expect(b.peek(0)).toBeUndefined();
 });
});

it('decoded rational-FPS timestamps do not select the preceding frame after quantization',()=>{const fps=24000/1001;for(const frame of [1,35,100,216,540]){const pts=Math.floor(frame/fps*1e6)/1e6;expect(Math.floor(decodedFrameTime(pts,fps)*fps+1e-8)).toBe(frame)}});
