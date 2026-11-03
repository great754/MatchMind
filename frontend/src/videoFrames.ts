/** Observe decoded video frames; rAF is the fallback for older browsers. */
export function observeVideoFrames(video:HTMLVideoElement, sample:(time:number,decoded:boolean)=>void){
 let stopped=false,handle=0;
 if(typeof video.requestVideoFrameCallback==='function'){
  const tick:VideoFrameRequestCallback=(_now,frame)=>{if(stopped)return;sample(frame.mediaTime,true);handle=video.requestVideoFrameCallback(tick)};
  handle=video.requestVideoFrameCallback(tick);
  return()=>{stopped=true;video.cancelVideoFrameCallback(handle)};
 }
 const tick=()=>{if(stopped)return;sample(video.currentTime,false);handle=requestAnimationFrame(tick)};
 handle=requestAnimationFrame(tick);return()=>{stopped=true;cancelAnimationFrame(handle)};
}

/** Bounded look-ahead, shared in-flight requests and exact-frame delivery.
 * Late responses may populate the cache but cannot replace the wanted frame.
 */
export class VideoFrameBuffer<T>{
 private cache=new Map<number,T>();private pending:number[]=[];private inFlight=new Set<number>();private controller=new AbortController();private wanted=0;private deliver:((row:T)=>void)|null=null;
 constructor(private load:(frame:number,signal:AbortSignal)=>Promise<T>,private total:number,private ahead:number,private failed:(error:unknown)=>void,private capacity=120,private concurrency=4){}
 peek(frame:number){return this.cache.get(frame)}
 watch(frame:number,deliver:(row:T)=>void){
  if(this.controller.signal.aborted)return;
  this.wanted=frame;this.deliver=deliver;const cached=this.cache.get(frame);if(cached!==undefined)deliver(cached);
  this.pending=[];
  for(let i=frame;i<Math.min(this.total,frame+this.ahead+1);i++)if(!this.cache.has(i)&&!this.inFlight.has(i))this.pending.push(i);
  this.pump();
 }
 close(){this.controller.abort();this.pending=[];this.deliver=null;this.cache.clear()}
 private pump(){
  while(!this.controller.signal.aborted&&this.inFlight.size<this.concurrency&&this.pending.length){
   const frame=this.pending.shift()!;this.inFlight.add(frame);
   this.load(frame,this.controller.signal).then(row=>{
    if(this.controller.signal.aborted)return;this.cache.set(frame,row);
    if(frame===this.wanted)this.deliver?.(row);
    while(this.cache.size>this.capacity){const furthest=[...this.cache.keys()].sort((a,b)=>Math.abs(b-this.wanted)-Math.abs(a-this.wanted))[0];this.cache.delete(furthest)}
   }).catch(error=>{if(!this.controller.signal.aborted&&frame===this.wanted)this.failed(error)}).finally(()=>{this.inFlight.delete(frame);this.pump()});
  }
 }
}

// Decoded PTS is microsecond-quantized: snap its frame boundary, never a seek time.
export const decodedFrameTime=(time:number,fps:number)=>Math.round(time*fps)/fps;
