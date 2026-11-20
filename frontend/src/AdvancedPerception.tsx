import {useEffect,useRef,useState} from 'react';
import type {BallState} from './BallDebug';
import type {AdvancedFrame,AdvancedMetadata} from './advancedContracts';
import {experimentLabel} from './advancedContracts';
import {observeVideoFrames,decodedFrameTime} from './videoFrames';

export const overlayMatchesFrame=(expected:number,frame:number|undefined)=>frame===expected;

export function Comparison({video,width,height,fps,frame,m20,ball,label}:{video:HTMLVideoElement|null,width:number,height:number,fps:number,frame?:Pick<AdvancedFrame,'frame'|'variant'>|null,m20?:{frame:number,ball:BallState}|null,ball?:BallState|null,label?:string}){
 const left=useRef<HTMLCanvasElement>(null),right=useRef<HTMLCanvasElement>(null);
 const redraw=useRef<((time:number,decoded?:boolean)=>void)|null>(null);
 const latest=useRef({frame,m20,ball});latest.current={frame,m20,ball};
 useEffect(()=>{
  if(!video)return;
  const draw=(time:number,decoded=false)=>{
   const sourceTime=video.paused?video.currentTime:decoded?decodedFrameTime(time,fps):time;
   const index=Math.floor(sourceTime*fps+1e-8);
   const data=latest.current;
   for(const [canvas,state,known] of [[left.current,data.m20?.ball,data.m20?.frame],[right.current,data.ball,data.frame?.frame]] as const){
    if(!canvas||video.readyState<2)continue;
    const context=canvas.getContext('2d');if(!context)continue;
    context.clearRect(0,0,width,height);context.drawImage(video,0,0,width,height);
    const matched=overlayMatchesFrame(index,known);
    if(matched&&state){
     const point=state.directly_observed?state.image_position:state.predicted_image_position;
     if(point){context.beginPath();context.strokeStyle=state.directly_observed?'#b5ff73':'#6be3ef';context.lineWidth=Math.max(2,width/350);context.setLineDash(state.directly_observed?[]:[8,5]);context.arc(point[0],point[1],Math.max(8,width/90),0,Math.PI*2);context.stroke();context.setLineDash([])}
    }
    context.fillStyle='rgba(10,18,23,.85)';context.fillRect(0,0,Math.min(width,480),Math.max(30,width/35));context.fillStyle='white';context.font=`${Math.max(13,width/70)}px sans-serif`;context.fillText(`Frame ${index} · ${matched?state?.directly_observed?'Observed':state?.predicted_image_position?'Inferred':'Unknown':'overlay loading'}`,10,Math.max(20,width/50));
   }
  };
  redraw.current=draw;
  const stop=observeVideoFrames(video,draw);const seek=()=>draw(video.currentTime);video.addEventListener('seeked',seek);draw(video.currentTime);
  return()=>{stop();video.removeEventListener('seeked',seek);if(redraw.current===draw)redraw.current=null};
 },[video,width,height,fps]);
 useEffect(()=>{if(video)redraw.current?.(video.currentTime)},[frame,m20,ball,video]);
 return <div className="advanced-comparison" aria-label={label?'Synchronized M20 and M22 video comparison':'Synchronized M20 and M21 video comparison'}><figure><figcaption>M20 control</figcaption><canvas ref={left} width={width} height={height} aria-label="M20 comparison video"/></figure><figure><figcaption>{label??`M21 · ${frame?experimentLabel(frame.variant):'selected experiment'}`}</figcaption><canvas ref={right} width={width} height={height} aria-label={label?'M22 comparison video':'M21 comparison video'}/></figure><p>Use the source-video controls below to play or seek both views. Both use the same decoded pixels; a marker appears only when its evidence matches the displayed frame.</p></div>;
}

export default function AdvancedPerception({metadata,frame,ball,m20,video,width,height,fps,onExperiment,onSeek}:{metadata:AdvancedMetadata|null,frame?:AdvancedFrame|null,ball?:BallState|null,m20?:{frame:number,ball:BallState}|null,video:HTMLVideoElement|null,width:number,height:number,fps:number,onExperiment:(id:string)=>void,onSeek:(time:number)=>void}){
 const [compare,setCompare]=useState(false);
 if(!metadata)return <section className="card advanced-panel"><p role="status">Loading M21 experiment comparisons…</p></section>;
 const current=metadata.experiments.find(e=>e.id===metadata.selected_experiment),baseline=metadata.experiments.find(e=>e.id==='m20_control');
 const percent=(value:number|null|undefined)=>value===null||value===undefined?'Unavailable':`${(value*100).toFixed(1)}%`;
 return <section className="card advanced-panel" aria-label="M21 Advanced Perception"><div className="section-bar"><h2>M21 Advanced Perception</h2><span className="source-pill">Opt-in experiment</span></div>
 <p>Human-reviewed development results · new independent test labels are pending. Counts below describe the reviewed sample, not the entire recording.</p>
 <div className="advanced-controls"><label>Experiment <select aria-label="M21 experiment" value={metadata.selected_experiment} onChange={e=>onExperiment(e.target.value)}>{metadata.experiments.map(e=><option key={e.id} value={e.id}>{experimentLabel(e.id)}</option>)}</select></label><label><input type="checkbox" checked={compare} onChange={e=>setCompare(e.target.checked)}/> Compare M20 and M21 side by side</label><a href="/?review=m22&recording=team_goal">Open football training review</a></div>
 <div className="advanced-metrics"><div><span>Ball precision</span><strong>{percent(current?.metrics.ball.precision)}</strong><small>M20: {percent(baseline?.metrics.ball.precision)}</small></div><div><span>Visible balls found</span><strong>{current?.metrics.ball.tp??'—'}/{(current?.metrics.ball.tp??0)+(current?.metrics.ball.fn??0)}</strong><small>M20: {baseline?.metrics.ball.tp??'—'} · recall {percent(current?.metrics.ball.recall)}</small></div><div><span>False ball detections</span><strong>{current?.metrics.ball.fp??'—'}</strong><small>M20: {baseline?.metrics.ball.fp??'—'}</small></div><div><span>Reviewed passes matched</span><strong>{current?.metrics.passes.tp??'—'}/{(current?.metrics.passes.tp??0)+(current?.metrics.passes.fn??0)}</strong><small>Qualified estimates only</small></div></div>
 {frame&&<p data-testid="m21-state">Frame {frame.frame} · {frame.directly_observed?'Observed ball':frame.reconstruction?'Offline reconstruction':'No accepted current ball'} · {frame.association_mode==='causal_ball_only'?'Causal ball layer; player and event context is offline':'Offline analysis; may use later evidence'}</p>}
 {compare&&<Comparison video={video} width={width} height={height} fps={fps} frame={frame} m20={m20} ball={ball}/>}
 <details><summary>Compare all experiments</summary><div className="trajectory-table"><table aria-label="M21 human-reviewed experiment comparison"><caption>This clip · unchanged reference rules · development selection, not final-test acceptance</caption><thead><tr><th>Experiment</th><th>Ball precision</th><th>Recall</th><th>False detections</th><th>Pass matches</th></tr></thead><tbody>{metadata.experiments.map(e=><tr key={e.id}><th scope="row"><button className="secondary" onClick={()=>onExperiment(e.id)}>{experimentLabel(e.id)}</button></th><td>{percent(e.metrics.ball.precision)}</td><td>{percent(e.metrics.ball.recall)}</td><td>{e.metrics.ball.fp}</td><td>{e.metrics.passes.tp}</td></tr>)}</tbody></table></div></details>
 {frame&&<details><summary>Why this frame was accepted or rejected</summary><p>Neural visual confidence is retained. Temporal scores are uncalibrated checks; they do not turn a prediction into an observation.</p><ul>{[...frame.candidates].sort((a,b)=>Number(b.selected)-Number(a.selected)||b.visual_confidence-a.visual_confidence).slice(0,16).map(c=><li key={c.candidate_id}><strong>{c.selected?'Selected':'Not selected'} · {c.candidate_id}</strong> · visual {(c.visual_confidence*100).toFixed(1)}% · {c.decision_reason?.replaceAll('_',' ').toLowerCase()??c.deterministic.reason.replaceAll('_',' ').toLowerCase()}{c.temporal_score!==null?` · temporal ${(c.temporal_score*100).toFixed(1)}%`:''}{c.cross_detector?` · second detector ${(c.cross_detector.visual_confidence*100).toFixed(1)}%`:''}</li>)}</ul>{frame.reconstruction&&<p>Inferred between source frames {frame.reconstruction.support_frames.join(' and ')}. This point is excluded from direct-observation pass gates.</p>}<button className="secondary" onClick={()=>onSeek(frame.timestamp)}>Seek to this evidence frame</button></details>}
 <details><summary>Player continuity, camera mapping and research limits</summary><p>Alternative player tracks are anonymous comparisons. Existing M9 identities and manual confirmations remain the source for player focus and pass attribution.</p><p>Conditional continuity matches: {current?.metrics.identity.correct_pairs??0}/{current?.metrics.identity.reviewed_matched_pairs??0}; unmatched pairs: {current?.metrics.identity.unknown_or_unmatched_pairs??0}. The development set contains no different-person negative references.</p><p>Camera refinements measure relative field motion; independently labelled pitch landmarks are still required for metric accuracy.</p><p>3D flight: {metadata.physics.accepted?'experimental fit only':'unavailable without verified full camera geometry'}. No authoritative ball height or shot speed is published.</p></details>
 </section>;
}
