import {useEffect,useState} from 'react';
import {z} from 'zod';
import {request} from './api';
import type {BallState} from './BallDebug';
import {Comparison} from './AdvancedPerception';
import {intelligentMetadataSchema,type IntelligentFrame} from './intelligentContracts';
export function IntelligentOverlay({frame,ball,rejected=false}:{frame:IntelligentFrame,ball:BallState,rejected?:boolean}){
 if(frame.frame!==ball.frame)return null;
 return <g aria-label="M23 intelligent tracking evidence" data-testid="m23-overlay" data-frame={frame.frame} pointerEvents="none">
 {frame.memory?.search_regions.map((r,i)=><rect key={'r'+i} data-testid="m23-search" x={r.bbox[0]} y={r.bbox[1]} width={r.bbox[2]} height={r.bbox[3]} fill="none" stroke="#f0bf67" strokeWidth="1.5" strokeDasharray="9 6"><title>Fresh visual search: {r.kind??'trajectory'}</title></rect>)}
 {frame.memory?.hypotheses.filter(h=>h.confirmed).map(h=>{const [[a,b],[,d]]=h.covariance,delta=Math.sqrt((a-d)**2+4*b*b),rx=Math.sqrt(4.60517*Math.max(0,(a+d+delta)/2)),ry=Math.sqrt(4.60517*Math.max(0,(a+d-delta)/2)),angle=Math.atan2(2*b,a-d)*90/Math.PI;return <ellipse key={h.track_id} data-testid="m23-uncertainty" cx={h.mean[0]} cy={h.mean[1]} rx={rx} ry={ry} transform={`rotate(${angle} ${h.mean[0]} ${h.mean[1]})`} fill="none" stroke="#6de1ff" strokeDasharray="5 5"><title>Model uncertainty; this is a prediction, not a detection</title></ellipse>})}
 {frame.trajectory_history.map(h=><circle key={h.frame} cx={h.current_camera_image[0]} cy={h.current_camera_image[1]} r="3" fill="#81d3ca" opacity=".6"><title>Camera-compensated historical observation, frame {h.frame}</title></circle>)}
 {frame.reconstruction&&<circle data-testid="m23-reconstructed" cx={frame.reconstruction.image[0]} cy={frame.reconstruction.image[1]} r="9" fill="none" stroke="#ffad70" strokeDasharray="3 3"><title>Offline reconstruction; not an observed football</title></circle>}
 {frame.memory?.predicted_image_position&&!frame.directly_observed&&<path d={`M${frame.memory.predicted_image_position[0]-6} ${frame.memory.predicted_image_position[1]}h12 M${frame.memory.predicted_image_position[0]} ${frame.memory.predicted_image_position[1]-6}v12`} stroke="#6de1ff" strokeWidth="2"><title>Predicted position only</title></path>}
 {frame.candidates.filter(c=>c.selected||rejected).map(c=><rect key={c.candidate_id} data-testid={c.selected?'m23-observed':'m23-rejected'} x={c.bbox[0]} y={c.bbox[1]} width={c.bbox[2]} height={c.bbox[3]} fill="none" stroke={c.selected?'#b5ff73':c.observation_eligible?'#df9eff':'#ff8c8c'} strokeWidth={c.selected?3:1} strokeDasharray={c.selected?undefined:'3 3'}><title>{c.selected?'Observed':'Unselected'} · {c.origin} · {c.decision_reason}</title></rect>)}
 </g>;
}
const percent=(v:number|null)=>v===null?'—':(100*v).toFixed(1)+'%';
export default function IntelligentTracking({recordingId,frame,ball,reference,video,width,height,fps,onSeek}:{recordingId:string,frame?:IntelligentFrame,ball?:BallState|null,reference?:{frame:number,ball:BallState},video:HTMLVideoElement|null,width:number,height:number,fps:number,onSeek:(t:number)=>void}){
 const [meta,setMeta]=useState<z.infer<typeof intelligentMetadataSchema>|null>(null),[error,setError]=useState(''),[benchmark,setBenchmark]=useState<'new_150'|'original'>('new_150'),[kind,setKind]=useState('all');
 useEffect(()=>{let active=true;setMeta(null);request('/api/matches/'+encodeURIComponent(recordingId)+'/cv/intelligent',intelligentMetadataSchema).then(r=>{if(active){setMeta(r);setError('')}}).catch(e=>{if(active)setError(e.message)});return()=>{active=false}},[recordingId]);
 const cases=meta?.metrics.navigation.filter(n=>kind==='all'||n.kind===kind)??[];
 const kinds=[...new Set(meta?.metrics.navigation.map(n=>n.kind))];
 return <section className="card" aria-label="M23 intelligent ball tracking"><div className="section-bar"><h2>M23 — Intelligent Ball Tracking</h2><span className="source-pill">Experimental · offline</span></div>
 <p>Observe → remember → predict → search → verify → link. Green boxes are current visual detections. Cyan ellipses are model predictions; amber boxes are fresh search crops; orange points are reconstructed positions.</p>
 <p role="note">These clips are development evidence. Independent accuracy, true reacquisition identity and trajectory fragmentation remain unverified. M14 remains the default.</p>
 {error&&<p role="alert">{error}</p>}
 {frame&&<><p data-testid="m23-state">Frame {frame.frame} · {frame.state} · causal memory: {frame.memory?.state??'unavailable'}{frame.memory?.reacquired_with_current_visual_evidence?' · visually reacquired by causal tracker':''}</p><p>{frame.reasons.join(' · ')}. {frame.memory?.reason}</p><p>Last causal observation: {frame.memory?.last_direct_observation?.frame??'none'} · missed {frame.memory?.missed_seconds?.toFixed(3)??'—'} s · {frame.memory?.camera_motion_source}. Player interaction: {frame.interaction.state} ({frame.interaction.reason}); proximity does not prove a touch or possession.</p></>}
 {frame&&ball&&reference&&<Comparison video={video} width={width} height={height} fps={fps} frame={{frame:frame.frame,variant:'full_m23'}} ball={ball} m20={reference} label="M23" referenceLabel="M21 temporal + offline"/>}
 {meta&&<><label>Evaluation sample <select aria-label="M23 evaluation sample" value={benchmark} onChange={e=>setBenchmark(e.target.value as typeof benchmark)}><option value="new_150">New 150-frame review · this recording</option><option value="original">Original ball references</option></select></label>
 <div style={{overflowX:'auto'}}><table aria-label="M23 controlled comparisons"><caption>This recording only · the original and new samples remain separate</caption><thead><tr><th>System</th><th>TP / FP / FN</th><th>Precision</th><th>Recall</th><th>F1</th><th>Known passes</th></tr></thead><tbody>{meta.comparisons.map(r=>{const m=r[benchmark];return <tr key={r.id}><th>{r.id.replaceAll('_',' ')}</th><td>{m.tp} / {m.fp} / {m.fn}</td><td>{percent(m.precision)}</td><td>{percent(m.recall)}</td><td>{m.f1?.toFixed(3)??'—'}</td><td>{r.passes.tp}/{r.passes.tp+r.passes.fn} · FP {r.passes.fp}</td></tr>})}</tbody></table></div>
 <details><summary>Frozen M20/M21/M22 comparisons on this recording</summary><ul>{meta.frozen_baselines.map(r=><li key={r.id}>{r.id}: {r.new_150.tp}/{r.new_150.fp}/{r.new_150.fn} TP/FP/FN · F1 {r.new_150.f1?.toFixed(3)}</li>)}</ul></details>
 <p>Whole clip: {Object.entries(meta.metrics.engineering.states).map(([k,v])=>`${k}: ${v}`).join(' · ')}. These are model states, not accuracy.</p>
 <p>Shared inference campaign: {meta.runtime.processing_fps.toFixed(2)} FPS · {meta.runtime.recorded_processing_seconds.toFixed(1)} s · {meta.runtime.timing.extra_crop_views} additional crops · peak process memory {(meta.runtime.peak_rss_bytes_macos/2**30).toFixed(2)} GiB. Cached global complementary outputs are reused; this is not a cold end-to-end real-time benchmark.</p>
 <label>Inspect an example <select aria-label="M23 example category" value={kind} onChange={e=>setKind(e.target.value)}><option value="all">All diagnostics</option>{kinds.map(k=><option key={k} value={k}>{k.replaceAll('_',' ')}</option>)}</select></label>
 <div className="review-navigation" style={{display:'flex',gap:8,flexWrap:'wrap',maxHeight:200,overflowY:'auto'}}>{cases.map((n,i)=><button className="secondary" key={n.kind+':'+n.frame+':'+i} onClick={()=>onSeek(n.frame/fps)}>{n.kind.replaceAll('_',' ')} · frame {n.frame}</button>)}</div>
 <p>Model set {meta.model_set_sha256.slice(0,12)} · evidence {meta.ball_revision.slice(0,12)}</p></>}
 </section>;
}
