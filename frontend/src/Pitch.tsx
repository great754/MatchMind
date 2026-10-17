import type {Event,Replay,Tactical} from './contracts';
import {useIdentity} from './identity';
import {replayIndex} from './time';
export const teamColor=(data:Replay,team:string)=>data.side_colors[team==='left'?'0':'1']===0?'#79a8ff':'#ff837b';
export function Pitch({data,time,selected,onSelect,event,state,attributedOwner}:{data:Replay|null,time:number,selected:number|null,onSelect:(id:number)=>void,event:Event|null,state:Tactical|null,attributedOwner?:number|null}){
 const labels=useIdentity();
 if(!data)return <div className="empty-pitch">Tactical positions unavailable for this recording.</div>;
 const i=replayIndex(data,time),ball=data.ball[i],owner=attributedOwner===undefined?data.owner[i]:attributedOwner;const phase=data.states[i];
 const path=event&&['start_x_m','start_y_m','end_x_m','end_y_m'].every(k=>typeof event.metrics[k]==='number')?event.metrics:null;
 return <svg className="pitch" viewBox="-5 -5 115 78" role="img" aria-label={`Tactical pitch at ${time.toFixed(2)} seconds`}>
 <defs><pattern id="stripes" width="21" height="68" patternUnits="userSpaceOnUse"><rect width="10.5" height="68" fill="#142b29"/><rect x="10.5" width="10.5" height="68" fill="#18322f"/></pattern><marker id="arrow" markerWidth="4" markerHeight="4" refX="3" refY="2" orient="auto"><path d="M0 0L4 2L0 4" fill="#bcebc9"/></marker></defs>
 <rect width="105" height="68" rx=".6" fill="url(#stripes)"/><g fill="none" stroke="#8caaa0" strokeWidth=".3" opacity=".6"><rect width="105" height="68"/><path d="M52.5 0V68"/><circle cx="52.5" cy="34" r="9.15"/><path d="M0 13.84H16.5V54.16H0 M105 13.84H88.5V54.16H105 M0 24.84H5.5V43.16H0 M105 24.84H99.5V43.16H105"/><circle cx="11" cy="34" r=".35"/><circle cx="94" cy="34" r=".35"/><path d="M0 30.34H-2V37.66H0 M105 30.34H107V37.66H105"/></g>
 {path&&Math.abs(time-event!.timestamp)<3&&<path d={`M${path.start_x_m} ${path.start_y_m}L${path.end_x_m} ${path.end_y_m}`} stroke="#bcebc9" strokeWidth=".5" strokeDasharray="1 1" markerEnd="url(#arrow)"/>}
 {data.positions[i]?.map((p,j)=>p[0]===null||p[1]===null?null:<g key={data.ids[j]} role="button" tabIndex={0} aria-label={`Select ${labels.player(data.ids[j])}`} onClick={()=>onSelect(data.ids[j])} onKeyDown={e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();onSelect(data.ids[j]);}}} className="actor-dot">
 <circle cx={p[0]} cy={p[1]} r="2.5" fill="transparent"/>{owner===data.ids[j]&&<circle cx={p[0]} cy={p[1]} r="1.8" fill="none" stroke="#d1eb9d" strokeWidth=".35"/>}{selected===data.ids[j]&&<circle cx={p[0]} cy={p[1]} r="2.15" fill="none" stroke="white" strokeWidth=".45"/>}<circle cx={p[0]} cy={p[1]} r="1.05" fill={labels.color(labels.identity.players[data.ids[j]]?.team??data.teams[j])} stroke="#0a151b" strokeWidth=".3"/>{(selected===data.ids[j]||state?.pressure.carrier_actor_id===data.ids[j])&&<text x={p[0]} y={p[1]-3} textAnchor="middle" fill="#fff" fontSize="2.1">{labels.player(data.ids[j])}</text>}</g>)}
 {ball?.[0]!==null&&ball?.[1]!==null&&ball&&<circle cx={ball[0]!} cy={ball[1]!} r=".7" fill="white" stroke="#101b1c" strokeWidth=".3"/>}
 </svg>;
}
