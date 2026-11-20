import {useEffect,useRef,useState} from 'react';
type Box=[number,number,number,number];
type Proposal={proposal_id:string,bbox:Box,confidence:number,source_pipeline:string};
type Task={task_id:string,frame:number,timestamp:number,sequence_id:string,sequence_frames:number[],proposals:Proposal[],selection_reason:string};
type Label={task_id:string,reviewer_id:string,state:'visible'|'not_visible'|'uncertain',bbox:Box|null,accepted_proposal:string|null,rejected_proposals:string[],notes:string};
type Data={dataset_fingerprint:string,reference_role:'development_only',model_proposals_are_truth:false,source:{width:number,height:number,fps:number,video_sha256:string},tasks:Task[],labels:Label[],sources:{recording_id:string,tasks:number,saved_tasks:number}[]};
const empty=()=>({state:'uncertain' as Label['state'],bbox:null as Box|null,accepted_proposal:null as string|null,rejected_proposals:[] as string[],notes:''});

export default function M22HumanReview(){
 const requested=new URLSearchParams(location.search).get('recording');const recording=['test_match','team_goal','southampton'].includes(requested??'')?requested!:'test_match';
 const base=`/api/cv/m22/matches/${recording}/review`;
 const [data,setData]=useState<Data|null>(null),[index,setIndex]=useState(0),[reviewer,setReviewer]=useState(''),[label,setLabel]=useState(empty),[attest,setAttest]=useState(false),[busy,setBusy]=useState(false),[message,setMessage]=useState(''),[saved,setSaved]=useState(false),[zoom,setZoom]=useState(1),[centerMode,setCenterMode]=useState(false),[dirty,setDirty]=useState(false);
 const anchor=useRef<[number,number]|null>(null);const current=data?.tasks[index];
 useEffect(()=>{let active=true;fetch(base+'/tasks').then(async r=>{if(!r.ok)throw Error('M22 development tasks are unavailable.');return r.json()}).then(value=>{if(!active)return;setData(value);const known=new Set(value.labels.map((r:Label)=>r.task_id));const next=value.tasks.findIndex((t:Task)=>!known.has(t.task_id));setIndex(next<0?0:next)}).catch(e=>{if(active)setMessage(e.message)});return()=>{active=false}},[base]);
 useEffect(()=>{
  const old=data?.labels.find(r=>r.task_id===current?.task_id&&(!reviewer||r.reviewer_id===reviewer));
  setLabel(old?{state:old.state,bbox:old.bbox,accepted_proposal:old.accepted_proposal??null,rejected_proposals:old.rejected_proposals??[],notes:old.notes??''}:empty());setAttest(false);setSaved(Boolean(old));setDirty(false);setMessage('');
 },[current?.task_id,reviewer]);
 const edit=(value:Partial<ReturnType<typeof empty>>)=>{setLabel(v=>({...v,...value}));setSaved(false);setDirty(true);setMessage('')};
 const point=(e:React.PointerEvent<SVGSVGElement>):[number,number]=>{const r=e.currentTarget.getBoundingClientRect();return [Math.max(0,Math.min(data!.source.width,(e.clientX-r.left)/r.width*data!.source.width)),Math.max(0,Math.min(data!.source.height,(e.clientY-r.top)/r.height*data!.source.height))]};
 const validBox=label.bbox&&label.bbox.every(Number.isFinite)&&label.bbox[0]>=0&&label.bbox[1]>=0&&label.bbox[2]>0&&label.bbox[3]>0&&data&&label.bbox[0]+label.bbox[2]<=data.source.width&&label.bbox[1]+label.bbox[3]<=data.source.height;
 const canSave=reviewer.trim().length>=2&&attest&&(label.state!=='visible'||validBox)&&!busy;
 async function save(){
  if(!data||!current||!canSave)return;setBusy(true);setMessage('');
  const body={...label,task_id:current.task_id,reviewer_id:reviewer.trim(),reviewed_from_source:true,origin:'human',model_assisted:true};
  try{
   const response=await fetch(base+'/labels',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const result=await response.json();
   if(!response.ok||result.saved!==true)throw Error(typeof result.detail==='string'?result.detail:'This label was not saved. Check your box and proposal decisions.');
   const next=await fetch(base+'/tasks');if(!next.ok)throw Error('Saved, but progress could not refresh. Reload to check the saved label.');setData(await next.json());setSaved(true);setDirty(false);setMessage('Saved human-verified development label.');
  }catch(e){setMessage(e instanceof Error?e.message:'Could not save. Your edits remain here.')}finally{setBusy(false)}
 }
 const navigate=(next:number)=>{if(dirty){setMessage('Save this label or discard edits before changing frames.');return}setIndex(next)};
 return <main className="app-shell human-review m22-review"><header className="section-bar"><h1>M22 football training review</h1><a href={`/?recording=${recording}`}>Return to analysis</a></header>
 <section className="card"><p>Only test_match, team_goal and southampton. These short clips were inspected for active football play. Proposals are model suggestions; your saved decisions become development supervision, never independent test truth.</p>
 <nav className="review-recordings" aria-label="M22 review recordings">{['test_match','team_goal','southampton'].map(name=><a key={name} href={`/?review=m22&recording=${name}`}>{name}</a>)}</nav>
 {data&&<p data-testid="m22-review-progress">Saved: {data.sources.reduce((n,s)=>n+s.saved_tasks,0)}/{data.sources.reduce((n,s)=>n+s.tasks,0)} total · {data.sources.find(s=>s.recording_id===recording)?.saved_tasks}/{data.tasks.length} in this clip</p>}
 <label className="review-field">Reviewer ID<input aria-label="Reviewer ID" maxLength={80} value={reviewer} onChange={e=>setReviewer(e.target.value)} disabled={busy}/></label>
 <a href="/api/cv/m22/review/export" download>Export temporal training labels</a>
 <details><summary>See the selected source footage and selection reasons</summary><p>No full-match footage is included. Each selected group contains up to five nearby frames. Selection prioritizes disagreements, missing candidates and uncertainty during active-play footage.</p>{data&&[0,1].map(page=><img key={page} loading="lazy" src={base+`/contact-sheet/${page}.jpg`} alt={`${recording} selected source frames, page ${page+1}`} style={{width:'100%'}}/>)}</details></section>
 {data&&current&&<section className="card" aria-label="M22 ball annotation"><h2>Frame {current.frame} · {current.timestamp.toFixed(3)}s · {index+1}/{data.tasks.length}</h2><p>{current.selection_reason}. Sequence: {current.sequence_frames.join(', ')}.</p>
 <div className="m22-context">{[-2,-1].map(delta=><figure key={delta}><img src={base+`/frame.jpg?frame=${Math.max(0,current.frame+delta)}`} alt={`Source context frame ${Math.max(0,current.frame+delta)}`}/><figcaption>Context · frame {Math.max(0,current.frame+delta)}</figcaption></figure>)}</div>
 <label className="review-field">Frame zoom<input aria-label="Frame zoom" type="range" min="1" max="4" step=".5" value={zoom} onChange={e=>setZoom(Number(e.target.value))}/></label>
 <label className="review-check"><input type="checkbox" checked={centerMode} onChange={e=>setCenterMode(e.target.checked)}/>Move box center with a click</label>
 <p>Box only the in-play ball. Accept a correct proposal, draw a replacement, or choose Not visible / Uncertain. Rejecting a proposal marks only that region as a non-ball; it does not mark the whole frame absent.</p>
 <div style={{overflow:'auto',maxHeight:650}}><svg role="img" aria-label="Current source frame: verify the football" viewBox={`0 0 ${data.source.width} ${data.source.height}`} style={{width:`${zoom*100}%`,display:'block',touchAction:'none',cursor:'crosshair'}} onPointerDown={e=>{if(busy)return;anchor.current=point(e);e.currentTarget.setPointerCapture?.(e.pointerId)}} onPointerCancel={()=>{anchor.current=null}} onPointerUp={e=>{if(busy||!anchor.current)return;const [x,y]=anchor.current,[a,b]=point(e);anchor.current=null;if(centerMode){const w=label.bbox?.[2]??8,h=label.bbox?.[3]??8;edit({state:'visible',bbox:[Math.max(0,Math.min(data.source.width-w,a-w/2)),Math.max(0,Math.min(data.source.height-h,b-h/2)),w,h],accepted_proposal:null})}else if(Math.abs(a-x)>0&&Math.abs(b-y)>0)edit({state:'visible',bbox:[Math.min(x,a),Math.min(y,b),Math.abs(a-x),Math.abs(b-y)],accepted_proposal:null})}}>
 <image href={base+`/frame.jpg?frame=${current.frame}`} width={data.source.width} height={data.source.height}/>
 {current.proposals.map((p,i)=><g key={p.proposal_id} opacity={label.rejected_proposals.includes(p.proposal_id)?.3:.8}><rect x={p.bbox[0]} y={p.bbox[1]} width={p.bbox[2]} height={p.bbox[3]} fill="none" stroke="#61ccff" strokeDasharray="5 3"/><text x={p.bbox[0]} y={Math.max(12,p.bbox[1]-3)} fill="#61ccff" fontSize={Math.max(10,data.source.width/80)}>Proposal {i+1}</text></g>)}
 {label.bbox&&<rect x={label.bbox[0]} y={label.bbox[1]} width={label.bbox[2]} height={label.bbox[3]} fill="none" stroke="#ffdf00" strokeWidth={Math.max(2,data.source.width/400)}/>}</svg></div>
 <div className="review-actions">{current.proposals.map((p,i)=><div key={p.proposal_id}><span>Proposal {i+1} · {p.source_pipeline} · {(p.confidence*100).toFixed(1)}%</span><button disabled={busy} type="button" onClick={()=>edit({state:'visible',bbox:p.bbox,accepted_proposal:p.proposal_id,rejected_proposals:label.rejected_proposals.filter(id=>id!==p.proposal_id)})}>Accept proposal {i+1}</button><button disabled={busy} type="button" onClick={()=>edit({accepted_proposal:label.accepted_proposal===p.proposal_id?null:label.accepted_proposal,rejected_proposals:label.rejected_proposals.includes(p.proposal_id)?label.rejected_proposals.filter(id=>id!==p.proposal_id):[...label.rejected_proposals,p.proposal_id],...(label.accepted_proposal===p.proposal_id?{bbox:null,state:'uncertain' as const}:{})})}>{label.rejected_proposals.includes(p.proposal_id)?'Undo rejection':'Not the ball'} {i+1}</button></div>)}</div>
 <label className="review-field">Ball visibility<select aria-label="Ball visibility" value={label.state} disabled={busy} onChange={e=>edit({state:e.target.value as Label['state'],...(e.target.value!=='visible'?{bbox:null,accepted_proposal:null}:{})})}><option value="uncertain">Uncertain</option><option value="visible">Visible</option><option value="not_visible">Not visible</option></select></label>
 <label className="review-field">Notes<textarea aria-label="Notes" maxLength={500} value={label.notes} disabled={busy} onChange={e=>edit({notes:e.target.value})}/></label>
 <label className="review-check"><input type="checkbox" disabled={busy} checked={attest} onChange={e=>setAttest(e.target.checked)}/>I checked the actual source frame and corrected any proposal errors.</label>
 {!canSave&&!busy&&<p>To save: enter your reviewer ID, verify the source, and draw a valid box if the ball is visible.</p>}
 <div className="review-actions"><button disabled={busy||index===0} onClick={()=>navigate(index-1)}>Previous selected frame</button><button disabled={!canSave} className="primary" onClick={()=>void save()}>{busy?'Saving…':'Save verified development label'}</button><button disabled={busy||index===data.tasks.length-1} onClick={()=>navigate(index+1)}>Next selected frame</button>{dirty&&<button disabled={busy} onClick={()=>{setLabel(empty());setDirty(false);setSaved(false);setMessage('Edits discarded. Saved labels remain unchanged.')}}>Discard edits</button>}</div>
 {saved&&<p>Saved label is available when you return to this frame.</p>}
 </section>}
 {message&&<p role={message.startsWith('Saved')?'status':'alert'}>{message}</p>}
 </main>;
}
