import {useCallback,useEffect,useRef,useState} from 'react';
type Box=[number,number,number,number];
type Proposal={proposal_id:string,bbox:Box,confidence:number|null,source_pipeline:string};
type Task={task_id:string,frame:number,timestamp:number,sequence_id:string,sequence_frames:number[],proposals:Proposal[],selection_reason:string};
type Choice={state:'visible'|'not_visible'|'uncertain',bbox:Box|null,accepted_proposal:string|null,rejected_proposals:string[],reviewed_proposals:string[],notes:string};
export type Label=Choice&{task_id:string,reviewer_id:string};
export type Data={dataset_fingerprint:string,reference_role:'development_only',model_proposals_are_truth:false,supports_reviewed_proposals?:boolean,source:{recording_id?:string,width:number,height:number,fps:number,video_sha256:string},tasks:Task[],labels:Label[],sources:{recording_id:string,tasks:number,saved_tasks:number}[]};
const originalSources=['test_match','team_goal','southampton'];
const empty=():Choice=>({state:'uncertain',bbox:null,accepted_proposal:null,rejected_proposals:[],reviewed_proposals:[],notes:''});
function readStorage(key:string){try{return localStorage.getItem(key)}catch{return null}}
function store(key:string,value:string|null){try{if(value===null)localStorage.removeItem(key);else localStorage.setItem(key,value);return true}catch{return false}}
function overlap(a:Box,b:Box){const intersection=Math.max(0,Math.min(a[0]+a[2],b[0]+b[2])-Math.max(a[0],b[0]))*Math.max(0,Math.min(a[1]+a[3],b[1]+b[3])-Math.max(a[1],b[1]));return intersection/(a[2]*a[3]+b[2]*b[3]-intersection)}
function restored(label?:Label):Choice{return label?{state:label.state,bbox:label.bbox,accepted_proposal:label.accepted_proposal??null,rejected_proposals:label.rejected_proposals??[],reviewed_proposals:label.reviewed_proposals?.length?label.reviewed_proposals:[...(label.accepted_proposal?[label.accepted_proposal]:[]),...(label.rejected_proposals??[])],notes:label.notes??''}:empty()}

export default function M22HumanReview({reviewKey='m22',sourceRecordings=originalSources}:{reviewKey?:'m22'|'m23',sourceRecordings?:string[]}={}){
 const reviewerKey=`tactiq.${reviewKey}.reviewer`;
 const requested=new URLSearchParams(location.search).get('recording');const recording=sourceRecordings.includes(requested??'')?requested!:sourceRecordings[0];
 const base=`/api/cv/${reviewKey}/matches/${recording}/review`;
 const [data,setData]=useState<Data|null>(null),[index,setIndex]=useState(0),[reviewer,setReviewer]=useState(''),[busy,setBusy]=useState(false),[message,setMessage]=useState(''),[error,setError]=useState('');
 useEffect(()=>{let active=true;fetch(base+'/tasks').then(async r=>{if(!r.ok)throw Error('Review frames are unavailable. Please reload.');return r.json()}).then((value:Data)=>{
  if(!active)return;const names=[...new Set(value.labels.map(r=>r.reviewer_id))];const id=readStorage(reviewerKey)||(names.length===1?names[0]:`Reviewer-${crypto.randomUUID().slice(0,8)}`);
  store(reviewerKey,id);setReviewer(id);setData(value);const known=new Set(value.labels.filter(r=>r.reviewer_id===id).map(r=>r.task_id));const next=value.tasks.findIndex(t=>!known.has(t.task_id));setIndex(next<0?value.tasks.length-1:next);
 }).catch(e=>{if(active)setError(e.message)});return()=>{active=false}},[base,reviewerKey]);
 const savedLabel=useCallback((value:Label)=>{
  setData(previous=>{if(!previous)return previous;const labels=[...previous.labels.filter(r=>r.task_id!==value.task_id||r.reviewer_id!==value.reviewer_id),value];return {...previous,labels,sources:previous.sources.map(s=>s.recording_id===recording?{...s,saved_tasks:new Set(labels.map(r=>r.task_id)).size}:s)}});
  const known=new Set([...(data?.labels.filter(r=>r.reviewer_id===value.reviewer_id).map(r=>r.task_id)??[]),value.task_id]);
  const after=data?.tasks.findIndex((t,i)=>i>index&&!known.has(t.task_id))??-1;
  const next=after>=0?after:data?.tasks.findIndex(t=>!known.has(t.task_id))??-1;
  if(next>=0)setIndex(next);
  setMessage(next<0?'Saved. This clip is complete. You can go back to make corrections.':'Saved. Moved to the next frame.');
 },[data,index,recording]);
 const current=data?.tasks[index];const count=data?.sources.find(s=>s.recording_id===recording)?.saved_tasks??0;
 return <main className="app-shell human-review m22-review">
  <header className="section-bar"><div><h1>{reviewKey==='m23'?'Giroud ball review':'M22 football training review'}</h1><p>Check the ball. Choose each proposal. We save and move on automatically.</p></div><a href={`/?recording=${recording}${reviewKey==='m23'?'&ball_pipeline=m23':''}`}>Back to analysis</a></header>
  <section className="card m22-review-toolbar"><label className="review-field">Clip<select aria-label="Review clip" value={recording} disabled={busy} onChange={e=>location.assign(`/?review=${reviewKey}&recording=${e.target.value}`)}>{sourceRecordings.map(name=><option key={name}>{name}</option>)}</select></label><p data-testid={`${reviewKey}-review-progress`}><strong>{count}/{data?.tasks.length??(reviewKey==='m23'?40:50)}</strong> frames reviewed in this clip</p><span className="source-pill">Auto-save on</span></section>
  {message&&<p className="review-save-success" role="status">{message}</p>}{error&&<p role="alert">{error}</p>}
  {data&&current&&reviewer&&<FrameReview key={`${recording}:${reviewer}:${current.task_id}`} reviewKey={reviewKey} data={data} task={current} base={base} reviewer={reviewer} old={data.labels.find(r=>r.task_id===current.task_id&&r.reviewer_id===reviewer)} index={index} busyChanged={setBusy} savedLabel={savedLabel} navigate={setIndex} clearMessage={()=>setMessage('')}/>}
  <details className="card m22-review-help"><summary>Tips & examples</summary><div className="m22-example-grid"><p><strong>Accept a proposal</strong><br/>Its box covers the actual football. Review the other proposals too; more than one may cover the same ball.</p><p><strong>Not the ball</strong><br/>The box covers a boot, player, pitch line or another object. This rejects only that box.</p><p><strong>No good proposal?</strong><br/>Draw a tight box around the ball. If it is hidden, choose “Ball not visible”. If you cannot tell, choose “Can't tell”.</p></div><p>Your choices confirm that you checked the source image. Only your completed decisions are saved to the server. Incomplete choices stay as a draft in this browser. A model suggestion is never saved just because you opened a frame.</p></details>
  {data&&<details className="card m22-review-help"><summary>Review settings & source details</summary><label className="review-field">Reviewer ID<input aria-label="Reviewer ID" value={reviewer} maxLength={80} disabled={busy} onChange={e=>{const id=e.target.value;setReviewer(id);if(id.trim().length>=2)store(reviewerKey,id)}}/></label><p>Development labels only · {data.sources.reduce((n,s)=>n+s.saved_tasks,0)}/{data.sources.reduce((n,s)=>n+s.tasks,0)} reviewed across {data.sources.length} approved {data.sources.length===1?'clip':'clips'}. Original reviews are preserved.</p><a href={`/api/cv/${reviewKey}/review/export`} download>Export training labels</a><details><summary>See the selected source footage and selection reasons</summary><p>Only the approved short clips are included. {reviewKey==='m23'?'32 frames are spread across this clip in short sequences; eight more cover diagnostic examples.':'Selection prioritizes model disagreements, missing candidates and uncertainty.'}</p>{[0,1].map(page=><img key={page} loading="lazy" src={base+`/contact-sheet/${page}.jpg`} alt={`${recording} selected source frames, page ${page+1}`} style={{width:'100%'}}/>)}</details></details>}
 </main>;
}

export function FrameReview({reviewKey,data,task,base,reviewer,old,index,busyChanged,savedLabel,navigate,clearMessage}:{reviewKey:'m22'|'m23'|'m25',data:Data,task:Task,base:string,reviewer:string,old?:Label,index:number,busyChanged:(busy:boolean)=>void,savedLabel:(label:Label)=>void,navigate:(index:number)=>void,clearMessage:()=>void}){
 const [hiddenModels,setHiddenModels]=useState<string[]>([]);
 const draftKey=`tactiq.${reviewKey}.draft:${data.dataset_fingerprint}:${reviewer}:${task.task_id}`;
 const [initial]=useState(()=>{try{const value=JSON.parse(readStorage(draftKey)??'null');return value&&typeof value.decisionMade==='boolean'&&Array.isArray(value.label?.reviewed_proposals)?value:null}catch{return null}});
 const [label,setLabel]=useState<Choice>(()=>initial?.label??restored(old)),[decisionMade,setDecisionMade]=useState(Boolean(initial?.decisionMade??old)),[dirty,setDirty]=useState(Boolean(initial)),[saving,setSaving]=useState(false),[saveError,setSaveError]=useState(''),[zoom,setZoom]=useState(1),[centerMode,setCenterMode]=useState(false),[draftStored,setDraftStored]=useState(Boolean(initial)),[editingNotes,setEditingNotes]=useState(false);
 const [sourceReady,setSourceReady]=useState(false),[sourceError,setSourceError]=useState(false);
 const anchor=useRef<[number,number]|null>(null),inFlight=useRef(false),mounted=useRef(true);
 const callbacks=useRef({busyChanged,savedLabel});callbacks.current={busyChanged,savedLabel};
 useEffect(()=>{mounted.current=true;return()=>{mounted.current=false;callbacks.current.busyChanged(false)}},[]);
 const validBox=label.bbox&&label.bbox.every(Number.isFinite)&&label.bbox[0]>=0&&label.bbox[1]>=0&&label.bbox[2]>0&&label.bbox[3]>0&&label.bbox[0]+label.bbox[2]<=data.source.width&&label.bbox[1]+label.bbox[3]<=data.source.height;
 const reviewed=task.proposals.filter(p=>label.reviewed_proposals.includes(p.proposal_id)).length;
 const complete=sourceReady&&decisionMade&&reviewed===task.proposals.length&&(label.state!=='visible'||Boolean(validBox));
 const edit=(value:Partial<Choice>,decision=decisionMade)=>{if(inFlight.current)return;setLabel(previous=>({...previous,...value}));setDecisionMade(decision);setDirty(true);setSaveError('');clearMessage()};
 useEffect(()=>{if(dirty)setDraftStored(store(draftKey,JSON.stringify({label,decisionMade})))},[label,decisionMade,dirty,draftKey]);
 useEffect(()=>{
  if(!dirty||!complete||saveError||editingNotes||reviewer.trim().length<2)return;
  const timer=setTimeout(async()=>{
   if(inFlight.current)return;inFlight.current=true;setSaving(true);callbacks.current.busyChanged(true);
   const {reviewed_proposals,...choice}=label;
   const body={...choice,...(data.supports_reviewed_proposals?{reviewed_proposals}:{}),task_id:task.task_id,reviewer_id:reviewer.trim(),reviewed_from_source:true,origin:'human',model_assisted:true};
   const controller=new AbortController();const timeout=setTimeout(()=>controller.abort(),20000);
   try{
    const response=await fetch(base+'/labels',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body),signal:controller.signal});const result=await response.json();
    if(!response.ok||result.saved!==true||result.task_id!==task.task_id)throw Error(typeof result.detail==='string'?result.detail:'The server did not confirm your save. Your choices are still here.');
    store(draftKey,null);
    if(mounted.current){setDirty(false);callbacks.current.savedLabel({...label,task_id:task.task_id,reviewer_id:reviewer.trim()})}
   }catch(e){if(mounted.current)setSaveError(controller.signal.aborted?'Saving timed out. Your choices are kept here; retry when the service is available.':e instanceof Error?e.message:'Could not save. Your choices are still here.')}finally{clearTimeout(timeout);inFlight.current=false;if(mounted.current){setSaving(false);callbacks.current.busyChanged(false)}}
  },600);
  return()=>clearTimeout(timer);
 },[label,dirty,complete,saveError,editingNotes,reviewer,base,task.task_id,data.supports_reviewed_proposals,draftKey]);
 const accept=(proposal:Proposal)=>{
  const rejected=label.rejected_proposals.filter(id=>{const p=task.proposals.find(p=>p.proposal_id===id);return p&&overlap(proposal.bbox,p.bbox)<=.1});
  const reviewedIds=label.reviewed_proposals.filter(id=>{const p=task.proposals.find(p=>p.proposal_id===id);return rejected.includes(id)||(p&&overlap(proposal.bbox,p.bbox)>.1)});
  edit({state:'visible',bbox:proposal.bbox,accepted_proposal:proposal.proposal_id,rejected_proposals:rejected,reviewed_proposals:[...new Set([...reviewedIds,proposal.proposal_id])]},true);
 };
 const reject=(proposal:Proposal)=>{
  const id=proposal.proposal_id,undo=label.rejected_proposals.includes(id),wasAccepted=label.accepted_proposal===id;
  const rejected=undo?label.rejected_proposals.filter(r=>r!==id):[...label.rejected_proposals,id];
  edit({rejected_proposals:rejected,reviewed_proposals:wasAccepted?rejected:undo?label.reviewed_proposals.filter(r=>r!==id):[...new Set([...label.reviewed_proposals,id])],...(wasAccepted?{accepted_proposal:null,bbox:null,state:'uncertain' as const}:{})},wasAccepted?false:decisionMade);
 };
 const visibility=(state:'not_visible'|'uncertain')=>edit({state,bbox:null,accepted_proposal:null,reviewed_proposals:task.proposals.map(p=>p.proposal_id),rejected_proposals:state==='not_visible'?task.proposals.map(p=>p.proposal_id):label.rejected_proposals},true);
 const point=(e:React.PointerEvent<SVGSVGElement>):[number,number]=>{const r=e.currentTarget.getBoundingClientRect();return [Math.max(0,Math.min(data.source.width,(e.clientX-r.left)/r.width*data.source.width)),Math.max(0,Math.min(data.source.height,(e.clientY-r.top)/r.height*data.source.height))]};
 const draw=(bbox:Box)=>{
  const rejected=label.rejected_proposals.filter(id=>{const p=task.proposals.find(p=>p.proposal_id===id);return p&&overlap(bbox,p.bbox)<=.1});
  const reviewedIds=label.reviewed_proposals.filter(id=>{const p=task.proposals.find(p=>p.proposal_id===id);return rejected.includes(id)||(p&&overlap(bbox,p.bbox)>.1)});
  edit({state:'visible',bbox,accepted_proposal:null,rejected_proposals:rejected,reviewed_proposals:reviewedIds},true);
 };
 return <section className="card" aria-label={`${reviewKey.toUpperCase()} ball annotation`}>
  <div className="section-bar"><div><h2>Frame {task.frame} · {task.timestamp.toFixed(3)}s</h2><p>Question {index+1} of {data.tasks.length} · {reviewed}/{task.proposals.length} proposals reviewed</p></div><label className="m22-zoom">Zoom<input aria-label="Frame zoom" type="range" min="1" max="4" step=".5" value={zoom} onChange={e=>setZoom(Number(e.target.value))}/></label></div>
  {reviewKey==='m25'&&<fieldset><legend>Show model overlays</legend>{[...new Set(task.proposals.map(p=>p.source_pipeline))].map(model=><label key={model}><input type="checkbox" checked={!hiddenModels.includes(model)} onChange={e=>setHiddenModels(v=>e.target.checked?v.filter(x=>x!==model):[...v,model])}/>{model}</label>)}</fieldset>}
  <div className="m22-review-workspace"><div><div className="m22-image-scroll"><svg role="img" aria-label="Current source frame: verify the football" viewBox={`0 0 ${data.source.width} ${data.source.height}`} style={{width:`${zoom*100}%`,display:'block',touchAction:'none',cursor:'crosshair'}} onPointerDown={e=>{if(saving||!sourceReady)return;anchor.current=point(e);e.currentTarget.setPointerCapture?.(e.pointerId)}} onPointerCancel={()=>{anchor.current=null}} onPointerUp={e=>{if(saving||!sourceReady||!anchor.current)return;const [x,y]=anchor.current,[a,b]=point(e);anchor.current=null;if(centerMode){const w=label.bbox?.[2]??8,h=label.bbox?.[3]??8;draw([Math.max(0,Math.min(data.source.width-w,a-w/2)),Math.max(0,Math.min(data.source.height-h,b-h/2)),w,h])}else if(Math.abs(a-x)>0&&Math.abs(b-y)>0)draw([Math.min(x,a),Math.min(y,b),Math.abs(a-x),Math.abs(b-y)])}}>
   <image href={base+`/frame.jpg?frame=${task.frame}`} width={data.source.width} height={data.source.height} onLoad={()=>{setSourceReady(true);setSourceError(false)}} onError={()=>{setSourceReady(false);setSourceError(true)}}/>
   {task.proposals.map((p,i)=><g key={p.proposal_id} display={hiddenModels.includes(p.source_pipeline)?'none':undefined} opacity={label.rejected_proposals.includes(p.proposal_id)?.25:.9}><rect x={p.bbox[0]} y={p.bbox[1]} width={p.bbox[2]} height={p.bbox[3]} fill="none" stroke="#61ccff" strokeWidth={Math.max(1,data.source.width/500)} strokeDasharray="5 3"/><text x={p.bbox[0]} y={Math.max(14,p.bbox[1]-4)} fill="#61ccff" stroke="#101820" strokeWidth={.6} paintOrder="stroke" fontSize={Math.max(13,data.source.width/55)}>{i+1}</text></g>)}
   {label.bbox&&<rect x={label.bbox[0]} y={label.bbox[1]} width={label.bbox[2]} height={label.bbox[3]} fill="none" stroke="#ffdf00" strokeWidth={Math.max(2,data.source.width/400)}/>}</svg></div><p className="panel-help">Blue numbered boxes are suggestions. Yellow is your chosen ball box. Drag on the image to correct it.</p></div>
  <div className="m22-proposal-list"><h3>Which boxes show the ball?</h3>{!task.proposals.length&&<p>No proposals here. Draw the ball's box, or choose an option below.</p>}{task.proposals.map((p,i)=>{
   const rejected=label.rejected_proposals.includes(p.proposal_id),accepted=label.accepted_proposal===p.proposal_id,done=label.reviewed_proposals.includes(p.proposal_id);
   const overlapsChosen=label.bbox&&overlap(label.bbox,p.bbox)>.1&&!accepted;
   const cropWidth=Math.min(data.source.width,Math.max(32,p.bbox[2]*4)),cropHeight=Math.min(data.source.height,Math.max(32,p.bbox[3]*4));
   const cropX=Math.max(0,Math.min(data.source.width-cropWidth,p.bbox[0]+p.bbox[2]/2-cropWidth/2)),cropY=Math.max(0,Math.min(data.source.height-cropHeight,p.bbox[1]+p.bbox[3]/2-cropHeight/2));
   return <div key={p.proposal_id} className={`m22-proposal ${done?'is-reviewed':''}`}><div className="section-bar"><strong>Proposal {i+1} · {p.source_pipeline}</strong><span>{rejected?'Not the ball':accepted?'Accepted':done?label.state==='visible'?'Same ball · reviewed':'Reviewed · uncertain':'Needs review'}</span></div><div className="m22-proposal-choice"><svg role="img" aria-label={`Zoomed proposal ${i+1}`} viewBox={`${cropX} ${cropY} ${cropWidth} ${cropHeight}`}><image href={base+`/frame.jpg?frame=${task.frame}`} width={data.source.width} height={data.source.height}/><rect x={p.bbox[0]} y={p.bbox[1]} width={p.bbox[2]} height={p.bbox[3]} fill="none" stroke="#61ccff" strokeWidth={cropWidth/64}/></svg><div className="m22-proposal-buttons"><button className={accepted?'primary':'secondary'} disabled={saving||!sourceReady} aria-pressed={accepted||Boolean(done&&!rejected&&label.state==='visible')} onClick={()=>accept(p)}>Accept proposal {i+1}</button><button disabled={saving||!sourceReady||Boolean(overlapsChosen)} aria-pressed={rejected} onClick={()=>reject(p)}>{rejected?'Undo rejection':'Not the ball'} {i+1}</button></div></div>{overlapsChosen&&!done&&<small>This overlaps your chosen ball. Accept it too if it shows the same ball.</small>}</div>;
  })}<div className="m22-visibility"><p>Can't find the ball?</p><button disabled={saving||!sourceReady} onClick={()=>visibility('not_visible')}>Ball not visible</button><button disabled={saving||!sourceReady} onClick={()=>visibility('uncertain')}>Can't tell</button></div></div></div>
  {sourceError&&<p role="alert">The source image could not load. Reload before reviewing this frame; nothing has been saved.</p>}
  <div className="m22-auto-status" role="status" aria-live="polite">{saving?'Saving your review…':saveError?'Not saved — your choices are kept here.':dirty?(complete?'Review complete — saving and moving on…':draftStored?'Draft kept in this browser. Review the remaining proposals.':'Review the remaining proposals.'):old?'Saved review. Change any choice to correct it.':'Choose each proposal, or use a visibility option. No Save button needed.'}</div>
  {dirty&&reviewed===task.proposals.length&&!decisionMade&&<p>All proposals rejected. Draw the ball elsewhere, choose “Ball not visible”, or choose “Can't tell”.</p>}
  {saveError&&<div className="review-save-error" role="alert"><p>{saveError}</p><button onClick={()=>setSaveError('')}>Retry automatic save</button></div>}
  <div className="review-actions m22-frame-navigation"><button disabled={saving||index===0} onClick={()=>navigate(index-1)}>Previous frame</button><span>{index+1}/{data.tasks.length}</span><button disabled={saving||index===data.tasks.length-1} onClick={()=>navigate(index+1)}>Next frame</button></div>
  <details className="m22-review-help"><summary>More context & optional notes</summary>{reviewKey==='m23'&&data.source.recording_id&&<div><p>Use the original video to locate the ball, then return to the still frame above.</p><video aria-label="Source video context" controls preload="none" src={'/api/matches/'+encodeURIComponent(data.source.recording_id)+'/video.mp4'} onLoadedMetadata={e=>{e.currentTarget.currentTime=Math.max(0,task.timestamp-.6)}} style={{width:'100%',maxHeight:400}}/></div>}<div className="m22-context">{(reviewKey==='m25'?[]:[-2,-1]).map(delta=><figure key={delta}><img loading="lazy" src={base+`/frame.jpg?frame=${Math.max(0,task.frame+delta)}`} alt={`Source context frame ${Math.max(0,task.frame+delta)}`}/><figcaption>Previous frame {Math.max(0,task.frame+delta)}</figcaption></figure>)}</div><label className="review-check"><input type="checkbox" checked={centerMode} disabled={saving} onChange={e=>setCenterMode(e.target.checked)}/>Move box center with a click</label><label className="review-field">Notes<textarea aria-label="Notes" maxLength={500} value={label.notes} disabled={saving} onFocus={()=>setEditingNotes(true)} onBlur={()=>setEditingNotes(false)} onChange={e=>edit({notes:e.target.value})}/></label><p>{task.selection_reason}. Source sequence: {task.sequence_frames.join(', ')}. Model suggestions are not ground truth.</p></details>
 </section>;
}
