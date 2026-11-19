import {useEffect,useRef,useState} from 'react';
import {mediaUrl} from './api';
import {reviewProblems,reviewServerError,sourceReviewFrame} from './cvReviewValidation';
import type {ReviewBox as Box,ReviewTask as Task,ReviewSource,ReferencePass as Pass,ReviewLabel} from './cvReviewValidation';
type Dataset={dataset_fingerprint:string,source:ReviewSource,tasks:Task[],labels:unknown[]};

function BoxFrame({url,width,height,box,onBox,label,disabled=false,magnify=false}:{url:string,width:number,height:number,box:Box|null,onBox:(box:Box|null)=>void,label:string,disabled?:boolean,magnify?:boolean}){
 const [zoom,setZoom]=useState(1);
 const anchor=useRef<[number,number]|null>(null);
 const point=(e:React.PointerEvent<SVGSVGElement>):[number,number]=>{
  const r=e.currentTarget.getBoundingClientRect();
  return [Math.max(0,Math.min(width,(e.clientX-r.left)/r.width*width)),Math.max(0,Math.min(height,(e.clientY-r.top)/r.height*height))];
 };
 return <figure>
  <figcaption>{label} · drag to draw a box</figcaption>
  {magnify&&<label className="review-field">Frame zoom <input aria-label={`Zoom: ${label}`} type="range" min="1" max="4" step=".5" value={zoom} onChange={e=>setZoom(Number(e.target.value))}/><span>{zoom}× · scroll to inspect the ball, then draw a tight box.</span></label>}
  <div style={{overflow:'auto',maxHeight:magnify?650:undefined}}>
  <svg role="img" aria-label={label} aria-disabled={disabled} viewBox={`0 0 ${width} ${height}`}
   style={{width:`${zoom*100}%`,height:'auto',display:'block',maxWidth:magnify?undefined:900,touchAction:'none',cursor:disabled?'wait':'crosshair'}}
   onPointerDown={e=>{if(disabled)return;anchor.current=point(e);e.currentTarget.setPointerCapture?.(e.pointerId)}}
   onPointerCancel={()=>{anchor.current=null}}
   onPointerUp={e=>{if(disabled)return;if(anchor.current){const [x,y]=anchor.current,[a,b]=point(e);onBox([Math.min(x,a),Math.min(y,b),Math.abs(a-x),Math.abs(b-y)]);anchor.current=null}}}>
   <image href={url} width={width} height={height}/>
   {box&&<rect x={box[0]} y={box[1]} width={box[2]} height={box[3]} fill="none" stroke="#ffdf00" strokeWidth={Math.max(2,width/400)}/>}
  </svg>
  </div>
  <button type="button" className="secondary" disabled={disabled||!box} onClick={()=>onBox(null)} aria-label={`Clear box: ${label}`}>Clear box</button>
 </figure>;
}

export default function CVHumanReview({recordingId,apiRoot='/api',reviewKey='m19',sourceRecordings=['test_match','team_goal','southampton']}:{recordingId:string,apiRoot?:string,reviewKey?:'m19'|'m21',sourceRecordings?:string[]}){
 const [data,setData]=useState<Dataset|null>(null),[index,setIndex]=useState(0),[reviewer,setReviewer]=useState(''),[state,setState]=useState('uncertain'),[box,setBox]=useState<Box|null>(null),[other,setOther]=useState<Box|null>(null),[passes,setPasses]=useState<Pass[]>([]),[release,setRelease]=useState<number|null>(null),[exhaustive,setExhaustive]=useState(false),[attest,setAttest]=useState(false),[notes,setNotes]=useState(''),[message,setMessage]=useState(''),[busy,setBusy]=useState(false);
 const video=useRef<HTMLVideoElement>(null);const base=apiRoot+'/matches/'+encodeURIComponent(recordingId);
 useEffect(()=>{let active=true;fetch(base+'/cv/review/tasks').then(async r=>{if(!r.ok)throw Error('Review tasks unavailable');return r.json()}).then(d=>{if(active)setData(d)}).catch(e=>{if(active)setMessage(e.message)});return()=>{active=false}},[base]);
 useEffect(()=>{setState('uncertain');setBox(null);setOther(null);setPasses([]);setRelease(null);setExhaustive(false);setAttest(false);setNotes('');setMessage('')},[index]);
 const task=data?.tasks[index];
 const label:ReviewLabel|undefined=task?{
  task_id:task.task_id,reviewer_id:reviewer.trim(),origin:'human',reviewed_from_source:attest,model_assisted:false,state,
  bbox:task.kind==='identity'||task.kind==='ball'&&state==='visible'?box:null,
  other_bbox:task.kind==='identity'?other:null,exhaustive:task.kind==='passes'&&exhaustive,passes:task.kind==='passes'?passes:[],notes
 }:undefined;
 const problems=task&&data&&label?reviewProblems(task,data.source,label,release):['Wait for the review task to load.'];
 const messageElement=message?<p className={message.startsWith('Saved human reference.')?'review-save-success':'review-save-error'} role={message.startsWith('Saved human reference.')?'status':'alert'}>{message}</p>:null;

 async function save(){
  if(busy||!task||!data||!label)return;
  if(problems.length){setMessage(problems.join(' '));return}
  setBusy(true);setMessage('');
  try{
   const response=await fetch(base+'/cv/review/labels',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(label)});
   const body=await response.json().catch(()=>null);
   if(!response.ok)throw Error(reviewServerError(body,response.status));
   if(body?.saved!==true)throw Error('The server did not confirm this save. Your edits are still here; try again.');
   setMessage('Saved human reference. This does not change model results.');
  }catch(e){setMessage(e instanceof Error?e.message:'Could not save. Your edits are still here.')}
  finally{setBusy(false)}
 }
 function currentFrame(){
  const player=video.current;
  if(!data||!player||player.readyState<2||player.seeking)return null;
  return sourceReviewFrame(player.currentTime,data.source.fps,data.source.frames);
 }
 function markRelease(){
  if(!task||busy)return;
  const frame=currentFrame();
  if(frame===null){setMessage('Wait for the video to finish loading or seeking, then mark the release.');return}
  if(task.start_frame===undefined||task.end_frame===undefined||frame<task.start_frame||frame>=task.end_frame){setMessage(`Choose a release inside frames ${task.start_frame}–${(task.end_frame??0)-1}, leaving room for a later receipt.`);return}
  setRelease(frame);setMessage('');
 }
 function markReceipt(){
  if(!task||release===null||busy)return;
  const frame=currentFrame();
  if(frame===null){setMessage('Wait for the video to finish loading or seeking, then mark the receipt.');return}
  if(frame<=release){setMessage(`Move the video past release frame ${release}, then mark the receipt.`);return}
  if(task.end_frame===undefined||frame>task.end_frame){setMessage(`The receipt must be inside this review window, ending at frame ${task.end_frame}.`);return}
  if(passes.length>=30){setMessage('This window already has 30 pass references. Remove a reference before adding another.');return}
  setPasses(rows=>[...rows,{release_frame:release,receipt_frame:frame,outcome:'completed',sender_description:'',receiver_description:''}]);setRelease(null);setMessage('');
 }

 return <main className="app-shell human-review">
  <header className="section-bar"><h1>{reviewKey==='m21'?'M21 sealed source review':'M19 independent source review'}</h1><a href={'/?recording='+encodeURIComponent(reviewKey==='m21'?'test_match':recordingId)}>Return to analysis</a></header>
  <section className="card">
   <p>Review original footage without model overlays or candidate suggestions. Mark uncertainty honestly. Labels are stored separately from predictions and confirmed player identities.</p>
   <p>{reviewKey==='m21'?'New source-disjoint evaluation: tightly box the in-play ball. Consecutive frames support temporal evaluation. Include clearly different-player pairs where instructed; mark uncertainty when needed. No model has suggested these labels.':'60 sealed tasks: ball visibility, player continuity and complete pass/non-pass windows across three clips. Selection uses uniform source times.'}</p>
   <nav className="review-recordings" aria-label="Review recordings">{sourceRecordings.map(name=><a key={name} href={'/?recording='+name+'&review='+reviewKey}>{name}</a>)}</nav>
   <div className="review-setup"><label className="review-field">Reviewer ID <input aria-label="Reviewer ID" maxLength={80} disabled={busy} value={reviewer} onChange={e=>{setReviewer(e.target.value);setMessage('')}} placeholder="Your initials or review ID"/></label><a href={apiRoot+'/cv/review/export'} download>Export human labels</a></div>
  </section>
  {data&&task&&<section className="card" aria-label="Independent review task" aria-busy={busy}>
   <div className="section-bar"><h2>{recordingId} · {task.kind} · task {index+1}/{data.tasks.length}</h2><select aria-label="Review task" disabled={busy} value={index} onChange={e=>setIndex(Number(e.target.value))}>{data.tasks.map((t,i)=><option key={t.task_id} value={i}>{i+1} · {t.kind}</option>)}</select></div>
   {task.kind==='passes'?<>
    <p>Review every action from {task.start?.toFixed(2)} to {task.end?.toFixed(2)} seconds. Include all completed passes whose release and receipt fall inside this window. An exhaustively reviewed window with no passes is a valid negative reference.</p>
    <video key={task.task_id} ref={video} src={mediaUrl(`${apiRoot.replace(/^\/api/,'')}/matches/${recordingId}/video.mp4`)} controls style={{width:'100%',maxHeight:560}} onLoadedMetadata={e=>{e.currentTarget.currentTime=task.start??0}}/>
    <div className="review-actions"><button type="button" disabled={busy} onClick={()=>{if(video.current)video.current.currentTime=task.start??0}}>Replay window</button><button type="button" disabled={busy||passes.length>=30} onClick={markRelease}>Mark release frame</button><button type="button" disabled={busy||release===null||passes.length>=30} onClick={markReceipt}>Mark receipt and add pass</button></div>
    {release!==null&&<div className="review-actions"><p>Release marked at frame {release}.</p><button type="button" disabled={busy} onClick={()=>{setRelease(null);setMessage('')}}>Cancel release mark</button></div>}
    {passes.map((pass,i)=><p key={i}>Pass: frame {pass.release_frame} → {pass.receipt_frame} <button type="button" disabled={busy} onClick={()=>{setPasses(rows=>rows.filter((_,j)=>i!==j));setMessage('')}}>Remove</button></p>)}
    <label className="review-check"><input type="checkbox" disabled={busy} checked={exhaustive} onChange={e=>{setExhaustive(e.target.checked);setMessage('')}}/> I reviewed every action in the complete window</label>
   </>:<>
    {reviewKey==='m21'&&task.kind==='identity'&&<p className="panel-help">{data.tasks.filter(t=>t.kind==='identity').findIndex(t=>t.task_id===task.task_id)%2===1?'For this pair, choose one player in the first frame and a clearly different player in the second frame. Select “different” only if you can verify they are different people; otherwise mark uncertain.':'For this pair, choose any recognizable player in the first frame and find that same person in the second frame. Select “same” only if you can verify the match; otherwise mark uncertain.'} Each pair is independent; you can choose another player on the next task.</p>}
    <BoxFrame magnify={reviewKey==='m21'} label={task.kind==='ball'?'Source frame: box the visible ball':'First frame: box one player'} url={base+'/cv/review/frame.jpg?frame='+task.frame} width={data.source.width} height={data.source.height} box={box} onBox={value=>{setBox(value);setMessage('')}} disabled={busy}/>
    {task.kind==='identity'&&<BoxFrame magnify={reviewKey==='m21'} label="Second frame: box the player being compared" url={base+'/cv/review/frame.jpg?frame='+task.other_frame} width={data.source.width} height={data.source.height} box={other} onBox={value=>{setOther(value);setMessage('')}} disabled={busy}/>}
   </>}
   <div className="review-fields"><label className="review-field">Review result <select aria-label="Review result" disabled={busy} value={state} onChange={e=>{setState(e.target.value);setMessage('')}}>{(task.kind==='ball'?['uncertain','visible','not_visible']:task.kind==='identity'?['uncertain','same','different']:['uncertain','reviewed']).map(s=><option key={s} value={s}>{s.replace('_',' ')}</option>)}</select></label><label className="review-field review-notes">Notes <textarea aria-label="Notes" disabled={busy} value={notes} maxLength={500} onChange={e=>{setNotes(e.target.value);setMessage('')}}/></label></div>
   <label className="review-check"><input type="checkbox" disabled={busy} checked={attest} onChange={e=>{setAttest(e.target.checked);setMessage('')}}/> I personally reviewed the source without model outputs or suggestions</label>
   {problems.length>0&&<div className="review-save-requirements" id="review-save-requirements" role="note" aria-label="Before saving"><strong>To save this task:</strong><ul>{problems.map(problem=><li key={problem}>{problem}</li>)}</ul></div>}
   {messageElement}
   <p className="review-save-help">Save each task before choosing Next. Nothing is sent automatically while you edit.</p>
   <div className="review-actions review-navigation"><button type="button" disabled={busy||index===0} onClick={()=>setIndex(i=>i-1)}>Previous</button><button type="button" className="primary" disabled={busy||problems.length>0} aria-describedby={problems.length?'review-save-requirements':undefined} onClick={()=>void save()}>{busy?'Saving…':'Save human reference'}</button><button type="button" disabled={busy||index===data.tasks.length-1} onClick={()=>setIndex(i=>i+1)}>Next</button></div>
  </section>}
  {!task&&messageElement}
 </main>;
}
