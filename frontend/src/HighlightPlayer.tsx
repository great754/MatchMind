import {forwardRef,useEffect,useImperativeHandle,useRef,useState} from 'react';
import {z} from 'zod';
import {request,mediaUrl} from './api';
import {preciseTime} from './time';
const event=z.object({event_id:z.string(),event_type:z.string(),start:z.number(),end:z.number(),timestamp:z.number(),verification:z.string(),player:z.number().nullable(),safe_for_automatic:z.boolean(),metrics:z.record(z.string(),z.number()),evidence_source:z.string()});
const segment=z.object({start:z.number(),end:z.number(),output_start:z.number(),output_end:z.number(),event_ids:z.array(z.string()),reason:z.string()});
const jobSchema=z.object({job_id:z.string(),recording:z.string(),state:z.enum(['QUEUED','PROCESSING','COMPLETED','FAILED','CANCELLED']),message:z.string(),events:z.array(event),video_url:z.string().nullable(),thumbnail_url:z.string().nullable(),provider:z.string(),provider_diagnostic:z.string().nullable().optional(),download_permitted:z.boolean(),cache_hit:z.boolean(),manifest:z.object({segments:z.array(segment),render:z.object({duration:z.number(),size_bytes:z.number()})}).passthrough().nullable()});
type Job=z.infer<typeof jobSchema>;
export type HighlightHandle={ask:(q:string)=>void,isFollowup:(q:string)=>boolean};
export const isHighlightRequest=(q:string)=>/\b(highlights?|recap|longest pass|farthest pass|furthest pass|fastest shot|buildup|build-up)\b/i.test(q)||/\b(show|video|clips?|compilation)\b/i.test(q)&&/\b(goals?|shots?|passes|saves?|tackles?|crosses|scored|player)\b/i.test(q);
const HighlightPlayer=forwardRef<HighlightHandle,{recordingId:string,onOpen:(time:number)=>void}>(function HighlightPlayer({recordingId,onOpen},ref){
 const [text,setText]=useState(''),[job,setJob]=useState<Job|null>(null),[error,setError]=useState(''),[submitting,setSubmitting]=useState(false),[playing,setPlaying]=useState(false);
 const container=useRef<HTMLElement>(null);
 const session=useRef(crypto.randomUUID()),previous=useRef<string|null>(null),version=useRef(0),video=useRef<HTMLVideoElement>(null);
 const base='/api/matches/'+encodeURIComponent(recordingId)+'/highlights';
 const active=job?.state==='QUEUED'||job?.state==='PROCESSING';
 useEffect(()=>()=>{version.current++},[]);
 async function ask(q:string){
  if(!q.trim()||submitting||active)return;container.current?.scrollIntoView?.({behavior:'smooth',block:'start'});setText(q);setError('');setSubmitting(true);const current=++version.current;
  try{const data=await request(base,jobSchema,{method:'POST',body:JSON.stringify({question:q,session_id:session.current,previous_job:previous.current,provider:'auto'})},45000);if(version.current===current){setJob(data);if(data.state==='COMPLETED')previous.current=data.job_id}}
  catch(e){if(version.current===current)setError(e instanceof Error?e.message:'Could not create highlights')}
  finally{if(version.current===current)setSubmitting(false)}
 }
 useImperativeHandle(ref,()=>({ask:q=>void ask(q),isFollowup:q=>Boolean(previous.current)&&/\b(second one|first one|third one|who made that pass|more buildup|shorter clips|longer clips)\b/i.test(q)}));
 useEffect(()=>{
  if(!job||!active)return;let stopped=false;let timer:ReturnType<typeof setTimeout>;
  const poll=async()=>{try{const data=await request(base+'/'+job.job_id+'?session_id='+session.current,jobSchema);if(stopped)return;setJob(data);if(data.state==='COMPLETED')previous.current=data.job_id;if(data.state==='QUEUED'||data.state==='PROCESSING')timer=setTimeout(poll,700)}catch(e){if(!stopped){setError(e instanceof Error?e.message:'Status unavailable');timer=setTimeout(poll,2500)}}};timer=setTimeout(poll,500);return()=>{stopped=true;clearTimeout(timer)};
 },[job?.job_id,active,base]);
 async function cancel(){if(!job)return;try{await request(base+'/'+job.job_id+'?session_id='+session.current,jobSchema,{method:'DELETE'})}catch(e){setError(e instanceof Error?e.message:'Cancellation unavailable')}}
 const title=job?.events.length===1&&job.events[0].event_type==='pass'?'Recorded pass':job?.events.every(e=>e.event_type==='goal')?'Goal highlights':'Match highlights';
 return <section ref={container} className="card highlight-creator" aria-labelledby={'highlight-heading-'+recordingId}>
 <div className="section-bar"><div><h2 id={'highlight-heading-'+recordingId}>AI Highlight Creator</h2><p>Ask for a video of the moments you want to watch.</p></div><span className="source-pill">TactiqAI</span></div>
 <form onSubmit={e=>{e.preventDefault();void ask(text)}} className="question-form"><label htmlFor={'highlight-input-'+recordingId} className="sr-only">Ask for highlights</label><input id={'highlight-input-'+recordingId} value={text} maxLength={1000} onChange={e=>setText(e.target.value)} placeholder="Show me all the goals"/><button className="primary" disabled={submitting||active||!text.trim()}>{submitting?'Finding moments…':'Create highlight video'}</button></form>
 <div className="highlight-suggestions">{['Show me all the goals','What was the longest pass?','Show me all the shots','Make a one-minute highlight video'].map(q=><button className="secondary" disabled={active||submitting} key={q} onClick={()=>void ask(q)}>{q}</button>)}</div>
 {error&&<p role="alert">{error}</p>}
 {(submitting||active)&&<div role="status"><p>{job?.state==='PROCESSING'?'Creating your video…':'Finding your recorded moments…'}</p>{active&&<button className="secondary" onClick={()=>void cancel()}>Cancel highlight</button>}</div>}
 {job&&!active&&<div aria-live="polite"><p data-testid="highlight-answer">{job.message}</p>{job.provider_diagnostic&&<p className="section-footnote">{job.provider_diagnostic}</p>}
 {job.video_url&&job.manifest&&<><h3>{title}</h3><video className="highlight-video" data-testid="highlight-video" ref={video} controls preload="metadata" src={mediaUrl(job.video_url)} poster={job.thumbnail_url?mediaUrl(job.thumbnail_url):undefined} onPlay={()=>setPlaying(true)} onPause={()=>setPlaying(false)} onError={()=>setError('The highlight video could not be played. Try creating it again.')}/>
 <div className="highlight-controls"><button className="primary" onClick={()=>{if(video.current)void(playing?video.current.pause():video.current.play())}}>{playing?'Pause highlights':'Play all highlights'}</button><button className="secondary" onClick={()=>void ask('More buildup')}>More buildup</button><button className="secondary" onClick={()=>void ask('Shorter clips')}>Shorter clips</button>{job.download_permitted&&<a className="secondary" href={mediaUrl(job.video_url)+'&download=true'} download>Download video</a>}</div>
 <ol>{job.events.map(e=>{const s=job.manifest!.segments.find(s=>s.event_ids.includes(e.event_id));return <li key={e.event_id}><strong>{e.event_type.replaceAll('_',' ')} · {preciseTime(e.start)}</strong> <span>{e.verification==='annotated'?'Annotation-supported':e.verification==='candidate'?'Unconfirmed review':'Tracking estimate'}</span><button className="secondary" onClick={()=>{if(video.current&&s){video.current.currentTime=s.output_start;void video.current.play()}}}>Play this moment</button><button className="secondary" onClick={()=>onOpen(s?.start??e.start)}>Open original footage</button><details><summary>Why this moment?</summary><p>{s?.reason}</p><p>Source: {e.evidence_source} · Event: {e.event_id}</p>{e.metrics.distance_m!==undefined&&<p>Recorded endpoint distance: {e.metrics.distance_m.toFixed(1)} m (tracking estimate)</p>}</details></li>})}</ol>
 <p className="section-footnote">Clip times refer to this recording. Buildup uses context windows, not a verified attacking sequence. Short recaps prioritize recorded goals, shots and other supported moments; “exciting” is not an objective score. {!job.download_permitted&&'Downloads are not enabled for these media rights.'}</p></>}
 </div>}
 </section>;
});
export default HighlightPlayer;
