import {lazy,Suspense,useEffect,useState} from 'react';
import App from './App';
import {recordingList} from './api';
import VideoOnlyRecording from './VideoOnlyRecording';
import CVReview from './CVReview';
const CVHumanReview=lazy(()=>import('./CVHumanReview'));
type Recording={recording_id:string,match_id:string,status:'available'|'not_processed'|'unavailable'|'error',media_available?:boolean,media_url?:string|null,source?:string,capabilities?:{cv_world_model_available:boolean,analytics_available:boolean,ask_available:boolean}};
const selectable=(r:Recording)=>r.status==='available'||Boolean(r.media_available&&r.media_url);
export default function RecordingsApp(){
 const [items,setItems]=useState<Recording[]>([]);
 const [loading,setLoading]=useState(true),[catalogRevision,setCatalogRevision]=useState(0);
 const [selected,setSelected]=useState(new URLSearchParams(window.location.search).get('recording')||'118575_clip'),[error,setError]=useState(''),[reviewMode,setReviewMode]=useState('analytics');
 useEffect(()=>{let alive=true;setLoading(true);setError('');recordingList().then(data=>{if(!alive)return;setItems(data.recordings);if(!data.recordings.some(r=>r.recording_id===selected&&selectable(r))){const first=data.recordings.find(selectable);if(first)setSelected(first.recording_id)}}).catch(()=>{if(alive)setError('Recording list unavailable; showing the default recording.')}).finally(()=>{if(alive)setLoading(false)});return()=>{alive=false}},[catalogRevision]);
 useEffect(()=>{const open=(event:Event)=>{const d=(event as CustomEvent).detail;if(items.some(r=>r.recording_id===d?.recording_id&&selectable(r))&&typeof d.timestamp==='number'&&Number.isFinite(d.timestamp)&&d.timestamp>=0){sessionStorage.setItem('tactiqai:pending-source-seek',JSON.stringify(d));setSelected(d.recording_id);setReviewMode('analytics')}};window.addEventListener('tactiqai-open-recording',open);return()=>window.removeEventListener('tactiqai-open-recording',open)},[items]);
 const options:Recording[]=items.length?items:[{recording_id:'118575_clip',match_id:'118575',status:'available'}];
 const current=items.find(r=>r.recording_id===selected);
 const picker=<><label className="select-label"><span className="sr-only">Recording</span><select aria-label="Recording" value={selected} onChange={e=>{setSelected(e.target.value);setReviewMode('analytics')}} title={error||'Choose a match recording'}>{options.map(r=><option key={r.recording_id} value={r.recording_id} disabled={!selectable(r)}>{r.recording_id}{r.source==='cv_experimental'?' · CV Experimental':''}{r.status==='available'?'':r.media_available?' · video ready, analysis pending':r.status==='not_processed'?' · awaiting processing':r.status==='unavailable'?' · artifacts unavailable':' · artifact error'}</option>)}</select></label>{current?.capabilities?.cv_world_model_available&&current.capabilities.analytics_available&&<select aria-label="Review mode" value={reviewMode} onChange={e=>setReviewMode(e.target.value)}><option value="analytics">Match Intelligence</option><option value="cv">CV Experimental</option></select>}</>;
 if(new URLSearchParams(window.location.search).get('review')==='m19')return <Suspense fallback={<main className="app-shell"><p role="status">Loading source review…</p></main>}><CVHumanReview key={selected} recordingId={selected}/></Suspense>;
 if(loading)return <main className="app-shell"><section className="card profile-empty" role="status"><h1>Loading recordings…</h1><p>Checking the available video and analysis for this recording.</p></section></main>;
 if(!current&&selected!=='118575_clip')return <main className="app-shell"><section className="card profile-empty" role="alert"><h1>Recording list unavailable</h1><p>Reload the recording list to open the correct video workspace.</p><button className="secondary" onClick={()=>setCatalogRevision(value=>value+1)}>Retry loading recordings</button></section></main>;
 if(current?.capabilities?.cv_world_model_available&&(!current.capabilities.analytics_available||reviewMode==='cv'))return <CVReview key={selected} recordingId={selected} recordingPicker={picker} analyticsAvailable={current.capabilities.analytics_available}/>;
 if(current&&current.status!=='available'&&current.media_available&&current.media_url)return <VideoOnlyRecording key={selected} recordingId={selected} mediaPath={current.media_url} recordingPicker={picker}/>;
 return <App key={selected} recordingId={selected} recordingPicker={picker}/>;
}
