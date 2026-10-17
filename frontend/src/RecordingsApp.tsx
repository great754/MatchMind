import {useEffect,useState} from 'react';
import App from './App';
import {recordingList} from './api';
import VideoOnlyRecording from './VideoOnlyRecording';
type Recording={recording_id:string,match_id:string,status:'available'|'not_processed'|'unavailable'|'error',media_available?:boolean,media_url?:string|null};
const selectable=(r:Recording)=>r.status==='available'||Boolean(r.media_available&&r.media_url);
export default function RecordingsApp(){
 const [items,setItems]=useState<Recording[]>([]);
 const [selected,setSelected]=useState('118575_clip'),[error,setError]=useState('');
 useEffect(()=>{let alive=true;recordingList().then(data=>{if(!alive)return;setItems(data.recordings);if(!data.recordings.some(r=>r.recording_id===selected&&selectable(r))){const first=data.recordings.find(selectable);if(first)setSelected(first.recording_id)}}).catch(()=>{if(alive)setError('Recording list unavailable; showing the default recording.')});return()=>{alive=false}},[]);
 const options:Recording[]=items.length?items:[{recording_id:'118575_clip',match_id:'118575',status:'available'}];
 const picker=<label className="select-label"><span className="sr-only">Recording</span><select aria-label="Recording" value={selected} onChange={e=>setSelected(e.target.value)} title={error||'Choose a match recording'}>{options.map(r=><option key={r.recording_id} value={r.recording_id} disabled={!selectable(r)}>{r.recording_id}{r.status==='available'?'':r.media_available?' · video ready, analysis pending':r.status==='not_processed'?' · awaiting processing':r.status==='unavailable'?' · artifacts unavailable':' · artifact error'}</option>)}</select></label>;
 const current=items.find(r=>r.recording_id===selected);
 if(current&&current.status!=='available'&&current.media_available&&current.media_url)return <VideoOnlyRecording key={selected} recordingId={selected} mediaPath={current.media_url} recordingPicker={picker}/>;
 return <App key={selected} recordingId={selected} recordingPicker={picker}/>;
}
