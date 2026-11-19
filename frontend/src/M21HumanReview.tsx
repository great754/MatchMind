import {useEffect,useState} from 'react';
import CVHumanReview from './CVHumanReview';
type Source={recording_id:string,available:boolean,tasks:number,saved_tasks:number};
export default function M21HumanReview(){
 const [sources,setSources]=useState<Source[]>([]),[error,setError]=useState('');
 useEffect(()=>{let active=true;fetch('/api/cv/m21/review/sources').then(async r=>{if(!r.ok)throw Error('The sealed M21 sources are unavailable.');return r.json()}).then(d=>{if(active)setSources(d.sources)}).catch(e=>{if(active)setError(e.message)});return()=>{active=false}},[]);
 const requested=new URLSearchParams(window.location.search).get('recording');
 const selected=sources.find(s=>s.recording_id===requested&&s.available)??sources.find(s=>s.available);
 if(error)return <main className="app-shell"><p role="alert">{error}</p><a href="/?recording=test_match">Return to analysis</a></main>;
 if(!selected)return <main className="app-shell"><p role="status">Loading sealed source-only review…</p></main>;
 return <><div className="app-shell"><p>Saved source tasks: {sources.reduce((n,s)=>n+s.saved_tasks,0)}/{sources.reduce((n,s)=>n+s.tasks,0)}. The two excerpts belong to one new match group; they are separate from the original 60 development reviews.</p></div><CVHumanReview key={selected.recording_id} recordingId={selected.recording_id} apiRoot="/api/cv/m21" reviewKey="m21" sourceRecordings={sources.filter(s=>s.available).map(s=>s.recording_id)}/></>;
}
