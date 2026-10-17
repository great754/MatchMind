import {useState,type ReactNode} from 'react';
import {mediaUrl} from './api';

export default function VideoOnlyRecording({recordingId,mediaPath,recordingPicker}:{recordingId:string,mediaPath:string,recordingPicker:ReactNode}){
 const [failed,setFailed]=useState(false);
 return <><header className="topbar"><a className="brand" href="#">TactiqAI</a><span className="workspace-label">Match Intelligence</span><div className="header-right">{recordingPicker}<span className="connection"><i/>Video connected</span></div></header><main className="workspace"><section className="card video-card"><div className="section-bar"><h2>Match footage · {recordingId}</h2><span className="quiet-label">Analysis pending</span></div><div className="video-stage">{failed?<div className="media-empty"><h3>Match video unavailable</h3><p>The video could not be loaded. Try another recording or refresh to retry.</p></div>:<video src={mediaUrl(mediaPath)} controls preload="metadata" onError={()=>setFailed(true)}/>}</div></section><section className="card"><div className="section-bar"><h2>Match analysis is not available yet</h2></div><p className="empty-copy">You can watch and seek through this recording. Tactical replay, match statistics and Ask TactiqAI will be available once its analysis is ready.</p></section></main></>;
}
