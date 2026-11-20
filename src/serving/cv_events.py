"""Strict optional CV event sidecars; never grant canonical artifact readiness."""
import json
from pathlib import Path
from fastapi import HTTPException
from src.vision.events.passes import VideoPass,PassConfig,signature,events_from_evidence,frame_evidence
from .cv import digest

class CVEventReview:
    def __init__(self,directory,review):
        self.directory=Path(directory).resolve();self.review=review;self.path=self.directory/'events.json';self.evidence=self.directory/'evidence_frames.jsonl';self.manifest=json.loads((self.directory/'manifest.json').read_text());m=self.manifest;b=review.ball_review
        if m['schema_version']!='cv-passes-v1' or b is None:raise ValueError('Missing verified ball source')
        for key in ['recording_id','fps','frames','width','height','video_sha256']:
            if m[key]!=review.manifest[key]:raise ValueError('CV event source/clock mismatch')
        if m['world_frames_sha256']!=review.manifest['frames_sha256'] or m['ball_frames_sha256']!=b.manifest['ball_frames_sha256'] or m['ball_detector_sha256']!=b.manifest['detector_sha256'] or m['ball_version']!=b.manifest.get('version','m14'):raise ValueError('CV event upstream source mismatch')
        cfg=PassConfig.model_validate(m['config'])
        if signature(m['config'])!=m['config_sha256'] or digest(self.path)!=m['events_sha256'] or digest(self.evidence)!=m['evidence_sha256']:raise ValueError('CV event hash mismatch')
        self.events=[VideoPass.model_validate(x).model_dump() for x in json.loads(self.path.read_text())]
        if m.get('algorithm_version') in ['cv-pass-m19-v1','cv-pass-m21-v1','cv-pass-m22-v1']:
            from src.vision.events.passes_m19 import existing_m9_receiver
            with self.evidence.open() as source:evidence=[json.loads(line) for line in source]
            replay,_,_=events_from_evidence(m['recording_id'],evidence,m['fps'],cfg,receiver_lookup=existing_m9_receiver)
            if m.get('algorithm_version')=='cv-pass-m21-v1' and m.get('second_stage_enabled'):
                from src.vision.advanced21.passes import verify_events
                replay,_=verify_events(replay,evidence,m['fps'])
            if replay!=self.events:raise ValueError('M19 event gate replay mismatch')
            if len(evidence)!=m['frames']:raise ValueError('M19 evidence frame count mismatch')
            for i,row in enumerate(evidence):
                world=review.frame((i+.001)/m['fps']).model_dump()
                actual=frame_evidence(world,b.frame(i).model_dump(),cfg)
                if signature(actual)!=signature(row):raise ValueError('M19 evidence differs from validated world/ball source')
        elif m.get('algorithm_version') not in [None,'cv-passes-v1']:raise ValueError('Unknown CV event algorithm')
        ids=set()
        for event in self.events:
            if event['event_id'] in ids or event['recording_id']!=m['recording_id'] or event['fps']!=m['fps'] or event['end_frame']>=m['frames']:raise ValueError('CV event identity/clock mismatch')
            ids.add(event['event_id'])
            for key,who in [('start_frame','sender_ref'),('end_frame','receiver_ref')]:
                frame=event[key];world=review.frame((frame+.001)/m['fps']);ball=b.frame(frame)
                people=world.player_world.players
                if not ball.directly_observed or not any(p.current_segment_id and 'segment:'+p.current_segment_id==event[who] and p.visibility_state=='VISIBLE' and p.identity_status not in ['UNKNOWN','AMBIGUOUS'] for p in people):raise ValueError('CV event endpoint is not a current observation')
            if event['status']=='estimated':
                observed=[i for i in range(event['start_frame'],event['end_frame']+1) if b.frame(i).directly_observed]
                if observed!=event['ball_frames'] or any(b.frame(i).track_id!=event['ball_track_id'] for i in observed):raise ValueError('Accepted event ball support does not match its source')
                if abs(len(observed)/(event['end_frame']-event['start_frame']+1)-event['ball_observed_fraction'])>1e-9:raise ValueError('Accepted event coverage mismatch')
                if event.get('receiver_release_ref'):
                    frame=review.frame((event['start_frame']+.001)/m['fps'])
                    if not any(p.current_segment_id and 'segment:'+p.current_segment_id==event['receiver_release_ref'] and p.entity_id==event['receiver_entity'] and p.visibility_state=='VISIBLE' for p in frame.player_world.players):raise ValueError('Receiver at release lacks explicit M9 evidence')
        summary=m['statistics']
        if summary['frames']!=m['frames'] or summary['completed_pass_estimates']!=sum(e['status']=='estimated' for e in self.events) or summary['transfer_candidates']!=sum(e['status']=='candidate' for e in self.events):raise ValueError('CV event count mismatch')
        if any(summary.get(k) is not None for k in ['goal_count','shot_count','pass_attempts','pass_accuracy']):raise ValueError('Unsupported CV metric cannot become a count')
        self.stamps=(self.path.stat().st_mtime_ns,self.evidence.stat().st_mtime_ns)
    def check(self):
        if self.stamps!=(self.path.stat().st_mtime_ns,self.evidence.stat().st_mtime_ns):raise HTTPException(503,'CV events changed; restart validation')
        if self.review.data.stat().st_mtime_ns!=self.review.data_stamp or self.review.video.stat().st_mtime_ns!=self.review.video_stamp:raise HTTPException(503,'CV source changed; restart validation')
        b=self.review.ball_review
        if (b.path.stat().st_mtime_ns,b.canonical.stat().st_mtime_ns)!=b.stamps:raise HTTPException(503,'Ball source changed; restart validation')
    def public(self,labels=None):
        self.check();labels=labels or {}
        return {'recording_id':self.manifest['recording_id'],'version':'cv-passes-v1','source':'VIDEO_CV','ball_version':self.manifest['ball_version'],'algorithm_version':self.manifest.get('algorithm_version','cv-passes-v1'),'experimental_reference_status':self.manifest.get('independent_reference_status','unreviewed'),'statistics':self.manifest['statistics'],'events':[{**e,'sender_label':labels.get(e['sender_ref'],'Unconfirmed player'),'receiver_label':labels.get(e['receiver_ref'],'Unconfirmed player')} for e in self.events],'unavailable':self.manifest['unavailable'],'limitations':['Experimental pass estimates from visual evidence, not official match statistics.','Sparse ball observations and identity/team errors cause missed or incorrect passes.','Uncertain transfers are review candidates, never completed-pass counts.','Pass attempts and accuracy remain unavailable; unsuccessful intent is not established.','Goals, shots, tackles and verified possession are not implemented by this detector.']}
