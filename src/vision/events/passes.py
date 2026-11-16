"""Offline pass hypotheses from current visual ball evidence and explicit player segments.

This layer never modifies canonical possession, infers ball height or labels goals.
"""
import hashlib,json,math
from collections import Counter
from typing import Literal
from pydantic import Field,model_validator
from src.query.schema import Closed

VERSION='cv-passes-v1'

class PassConfig(Closed):
    contact_distance_heights:float=Field(default=.5,gt=0,le=1)
    contact_ambiguity_margin:float=Field(default=.15,ge=0,le=1)
    minimum_contact_seconds:float=Field(default=.08,gt=0,le=1)
    contact_gap_seconds:float=Field(default=.12,ge=0,le=.5)
    minimum_flight_seconds:float=Field(default=.08,gt=0,le=1)
    maximum_transfer_seconds:float=Field(default=4.0,gt=0,le=10)
    maximum_ball_gap_seconds:float=Field(default=.24,gt=0,le=1)
    minimum_ball_observed_fraction:float=Field(default=.5,gt=0,le=1)
    minimum_separation_heights:float=Field(default=1.5,gt=0,le=10)
    minimum_team_support:float=Field(default=.65,ge=0,le=1)

class VideoPass(Closed):
    schema_version:Literal['cv-passes-v1']='cv-passes-v1'
    source:Literal['VIDEO_CV']='VIDEO_CV'
    recording_id:str
    event_id:str
    status:Literal['estimated','candidate']
    event_type:Literal['pass','transfer_candidate']
    sender_ref:str
    receiver_ref:str
    sender_entity:str
    receiver_entity:str
    team:Literal['left','right']|None
    start_frame:int=Field(ge=0)
    end_frame:int=Field(ge=0)
    fps:float=Field(gt=0)
    start:float=Field(ge=0)
    end:float=Field(ge=0)
    camera_epoch:int=Field(ge=0)
    ball_track_id:int|None
    ball_observed_fraction:float=Field(ge=0,le=1)
    ball_frames:list[int]
    sender_contact_frames:list[int]
    receiver_contact_frames:list[int]
    reasons:list[str]
    start_image:tuple[float,float]
    end_image:tuple[float,float]
    start_pitch:tuple[float,float]|None=None
    end_pitch:tuple[float,float]|None=None
    receiver_release_ref:str|None=None
    receiver_release_entity:str|None=None
    @model_validator(mode='after')
    def clocks(self):
        if self.start_frame>=self.end_frame or abs(self.start-self.start_frame/self.fps)>1e-6 or abs(self.end-self.end_frame/self.fps)>1e-6:raise ValueError('Invalid video event clock')
        if self.receiver_release_ref is not None and self.receiver_release_entity!=self.receiver_entity:raise ValueError('Receiver release lookup must retain explicit M9 identity')
        if self.sender_ref==self.receiver_ref or self.sender_entity==self.receiver_entity:raise ValueError('Self-transfer is not a pass')
        if self.ball_frames!=sorted(set(self.ball_frames)) or any(not self.start_frame<=i<=self.end_frame for i in self.ball_frames):raise ValueError('Invalid visual support frames')
        if not self.sender_contact_frames or not self.receiver_contact_frames or max(self.sender_contact_frames)!=self.start_frame or min(self.receiver_contact_frames)!=self.end_frame:raise ValueError('Missing endpoint contact evidence')
        if self.status=='estimated' and (self.team is None or self.reasons or self.event_type!='pass'):raise ValueError('Estimated pass requires passed evidence gates')
        if self.status=='candidate' and (not self.reasons or self.event_type!='transfer_candidate'):raise ValueError('Candidate requires explicit failed evidence gates')
        if (self.start_pitch is None)!=(self.end_pitch is None):raise ValueError('Both map endpoints required')
        for point in [self.start_pitch,self.end_pitch]:
            if point is not None and not (0<=point[0]<=105 and 0<=point[1]<=68):raise ValueError('Pitch endpoint outside field')
        return self

def signature(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def visible_players(world):
    boxes={e['track_id']:e['image_bbox'] for e in world['entities'] if e.get('track_id') and e.get('image_bbox') and e['visibility']=='visible'}
    boxes.update({u['observation']['track_id']:u['observation']['bbox'] for u in world['unassigned']})
    people={}
    for p in world['player_world']['players']:
        segment=p.get('current_segment_id');track=p.get('current_track_id')
        if p['visibility_state']!='VISIBLE' or p['identity_status'] in ['UNKNOWN','AMBIGUOUS'] or not segment or track not in boxes:continue
        bbox=boxes[track]
        if bbox[2]<=0 or bbox[3]<=0:continue
        ref='segment:'+segment;team=p['team'];point=p.get('current_position')
        metric=point if point is not None and p.get('position_provenance') not in [None,'UNKNOWN'] and p.get('position_uncertainty_m') is not None and p['position_uncertainty_m']<=8 and 0<=point[0]<=105 and 0<=point[1]<=68 else None
        people[ref]={'ref':ref,'entity':p['entity_id'],'bbox':bbox,'team':{'TEAM_A':'left','TEAM_B':'right'}.get(team['label']),'team_support':team['confidence'],'pitch':metric}
    return people

def frame_evidence(world,ball,cfg):
    people=visible_players(world);observed=bool(ball['directly_observed'] and ball['tracking_status'] in ['DETECTED','TRACKED'] and ball.get('image_position') is not None)
    owner=None;reason='BALL_NOT_DIRECTLY_OBSERVED';candidates=[]
    if observed:
        bx,by=ball['image_position']
        for ref,p in people.items():
            x,y,w,h=p['bbox'];distance=math.hypot(bx-(x+w/2),by-(y+h))/h
            if distance<=cfg.contact_distance_heights and y+.5*h<=by<=y+1.2*h:candidates.append((distance,ref))
        candidates.sort()
        if candidates and (len(candidates)==1 or candidates[1][0]-candidates[0][0]>=cfg.contact_ambiguity_margin):owner=candidates[0][1];reason='CLEAR_LOWER_BODY_PROXIMITY'
        else:reason='AMBIGUOUS_CONTACT' if candidates else 'NO_PLAYER_CONTACT'
    return {'frame':world['frame'],'epoch':world['camera']['epoch'],'cut':world['camera']['continuity']=='cut','observed':observed,'ball_track':ball.get('track_id'),'ball':ball.get('image_position') if observed else None,'owner':owner,'reason':reason,'people':people}

def stable_contacts(rows,fps,cfg):
    groups=[];active=None;minimum=max(2,math.ceil(cfg.minimum_contact_seconds*fps));gap=max(1,math.ceil(cfg.contact_gap_seconds*fps))
    for row in rows:
        owner=row['owner']
        if row['cut']:active=None
        if not owner:continue
        if active and active['ref']==owner and active['epoch']==row['epoch'] and row['frame']-active['frames'][-1]<=gap:
            active['frames'].append(row['frame'])
        else:
            active={'ref':owner,'epoch':row['epoch'],'frames':[row['frame']]};groups.append(active)
    return [g for g in groups if len(g['frames'])>=minimum]

def detect_passes(recording,world_rows,ball_rows,fps,cfg=None,*,receiver_lookup=None):
    cfg=cfg or PassConfig()
    if len(world_rows)!=len(ball_rows):raise ValueError('World/ball frame count mismatch')
    rows=[]
    for i,(w,b) in enumerate(zip(world_rows,ball_rows,strict=True)):
        if w['frame']!=i or b['frame']!=i or abs(w['timestamp']-i/fps)>1e-6 or abs(b['timestamp']-i/fps)>1e-6:raise ValueError('World/ball clock mismatch')
        rows.append(frame_evidence(w,b,cfg))
    return events_from_evidence(recording,rows,fps,cfg,receiver_lookup=receiver_lookup)


def events_from_evidence(recording,rows,fps,cfg=None,*,receiver_lookup=None):
    cfg=cfg or PassConfig()
    if any(r['frame']!=i for i,r in enumerate(rows)):raise ValueError('Evidence frame order mismatch')
    contacts=stable_contacts(rows,fps,cfg);events=[];seen=set()
    for a,b in zip(contacts,contacts[1:]):
        if a['ref']==b['ref']:continue
        start=a['frames'][-1];end=b['frames'][0]
        if end<=start or (end-start)/fps>cfg.maximum_transfer_seconds:continue
        first=rows[start];last=rows[end];sender=first['people'][a['ref']];receiver=last['people'][b['ref']]
        if sender['entity']==receiver['entity']:continue
        window=rows[start:end+1];observed=[r['frame'] for r in window if r['observed']];reasons=[]
        if any(r['cut'] or r['epoch']!=first['epoch'] for r in window):reasons.append('CAMERA_CUT_OR_EPOCH_CHANGE')
        if first['ball_track'] is None or any(r['ball_track']!=first['ball_track'] for r in window if r['observed']):reasons.append('BALL_TRACK_DISCONTINUITY')
        fraction=len(observed)/len(window)
        if fraction<cfg.minimum_ball_observed_fraction:reasons.append('INSUFFICIENT_OBSERVED_BALL_COVERAGE')
        if any((y-x-1)/fps>cfg.maximum_ball_gap_seconds for x,y in zip(observed,observed[1:])):reasons.append('BALL_OBSERVATION_GAP')
        flight=[r for r in window[1:-1] if r['observed'] and r['owner'] is None and r['reason']=='NO_PLAYER_CONTACT']
        if len(flight)<max(2,math.ceil(cfg.minimum_flight_seconds*fps)):reasons.append('NO_OBSERVED_RELEASE_AND_FLIGHT')
        if any(r['reason']=='AMBIGUOUS_CONTACT' for r in window):reasons.append('CONTESTED_LOWER_BODY_CONTACT')
        if sender['team'] is None or receiver['team'] is None or min(sender['team_support'],receiver['team_support'])<cfg.minimum_team_support:reasons.append('TEAM_UNCERTAIN')
        elif sender['team']!=receiver['team']:reasons.append('OPPONENT_TRANSFER_NOT_PROVEN_PASS_ATTEMPT')
        at_start=first['people'].get(b['ref'])
        if at_start is None and receiver_lookup is not None:at_start=receiver_lookup(first,receiver)
        if at_start is None:reasons.append('RECEIVER_NOT_VISIBLE_AT_RELEASE')
        else:
            sx,sy,sw,sh=sender['bbox'];rx,ry,rw,rh=at_start['bbox'];distance=math.hypot(sx+sw/2-(rx+rw/2),sy+sh-(ry+rh))/max(sh,rh)
            if distance<cfg.minimum_separation_heights:reasons.append('INSUFFICIENT_SAME_FRAME_PLAYER_SEPARATION')
        sp=sender['pitch'];rp=receiver['pitch']
        if sp is None or rp is None:sp=rp=None
        status='candidate' if reasons else 'estimated';key=(a['ref'],b['ref'],start,end)
        if key in seen:continue
        seen.add(key)
        events.append(VideoPass(recording_id=recording,event_id='cvpass_'+signature([VERSION,recording,key,cfg.model_dump()])[:24],status=status,event_type='transfer_candidate' if reasons else 'pass',sender_ref=a['ref'],receiver_ref=b['ref'],sender_entity=sender['entity'],receiver_entity=receiver['entity'],team=sender['team'] if sender['team']==receiver['team'] else None,start_frame=start,end_frame=end,fps=fps,start=start/fps,end=end/fps,camera_epoch=first['epoch'],ball_track_id=first['ball_track'],ball_observed_fraction=fraction,ball_frames=observed,sender_contact_frames=a['frames'],receiver_contact_frames=b['frames'],reasons=reasons,start_image=first['ball'],end_image=last['ball'],start_pitch=sp,end_pitch=rp,receiver_release_ref=at_start['ref'] if at_start is not None else None,receiver_release_entity=at_start['entity'] if at_start is not None else None).model_dump())
    counts=Counter(r['reason'] for r in rows)
    summary={'frames':len(rows),'ball_observed_frames':sum(r['observed'] for r in rows),'ball_observed_fraction':sum(r['observed'] for r in rows)/max(1,len(rows)),'stable_contact_runs':len(contacts),'completed_pass_estimates':sum(e['status']=='estimated' for e in events),'transfer_candidates':sum(e['status']=='candidate' for e in events),'frame_diagnostics':dict(counts),'candidate_reasons':dict(Counter(reason for e in events for reason in e['reasons'])),'goal_count':None,'shot_count':None,'pass_attempts':None,'pass_accuracy':None,'independent_event_accuracy':None}
    return events,summary,rows
