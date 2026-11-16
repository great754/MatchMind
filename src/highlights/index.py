"""Canonical index of pre-existing events. No new pass/shot/possession inference."""
import hashlib,json,math
from pathlib import Path
from collections import defaultdict
from .schema import VideoTimebase,HighlightEvent

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

class MatchEventIndex:
    version='m17.2'
    def __init__(self,recording,tools=None):
        self.recording=recording;self.unavailable_reason=None;self.events=[];self.by_type=defaultdict(list);self.by_player=defaultdict(list);self.by_team=defaultdict(list)
        if tools is None:
            self.timebase=None;self.hash=digest([self.version,recording,[]]);return
        s=tools.summary
        if not s.get("fps") or not s.get("frames") or s.get("half") not in [1,2,3,4] or "source_offset_frames" not in s:
            self.unavailable_reason="Highlight search is unavailable because this recording’s event clock could not be verified."
            self.timebase=None;self.hash=digest([self.version,recording,[]]);return
        self.timebase=VideoTimebase(fps=s['fps'],frames=s['frames'],source_offset_frames=s.get('source_offset_frames',0),period=s.get('half',1),period_start_match_seconds=0 if s.get('half',1)==1 else s.get('period_start_match_seconds'),alignment_sources=[{'name':Path(f['path']).name,'sha256':f['sha256']} for f in s.get('source_files',[]) if 'sync' in f['path'] or 'class_events' in f['path']])
        groups=defaultdict(list)
        for e in tools.events:groups[e.group_id].append(e.event_id)
        for e in tools.events:
            m=e.metrics;kind=None;status='unsupported';safe=False
            if e.source=='bas' and e.validated_by_bas and e.validation_state=='bas_reference':
                kind={'GOAL':'goal','SHOT_VALIDATED':'shot'}.get(e.type.value) or {'CROSS':'cross','PLAYER SUCCESSFUL TACKLE':'tackle'}.get(m.action_label)
                status='annotated';safe=True
            elif e.type.value=='PASS_COMPLETED' and e.source=='transfer':
                kind='pass';status='inferred';safe=m.outcome=='completed' and (m.observed_step_fraction or 0)>=.9
            elif e.type.value=='SHOT_CANDIDATE':kind='shot';status='candidate'
            elif e.type.value=='POSSESSION_CHANGED':kind='possession_change';status='inferred' # review only
            if kind is None:continue
            start=(m.start_frame if m.start_frame is not None else e.frame)/s['fps'];end=max(e.timestamp,(m.end_frame if m.end_frame is not None else e.frame)/s['fps'])
            if abs(e.timestamp-e.frame/s['fps'])>1e-5 or start<0 or end>self.timebase.frames/self.timebase.fps or start>end:continue
            metrics={}
            if kind=='pass' and m.distance_m is not None and math.isfinite(m.distance_m) and m.distance_m>=0:metrics['distance_m']=m.distance_m
            row={'event_id':e.event_id,'recording_id':recording,'group_id':e.group_id,'event_type':kind,'start':start,'end':end,'timestamp':e.timestamp,'timestamp_confidence':'frame_aligned_export; annotation synchronization inherits source audit','team':e.team,'player':e.player_id,'actor_id':e.actor_id,'related_players':[p for p in [e.receiver_id] if p is not None],'related_actor_ids':[p for p in [e.receiver_actor_id] if p is not None],'metrics':metrics,'evidence_source':e.source,'verification':status,'safe_for_automatic':safe,'validation_state':e.validation_state,'annotation_id':e.bas_annotation_id,'source_event_type':e.type.value,'source_video':f'/matches/{recording}/video.mp4','camera':recording,'timebase':self.timebase.model_dump(),'period_relative_seconds':self.timebase.period_seconds(start),'match_relative_seconds':self.timebase.match_seconds(start),'related_event_ids':[other for other in groups[e.group_id] if other!=e.event_id],'possession_sequence':None}
            self.events.append(HighlightEvent.model_validate(row).model_dump())
        self.events.sort(key=lambda e:(e['start'],e['event_id']))
        for e in self.events:
            self.by_type[e['event_type']].append(e);self.by_team[e['team']].append(e)
            for p in [e['player'],*e['related_players']]:self.by_player[p].append(e)
        self.hash=digest([self.version,recording,self.events])
    def capabilities(self):
        return {'recording_id':self.recording,'index_version':self.version,'index_hash':self.hash,'types':{k:{'automatic':sum(e['safe_for_automatic'] for e in self.by_type[k]),'review_only':sum(not e['safe_for_automatic'] for e in self.by_type[k])} for k in ['goal','shot','pass','cross','tackle','save','possession_change']},'pass_distance':'tracking-derived endpoint distance, not independently verified path length','shot_speed':False,'jersey_identity':False,'home_away_identity':False,'timebase':self.timebase.model_dump() if self.timebase else None,'limitation':self.unavailable_reason or ('CV-only footage has no verified football events.' if self.timebase is None else 'Annotations and tracking estimates retain separate provenance; missing events do not prove absence.')}
    def retrieve(self,plan):
        if self.unavailable_reason:return [],self.unavailable_reason
        if plan.metric in ['shot_speed','chance_quality']:return [],'I can’t reliably rank shots by speed or chance quality in this recording.'
        if plan.team in ['home','away']:return [],'I can’t verify which recorded side is the home or away team.'
        if plan.player is not None and plan.player_namespace=='jersey':return [],'Jersey numbers aren’t verified here. You can ask about a tracked player number instead.'
        rows=[e for k in (plan.event_types or ['goal','shot','pass','cross','tackle']) for e in self.by_type[k] if e['safe_for_automatic'] or plan.review]
        rows=list({e['event_id']:e for e in rows}.values())
        rows=[e for e in rows if (plan.team is None or e['team']==plan.team) and (plan.player is None or (plan.player in [e['actor_id'],*e['related_actor_ids']] if plan.player_namespace=='actor' else plan.player in [e['player'],*e['related_players']])) and (plan.start is None or e['start']>=plan.start) and (plan.end is None or e['start']<=plan.end)]
        rows.sort(key=lambda e:(e['start'],e['event_id']))
        if plan.selection=='maximum':
            if plan.metric!='distance_m' or not rows or any('distance_m' not in e['metrics'] for e in rows):return [],'I can show recorded passes, but I can’t reliably tell which one was the longest.'
            rows=sorted(rows,key=lambda e:(-e['metrics']['distance_m'],e['start']))[:1]
        elif plan.selection=='ordinal':
            if len(rows)<plan.ordinal:return [],f'I found {len(rows)} matching moments, but couldn’t confirm moment number {plan.ordinal}.'
            rows=rows[plan.ordinal-1:plan.ordinal]
        elif plan.selection=='recap':
            # Transparent priority, not an objective excitement or xG measurement.
            rows=sorted(rows,key=lambda e:({'goal':0,'shot':1,'cross':2,'tackle':3,'pass':4}.get(e['event_type'],5),e['start']))[:12];rows.sort(key=lambda e:e['start'])
        if not rows:
            noun='goals' if plan.event_types==['goal'] else 'matching moments'
            return [],f'I couldn’t confirm any {noun} in this recording.'
        return rows,None
