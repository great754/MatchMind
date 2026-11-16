"""Player actions/statistics over existing annotations and measurements only."""
import copy,math
from src.highlights.index import MatchEventIndex,digest
from .schema import Metric

def finite(value):
    try:return float(value) if value is not None and math.isfinite(float(value)) else None
    except (TypeError,ValueError):return None

class AssistantEventIndex(MatchEventIndex):
    version='m18.1'
    def __init__(self,recording,tools,catalog):
        super().__init__(recording,tools);self.catalog=catalog;self.transfer_rows={}
        if tools is None or self.timebase is None:return
        canonical={e.event_id:e for e in tools.events}
        for row in self.events:
            row['source_event_ids']=[row['event_id']]
            if row['event_type']=='pass':
                _,transfer=tools._transfer(row['event_id'])
                if transfer:
                    self.transfer_rows[row['event_id']]=transfer
                    for k in ['difficulty_score_0_100','start_x_m','start_y_m','end_x_m','end_y_m','sender_nearby_defenders','passing_lane_defenders','target_nearest_defender_at_launch_m','receiver_pressure_at_arrival','forward_progression_m']:
                        v=finite(transfer.get(k))
                        if v is not None:row['metrics'][k]=v
        extra=[]
        for row in self.events:
            e=canonical[row['event_id']]
            if row['event_type']=='pass' and row['safe_for_automatic'] and e.receiver_actor_id is not None:
                r=copy.deepcopy(row);r.update(event_id='reception_'+digest([recording,e.event_id])[:20],event_type='reception',actor_id=e.receiver_actor_id,player=e.receiver_id,start=row['end'],timestamp=row['end'],metrics={},related_players=[p for p in [e.player_id] if p is not None],related_actor_ids=[e.actor_id] if e.actor_id else [],related_event_ids=[e.event_id]);extra.append(r)
        for e in tools.events:
            kind={'HEADER':'header','DRIVE':'carry'}.get(e.metrics.action_label) if e.source=='bas' and e.validated_by_bas and e.bas_annotation_id is not None else None
            if kind:
                extra.append(self.from_canonical(e,kind))
            elif e.type.value=='TRANSFER_INTERCEPTED' and e.receiver_actor_id is not None and (e.metrics.observed_step_fraction or 0)>=.9:
                r=self.from_canonical(e,'interception');r.update(actor_id=e.receiver_actor_id,player=e.receiver_id,team=catalog.by_ref.get('actor:'+str(e.receiver_actor_id),{}).get('team'),verification='inferred');extra.append(r)
        self.events+=extra;self.events.sort(key=lambda e:(e['start'],e['event_id']));self.rehash()
    def attach_cv(self,layer):
        from src.highlights.schema import VideoTimebase
        self.cv_layer=layer;layer.check();m=layer.manifest
        self.timebase=VideoTimebase(fps=m['fps'],frames=m['frames'],period=None,period_start_match_seconds=None,alignment_sources=[{'name':'cv_events','sha256':m['events_sha256']},{'name':'cv_world','sha256':m['world_frames_sha256']}])
        self.unavailable_reason=None;self.version='cv-passes-v1';self.candidates=[]
        for e in layer.events:
            row={'event_id':e['event_id'],'group_id':e['event_id'],'recording_id':self.recording,'event_type':'pass','start':e['start'],'end':e['end'],'timestamp':e['start'],'timestamp_confidence':'Source CV frame clock; match period unknown','team':e['team'],'player':None,'actor_id':None,'player_ref':e['sender_ref'],'receiver_ref':e['receiver_ref'],'related_players':[],'related_actor_ids':[],'metrics':{'ball_observed_fraction':e['ball_observed_fraction']},'evidence_source':'VIDEO_CV_'+m['ball_version'],'verification':'inferred' if e['status']=='estimated' else 'candidate','safe_for_automatic':e['status']=='estimated','validation_state':'experimental_visual_pass_estimate' if e['status']=='estimated' else 'possible_transfer_needs_review','annotation_id':None,'source_event_type':'CV_PASS_ESTIMATE' if e['status']=='estimated' else 'CV_TRANSFER_CANDIDATE','source_video':f'/matches/{self.recording}/video.mp4','camera':self.recording,'timebase':self.timebase.model_dump(),'period_relative_seconds':e['start'],'match_relative_seconds':None,'related_event_ids':[],'source_event_ids':[e['event_id']],'possession_sequence':None}
            if e['status']=='candidate':self.candidates.append(row);continue
            if e['start_pitch'] is not None:
                values=e['start_pitch']+e['end_pitch']
                row['metrics'].update(dict(zip(['start_x_m','start_y_m','end_x_m','end_y_m'],values)))
            self.events.append(row)
            reception={**row,'event_id':e['event_id']+'_reception','event_type':'reception','player_ref':e['receiver_ref'],'receiver_ref':None,'start':e['end'],'end':e['end'],'timestamp':e['end'],'period_relative_seconds':e['end'],'related_event_ids':[e['event_id']]};self.events.append(reception)
        for e in self.events:self.by_type[e['event_type']].append(e)
        self.rehash()
    def from_canonical(self,e,kind):
        t=self.timebase;return {'event_id':e.event_id,'group_id':e.group_id,'recording_id':self.recording,'event_type':kind,'start':e.timestamp,'end':e.timestamp,'timestamp':e.timestamp,'timestamp_confidence':'existing synchronized annotation','team':e.team,'player':e.player_id,'actor_id':e.actor_id,'related_players':[],'related_actor_ids':[],'metrics':{},'evidence_source':e.source,'verification':'annotated','safe_for_automatic':True,'validation_state':e.validation_state,'annotation_id':e.bas_annotation_id,'source_event_type':e.type.value,'source_video':f'/matches/{self.recording}/video.mp4','camera':self.recording,'timebase':t.model_dump(),'period_relative_seconds':t.period_seconds(e.timestamp),'match_relative_seconds':t.match_seconds(e.timestamp),'related_event_ids':[],'source_event_ids':[e.event_id],'possession_sequence':None}
    def add_sprint(self,sprint):
        t=self.timebase;row={'event_id':sprint['event_id'],'group_id':sprint['event_id'],'recording_id':self.recording,'event_type':'sprint','start':sprint['start'],'end':sprint['end'],'timestamp':sprint['start'],'timestamp_confidence':'continuous source metric frames','team':sprint['team'],'player':sprint['player'],'actor_id':sprint['actor_id'],'related_players':[],'related_actor_ids':[],'metrics':{'speed_mps':sprint['speed_mps']},'evidence_source':'existing_player_pitch_trajectories','verification':'inferred','safe_for_automatic':True,'validation_state':'sustained_metric_estimate' if sprint.get('qualifies_as_sprint',True) else 'continuous_run_estimate','annotation_id':None,'source_event_type':'M18_SUSTAINED_RUN','source_video':f'/matches/{self.recording}/video.mp4','camera':self.recording,'timebase':t.model_dump(),'period_relative_seconds':t.period_seconds(sprint['start']),'match_relative_seconds':t.match_seconds(sprint['start']),'related_event_ids':[],'source_event_ids':[],'possession_sequence':None}
        if not any(e['event_id']==row['event_id'] for e in self.events):self.events.append(row);self.rehash()
        return row
    def rehash(self):
        source=getattr(self,'cv_layer',None)
        self.hash=digest([self.version,self.recording,self.events,getattr(self,'candidates',[]),{k:source.manifest[k] for k in ['events_sha256','world_frames_sha256','ball_frames_sha256']} if source else None])
    def select(self,actions=None,refs=None,team=None,start=None,end=None):
        if getattr(self,'cv_layer',None):self.cv_layer.check()
        actors={self.catalog.by_ref[r]['actor_id'] for r in refs or [] if r in self.catalog.by_ref and self.catalog.by_ref[r]['actor_id'] is not None};return [e for e in self.events if e['safe_for_automatic'] and (not actions or e['event_type'] in actions) and (not refs or e.get('player_ref') in refs or e['actor_id'] is not None and e['actor_id'] in actors) and (team is None or e['team']==team) and (start is None or e['start']>=start) and (end is None or e['start']<=end)]

class PlayerIntelligenceService:
    def __init__(self,index,tools,catalog):self.index=index;self.tools=tools;self.catalog=catalog
    def statistics(self,ref):
        p=self.catalog.by_ref[ref];window=(p['start'],p['end']);metrics=[]
        if self.tools is None:
            layer=getattr(self.index,'cv_layer',None)
            if layer is None:return metrics
            layer.check();rows=self.index.select(refs=[ref])
            for name,value,definition in [
                ('Completed pass estimates',sum(e['event_type']=='pass' for e in rows),'Completed visual pass estimates that passed the evidence gates; not a complete count of this player’s passes.'),
                ('Reception estimates',sum(e['event_type']=='reception' for e in rows),'Observed receiver endpoints of completed visual pass estimates.'),
                ('Possible transfers for review',sum(e['sender_ref']==ref for e in layer.events if e['status']=='candidate'),'Uncertain transfers excluded from completed-pass counts; not confirmed passes.')]:
                metrics.append(Metric(name=name,value=value,unit='events',definition=definition,source='cv-passes-v1/'+layer.manifest['ball_version'],support='estimated',samples=value,window=window,player_ref=ref,limitations=['Sparse ball visibility and uncertain identities/teams limit coverage; independent accuracy unavailable.']).model_dump())
            return metrics
        actor=p['actor_id'];rows=self.index.select(refs=[ref]);summary=next((r for r in self.tools.players if int(r['player_id'])==actor),{})
        for kind,title in [('shot','Recorded shots'),('goal','Recorded goals'),('pass','Completed pass estimates'),('reception','Reception estimates'),('tackle','Recorded tackles'),('interception','Interception estimates'),('cross','Recorded crosses'),('header','Recorded headers'),('carry','Recorded ball carries')]:
            es=[e for e in rows if e['event_type']==kind];metrics.append(Metric(name=title,value=len(es),unit='events',definition='Entries explicitly assigned to this player in the available footage; not exhaustive match coverage.',source='canonical_event_exports',support='estimated' if kind in ['pass','reception','interception'] else 'recorded',samples=len(es),window=window,player_ref=ref,limitations=['Missing entries do not prove that no other actions occurred.']).model_dump())
        attempts=[r for r in self.tools.transfers if int(r.get('sender_id') or -1)==actor and (finite(r.get('observed_step_fraction')) or 0)>=.9]
        completed=[r for r in attempts if r.get('outcome')=='completed']
        for name,value,unit,definition in [('Recorded pass attempts',len(attempts),'events','Coverage-qualified transfer attempts, including completed, intercepted, unresolved and out-of-play outcomes.'),('Estimated pass completion',100*len(completed)/len(attempts) if attempts else None,'%','Completed qualifying transfers divided by all qualifying attempts; not an independently verified match completion rate.')]:metrics.append(Metric(name=name,value=value,unit=unit,definition=definition,source='transfers.csv',support='estimated',samples=len(attempts),window=window,player_ref=ref).model_dump())
        for field,title,unit in [('measured_distance_m','Tracked distance','m'),('max_speed_mps','Peak estimated speed','m/s'),('controlled_seconds','Estimated direct ball control','s')]:
            v=finite(summary.get(field));metrics.append(Metric(name=title,value=v,unit=unit,definition='Existing player summary over observed source segments; peak speed is not a sustained sprint ranking.',source='player_summary.csv',support='estimated' if v is not None else 'unavailable',samples=int(self.tools.summary.get('frames',0)),window=window,player_ref=ref).model_dump())
        return metrics

class TacticalExplanationService:
    @staticmethod
    def pass_explanation(event):
        m=event['metrics'];parts=[]
        if m.get('distance_m') is not None:parts.append(f"it covered about {m['distance_m']:.1f} meters")
        if m.get('passing_lane_defenders',0)>0:parts.append(f"{int(m['passing_lane_defenders'])} opponent{'s were' if m['passing_lane_defenders']!=1 else ' was'} near the straight line between its start and finish")
        if m.get('target_nearest_defender_at_launch_m') is not None:parts.append(f"the nearest defender was about {m['target_nearest_defender_at_launch_m']:.1f} meters from its eventual destination when the pass began")
        if m.get('sender_nearby_defenders',0)>0:parts.append(f"{int(m['sender_nearby_defenders'])} opponent{'s were' if m['sender_nearby_defenders']!=1 else ' was'} near the passer")
        return ('Its difficulty is an estimate based on recorded distance, forward movement and nearby defenders. '+('; '.join(parts).capitalize()+'.' if parts else 'Those supporting measurements are unavailable.'))+' This does not measure the probability of completing the pass.'
