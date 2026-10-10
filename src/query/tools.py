"""Read-only bounded tools over existing exports. No provider or raw tracking access."""
from bisect import bisect_right
import hashlib
import json
import math
from pathlib import Path
import pandas as pd
from src.intelligence.schema import CanonicalEvent
from src.tactics.schema import TacticalState
from src.tactics.passing_network import transfer_network
from .schema import Fact,QueryResult


class MatchMindTools:
    def __init__(self,directory):
        self.directory=Path(directory)
        self.summary=self._json('summary.json',{})
        self.event_export=self._json('events.json',{'events':[],'player_labels':{}})
        self.events=[CanonicalEvent.model_validate(e) for e in self.event_export['events']]
        self.states=[TacticalState.model_validate(s) for s in self._json('tactical_states.json',{'states':[]})['states']]
        self.times=[s.timestamp for s in self.states]
        self.evidence=self._json('pre_event_evidence.json',{'items':[]})['items']
        self.players=self._csv('player_summary.csv')
        self.transfers=self._csv('transfers.csv')
        self.tactical_summary=self._json('tactical_summary.json',{})
        self.definitions=self._json('measurement_definitions.json',{})
        self.limitations=self.summary.get('limitations',[])

    def _json(self,name,default):
        p=self.directory/name
        return json.loads(p.read_text()) if p.exists() else default

    def _csv(self,name):
        p=self.directory/name
        return pd.read_csv(p).astype(object).where(lambda f:f.notna(),None).to_dict('records') if p.exists() else []

    @staticmethod
    def _team(team):
        if team not in (None,'left','right'):raise ValueError('team must be left/right or None')

    @staticmethod
    def _limit(limit):
        if isinstance(limit,bool) or not isinstance(limit,int) or not 1<=limit<=100:raise ValueError('limit must be 1..100')

    def _result(self,kind,records,extra=()):
        facts=[]
        for record_id,source,fields,timestamp,event_id in records:
            for field,value in fields.items():
                if isinstance(value,float) and not math.isfinite(value):value=None
                identity=f"{self.summary.get('match_id')}:{self.summary.get('scope')}:{self.summary.get('half')}:{source}:{record_id}:{timestamp}:{field}"
                facts.append(Fact(fact_id='fact_'+hashlib.sha256(identity.encode()).hexdigest()[:20],record_id=str(record_id),field=field,value=value,
                    source=f'{source}#record={record_id}&field={field}',timestamp=timestamp,event_id=event_id))
        return QueryResult(question_type=kind,status='ok' if any(f.value is not None for f in facts) else 'unknown',facts=facts,
            limitations=list(dict.fromkeys(self.limitations+list(extra))),provenance=sorted({f.source.split('#')[0] for f in facts}))

    def get_match_summary(self):
        return self._result('match_summary',[('summary','summary.json',{k:self.summary.get(k) for k in
            ('match_id','scope','half','frames','duration_seconds','trajectory_shot_candidates','bas_shots','player_coordinate_source')},None,None)])

    def get_player_summary(self,player_id):
        # Actor IDs only at the API boundary; display tracker IDs never silently alias actors.
        rows=[(str(p['player_id']),'player_summary.csv',p,p.get('peak_speed_seconds'),None) for p in self.players if p['player_id']==player_id]
        return self._result('player_summary',rows,('Actor/tracker IDs are not verified jersey numbers; player names are unavailable.',))

    def get_fastest_players(self,limit=5):
        self._limit(limit)
        players=sorted([p for p in self.players if p.get('max_speed_kmh') is not None],key=lambda p:(-p['max_speed_kmh'],p['player_id']))[:limit]
        return self._result('fastest_players',[(str(p['player_id']),'player_summary.csv',
            {k:p.get(k) for k in ('player_id','mot_id','team','max_speed_kmh','valid_motion_fraction','peak_speed_seconds')},p.get('peak_speed_seconds'),None) for p in players])

    def get_possession_summary(self,team=None):
        self._team(team)
        sides=self.summary.get('team_possession',{})
        rows=[(side,'summary.json',dict(team=side,**metrics),None,None) for side,metrics in sides.items() if team is None or side==team]
        if team is None and all(side in sides for side in ('left','right')):
            left=sides['left']['controlled_seconds'];right=sides['right']['controlled_seconds']
            winner='left' if left>right else 'right' if right>left else 'equal'
            rows.append(('comparison','summary.json',dict(team_with_more_inferred_control=winner,comparison_basis='controlled_seconds; excludes unowned frames'),None,None))
        return self._result('possession_summary',rows,
            ('Possession is inferred; percentages of all frames and of controlled frames have different denominators.',))

    def get_events(self,event_type=None,team=None,start_time=None,end_time=None):
        self._team(team)
        if any(t is not None and (not math.isfinite(t) or t<0) for t in (start_time,end_time)):raise ValueError('Times must be finite and nonnegative')
        if start_time is not None and end_time is not None and start_time>end_time:raise ValueError('Time range must be ordered')
        chosen=[e for e in self.events if (event_type is None or e.type.value==event_type) and (team is None or e.team==team)
            and (start_time is None or e.timestamp>=start_time) and (end_time is None or e.timestamp<=end_time)]
        return self._result('events',[(e.event_id,'events.json',dict(type=e.type.value,team=e.team,player_id=e.player_id,actor_id=e.actor_id,
            receiver_id=e.receiver_id,validation_state=e.validation_state,**e.metrics.model_dump(exclude_none=True)),e.timestamp,e.event_id) for e in chosen])

    def _transfer(self,event_id):
        event=next((e for e in self.events if e.event_id==event_id and e.source=='transfer'),None)
        if event is None:return None,None
        row=next((r for r in self.transfers if r['start_frame']==event.metrics.start_frame and r['sender_id']==event.actor_id),None)
        return event,row

    def get_transfer(self,event_id):
        event,row=self._transfer(event_id)
        fields=('transfer_id','sender_id','receiver_id','sender_team','outcome','start_seconds','end_seconds','distance_m','forward_progression_m',
            'difficulty_score_0_100','sender_nearby_defenders','passing_lane_defenders','target_nearest_defender_at_launch_m','receiver_pressure_at_arrival',
            'ball_speed_peak_kmh','duration_seconds','bas_supported_pass','validation_scope','start_x_m','start_y_m','end_x_m','end_y_m')
        return self._result('transfer',[(event_id,'transfers.csv',{k:row.get(k) for k in fields},row['start_seconds'],event_id)] if row else [],
            ('Difficulty is retrospective and heuristic; lane and target features use the realized endpoint, not a calibrated completion probability.',))

    def get_difficult_transfers(self,team=None,limit=5):
        self._team(team);self._limit(limit)
        events=[e for e in self.events if e.type.value=='PASS_COMPLETED' and (team is None or e.team==team) and e.metrics.difficulty_score_0_100 is not None]
        events=sorted(events,key=lambda e:(-e.metrics.difficulty_score_0_100,e.timestamp,e.event_id))[:limit]
        results=[self.get_transfer(e.event_id) for e in events]
        return QueryResult(question_type='difficult_transfers',status='ok' if results else 'unknown',facts=[f for r in results for f in r.facts],
            limitations=list(dict.fromkeys(self.limitations+['Ranking includes only completed observable transfers with available retrospective heuristic difficulty.'])),provenance=['transfers.csv','events.json'])

    def get_tactical_state(self,timestamp):
        if not math.isfinite(timestamp) or timestamp<0:raise ValueError('timestamp must be finite and nonnegative')
        # Never pretend the last snapshot describes an out-of-recording time.
        index=bisect_right(self.times,timestamp)-1 if timestamp<self.summary.get('duration_seconds',0) else -1
        if index<0:return self._result('tactical_state',[])
        s=self.states[index];records=[]
        for team in ('left','right'):
            side=getattr(s,team)
            records.append((team,'tactical_states.json',dict(team=team,phase=side.phase,block=side.block,
                formation=side.formation.estimate,formation_reason=side.formation.reason,**side.shape.model_dump()),s.timestamp,None))
        records.append(('pressure','tactical_states.json',s.pressure.model_dump(),s.timestamp,None))
        return self._result('tactical_state',records,('Tactical state is sampled at or before the requested time; geometric labels do not establish intent.',))

    def get_pre_event_evidence(self,event_id):
        item=next((x for x in self.evidence if x['event_id']==event_id),None)
        if not item:return self._result('pre_event_evidence',[])
        records=[(event_id,'pre_event_evidence.json',{k:item.get(k) for k in ('event_start_frame','sample_frame','sample_age_s','missing_reason','interpretation')},None,event_id)]
        context=item.get('pre_event_context')
        if context:
            if context['frame']>=item['event_start_frame']:raise ValueError('Evidence is not strictly pre-event')
            s=TacticalState.model_validate(context)
            records.append((event_id+':pressure','pre_event_evidence.json',s.pressure.model_dump(),s.timestamp,event_id))
            for team in ('left','right'):
                side=getattr(s,team)
                records.append((event_id+':'+team,'pre_event_evidence.json',dict(team=team,block=side.block,**side.shape.model_dump()),s.timestamp,event_id))
        return self._result('pre_event_evidence',records,('Strictly earlier geometry is evidence, not proof that pressure caused an event. Offline ball/ownership inputs remain noncausal.',))

    def get_event_sequence(self,event_id,lookback_seconds=15.):
        if not math.isfinite(lookback_seconds) or not 0<lookback_seconds<=60:raise ValueError('lookback must be within 0..60 seconds')
        target=next((e for e in self.events if e.event_id==event_id),None)
        if target is None:return self._result('events',[])
        start=(target.metrics.start_frame/self.summary.get('fps',25)) if target.metrics.start_frame is not None else target.timestamp
        preceding=[e for e in self.events if max(0,start-lookback_seconds)<=e.timestamp<start and e.source!='motion'][-10:]
        result=self.get_events(start_time=max(0,start-lookback_seconds),end_time=max(0,start-1/self.summary.get('fps',25)))
        allowed={e.event_id for e in preceding}
        return QueryResult(question_type='events',status='ok' if preceding else 'unknown',facts=[f for f in result.facts if f.event_id in allowed],
            limitations=result.limitations+['At most the ten latest non-motion events in the strictly earlier window; temporal order does not establish causality.'],provenance=result.provenance)

    def get_pressure_sequences(self,team=None):
        self._team(team)
        events=[e for e in self.events if e.type.value=='HIGH_PRESSURE_SEQUENCE' and (team is None or e.team==team)]
        return self._result('pressure_sequences',[(e.event_id,'events.json',dict(type=e.type.value,team=e.team,**e.tactical.model_dump()),e.timestamp,e.event_id) for e in events],
            ('Sustained closing-pressure sequences only. None detected is not proof that no real pressing occurred.',))

    def get_pressure_peaks(self,team=None):
        self._team(team)
        selected=[s for s in self.states if (team is None or s.pressure.attacking_team==team) and s.pressure.defenders_within_local is not None]
        selected=sorted(selected,key=lambda s:(-s.pressure.defenders_within_local,s.pressure.nearest_defender_distance_m or 0,s.timestamp))[:1]
        return self._result('pressure_sequences',[(str(s.frame),'tactical_states.json',s.pressure.model_dump(),s.timestamp,None) for s in selected],
            ('Ranked pressure snapshots; proximity alone does not establish a sustained press. Team filter means the team under pressure.',))

    def get_formation_timeline(self,team):
        self._team(team)
        if team is None:raise ValueError('team is required')
        rows=[];previous=None
        for s in self.states:
            f=getattr(s,team).formation
            signature=(f.estimate,f.reason)
            if signature!=previous:
                rows.append((team+':'+str(s.frame),'tactical_states.json',dict(team=team,**f.model_dump()),s.timestamp,None));previous=signature
        return self._result('formation_timeline',rows,('Formation is a conservative longitudinal line-count estimate; unknown means the evidence gates did not pass.',))

    def get_transfer_network(self,team=None,start_time=None,end_time=None):
        self._team(team)
        network=transfer_network(self.events,start_time,end_time,team)
        rows=[('network','events.json',dict(transfer_count=network['transfer_count'],team=team,start_seconds=start_time,end_seconds=end_time),None,None)]
        rows.extend((f"edge:{e['team']}:{e['sender_id']}:{e['receiver_id']}",'events.json',e,None,None) for e in network['edges'])
        for node in network['nodes']:
            partners={e['receiver_id'] if e['sender_id']==node['player_id'] else e['sender_id'] for e in network['edges']
                if e['team']==node['team'] and node['player_id'] in (e['sender_id'],e['receiver_id'])}
            rows.append((f"node:{node['team']}:{node['player_id']}",'events.json',dict(**node,distinct_teammates=len(partners)),None,None))
        return self._result('transfer_network',rows,('Directed completed observable transfers, not official passes; receipt must fall within the selected window.',))

    def get_event_validation(self,event_id):
        e=next((e for e in self.events if e.event_id==event_id),None)
        return self._result('event_validation',[(event_id,'events.json',dict(validation_state=e.validation_state,validated_by_bas=e.validated_by_bas,
            bas_annotation_id=e.bas_annotation_id,observation_kind=e.observation_kind),e.timestamp,event_id)] if e else [],
            ('BAS support validates timing/sender only; it is not independent outcome or tactical validation.',))
