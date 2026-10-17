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
        self.events_available=(self.directory/'events.json').is_file()
        self.event_export=self._json('events.json',{'events':[],'player_labels':{}})
        self.events=[CanonicalEvent.model_validate(e) for e in self.event_export['events']]
        self.states=[TacticalState.model_validate(s) for s in self._json('tactical_states.json',{'states':[]})['states']]
        self.times=[s.timestamp for s in self.states]
        self.evidence=self._json('pre_event_evidence.json',{'items':[]})['items']
        self.players=self._csv('player_summary.csv')
        self.transfers=self._csv('transfers.csv')
        self.transfers_available=(self.directory/'transfers.csv').is_file()
        self.tactical_summary=self._json('tactical_summary.json',{})
        self.definitions=self._json('measurement_definitions.json',{})
        from src.analytics.possession import attribute_possession
        from src.tactics.review import formation_windows
        self.ownership=self._json('possession_attribution.json',attribute_possession(self._csv('possession.csv'),self.summary.get('fps',25)))
        self.ownership_source='possession_attribution.json' if (self.directory/'possession_attribution.json').is_file() else 'possession.csv#continuity_20s'
        self.formation_source='formation_windows.json' if (self.directory/'formation_windows.json').is_file() else 'tactical_states.json#four_minute_aggregation'
        self.ownership_times=[r['timestamp'] for r in self.ownership]
        self.formations=self._json('formation_windows.json',formation_windows(self.states))
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

    def get_players(self,limit=25):
        self._limit(limit)
        return self._result('player_summary',[(str(p['player_id']),'player_summary.csv',p,None,None)
            for p in sorted(self.players,key=lambda p:p['player_id'])[:limit]])

    def get_distance_leaders(self,limit=5):
        self._limit(limit)
        players=sorted([p for p in self.players if p.get('measured_distance_m') is not None],
                       key=lambda p:(-p['measured_distance_m'],p['player_id']))[:limit]
        return self._result('player_summary',[(str(p['player_id']),'player_summary.csv',p,None,None) for p in players])

    def get_fastest_players(self,limit=5):
        self._limit(limit)
        players=sorted([p for p in self.players if p.get('max_speed_kmh') is not None],key=lambda p:(-p['max_speed_kmh'],p['player_id']))[:limit]
        return self._result('fastest_players',[(str(p['player_id']),'player_summary.csv',
            {k:p.get(k) for k in ('player_id','mot_id','team','max_speed_kmh','valid_motion_fraction','peak_speed_seconds')},p.get('peak_speed_seconds'),None) for p in players])

    def get_team_speed_peaks(self,team=None):
        self._team(team)
        rows=[]
        for side in ([team] if team else ['left','right']):
            candidates=[p for p in self.players if p.get('team')==side and p.get('max_speed_kmh') is not None]
            if not candidates:continue
            p=min(candidates,key=lambda p:(-p['max_speed_kmh'],p['player_id']))
            rows.append((str(p['player_id']),'player_summary.csv',
                {k:p.get(k) for k in ('player_id','mot_id','team','max_speed_kmh','valid_motion_fraction','peak_speed_seconds')},p.get('peak_speed_seconds'),None))
        return self._result('fastest_players',rows,('Each team is represented by its highest observed player peak speed, not an average team speed.',))

    def get_confirmed_pass_share(self,team=None):
        self._team(team)
        if self.transfers_available:
            source='transfers.csv'
            counts={side:sum(t.get('outcome')=='completed' and t.get('sender_team')==side
                             for t in self.transfers) for side in ('left','right')}
        elif self.events_available:
            source='events.json'
            counts={side:sum(e.type.value=='PASS_COMPLETED' and e.team==side for e in self.events)
                    for side in ('left','right')}
        else:
            return self._result('possession_summary',[],('Completed-transfer records and canonical events are unavailable for this recording.',))
        total=sum(counts.values())
        rows=[(side,source,dict(team=side,confirmed_passes=count,
            confirmed_pass_share=count/total if total else None,
            possession_basis='completed_observable_transfers'),None,None)
            for side,count in counts.items() if team is None or side==team]
        if team is None:
            rows.append(('total',source,dict(total_confirmed_passes=total),None,None))
            winner=('left' if counts['left']>counts['right'] else
                    'right' if counts['right']>counts['left'] else 'equal') if total else None
            rows.append(('comparison',source,dict(team_with_more_confirmed_passes=winner,
                comparison_basis='completed_observable_transfers; unconfirmed control excluded',
                pass_count_difference=abs(counts['left']-counts['right']),
                left_pass_rank=1 if counts['left']>=counts['right'] else 2,
                right_pass_rank=1 if counts['right']>=counts['left'] else 2),None,None))
        return self._result('possession_summary',rows,
            ('Confirmed pass share is the team\'s share of completed observable transfers, separate from possession attribution.',))

    def get_possession_summary(self,team=None):
        """Overall possession is the user's confirmed-completed-pass proxy."""
        return self.get_confirmed_pass_share(team)

    def get_attributed_control_summary(self,team=None):
        self._team(team)
        total=len(self.ownership)
        rows=[(side,self.ownership_source,dict(team=side,attributed_seconds=sum(r['team']==side for r in self.ownership)/self.summary.get('fps',25),
            attributed_share=sum(r['team']==side for r in self.ownership)/total if total else None),None,None)
            for side in ('left','right') if team is None or team==side]
        return self._result('possession_summary',rows,('Possession retains the last established controller during gaps up to 20 seconds, then becomes unknown. Completed-pass share is separate.',))

    def get_current_possession(self,timestamp):
        index=bisect_right(self.ownership_times,timestamp)-1
        r=self.ownership[index] if 0<=index<len(self.ownership) and timestamp<=self.ownership_times[-1] else None
        return self._result('possession_summary',[(str(r['frame']),self.ownership_source,r,r['timestamp'],None)] if r else [])

    def get_positions(self,timestamp):
        # Existing sample projection only, no vision or interpolation.
        replay=getattr(self,'replay',None)
        if replay is None:return self._result('tactical_state',[])
        i=min(len(replay.positions)-1,int(timestamp*replay.fps/replay.stride))
        if timestamp<0 or timestamp>replay.frames/replay.fps:return self._result('tactical_state',[])
        rows=[(str(actor),getattr(self,'replay_source','review_frames.json'),dict(actor_id=actor,team=replay.teams[j],x_m=p[0],y_m=p[1]),i*replay.stride/replay.fps,None)
              for j,(actor,p) in enumerate(zip(replay.ids,replay.positions[i]))]
        return self._result('tactical_state',rows,('Precomputed metric positions, not live observations.',))

    def get_pressure_level(self,timestamp):
        from src.tactics.review import pressure_level
        i=bisect_right(self.times,timestamp)-1
        if i<0 or timestamp>=self.summary.get('duration_seconds',0):return self._result('pressure_sequences',[])
        s=self.states[i]
        display=getattr(self,'review_pressure',[])
        replay=getattr(self,'replay',None)
        display_index=int(timestamp*replay.fps/replay.stride) if replay else i
        p=display[display_index] if display_index<len(display) else s.pressure
        return self._result('pressure_sequences',[(str(s.frame),'pressure_levels.json' if (self.directory/'pressure_levels.json').is_file() else getattr(self,'replay_source','tactical_states.json')+'#attributed_possessor_pressure',dict(pressure_level=pressure_level(p),
            team=p.attacking_team,actor_id=p.carrier_actor_id,sustained_press=s.pressure.sustained_press),s.timestamp,None)],
            ('Pressure level describes opponents around the attributed possessor, including during flight. Sustained events use the separate stricter confirmed-carrier detector.',))

    def get_formation_windows(self,team=None):
        self._team(team)
        return self._result('formation_timeline',[(r['team']+str(r['start_seconds']),self.formation_source,r,r['start_seconds'],None)
            for r in self.formations if team is None or r['team']==team],('Dominant formation over four-minute offline windows; inadequate coverage remains unknown.',))

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
        return self.get_formation_windows(team)

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
