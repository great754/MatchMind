"""Windowed facts at arbitrary clip-local timestamps; missing observations stay missing."""
from bisect import bisect_left, bisect_right
import math
import numpy as np
import pandas as pd
from .labels import recording_labels
from .schema import EventType as T


class ContextBuilder:
    def __init__(self,recording,result,events,tactical_states=None):
        self.recording,self.result,self.events=recording,result,events
        self.frames=[e.frame for e in events]
        self.labels=recording_labels(recording)
        self.tactical_states=tactical_states or []
        self.tactical_frames=[s.frame for s in self.tactical_states]
    def at(self,timestamp: float,window: int = 30) -> dict:
        r=self.recording
        if window not in (15,30,60) or not math.isfinite(timestamp) or not 0<=timestamp< len(r.ball)/r.fps:
            raise ValueError('Use a valid recording timestamp and a 15, 30 or 60 second window')
        end=min(len(r.ball)-1,int(math.floor(timestamp*r.fps+1e-8)))
        start=max(0,int(math.floor((timestamp-window)*r.fps+1e-8))+1) if timestamp>=window else 0
        p=self.result['possession'].iloc[start:end+1]; current=p.iloc[-1]
        owner=None if pd.isna(current.possessing_player_id) else int(current.possessing_player_id)
        streak=0
        if owner is not None:
            for value in reversed(self.result['possession'].possessing_player_id.iloc[:end+1]):
                if pd.isna(value) or int(value)!=owner: break
                streak+=1
        local=self.events[bisect_left(self.frames,start):bisect_right(self.frames,end)]
        transfers=[e for e in local if e.type in (T.PASS_COMPLETED,T.TRANSFER_INTERCEPTED,T.TRANSFER_UNRESOLVED,T.TRANSFER_OUT_OF_PLAY)]
        sequence=[]; sequence_team=None
        for event in transfers:
            if event.type!=T.PASS_COMPLETED: sequence=[];sequence_team=None;continue
            if sequence_team!=event.team or not sequence or sequence[-1]!=event.player_id:
                sequence=[event.player_id]
            sequence.append(event.receiver_id);sequence_team=event.team
        fastest=[]
        for actor,team,m in zip(r.player_ids,r.player_teams,self.result['players']):
            speeds=m.smoothed_speed_mps[start:end+1]; valid=np.isfinite(speeds)
            if valid.any(): fastest.append(dict(player_id=self.labels[int(actor)].tracker_id,team=team,peak_speed_kmh=float(speeds[valid].max()*3.6),valid_frames=int(valid.sum())))
        fastest.sort(key=lambda x:(-x['peak_speed_kmh'],x['player_id']))
        ball=self.result['ball']; segment=ball.segment_id[start:end+1]
        displacement=None
        if np.isfinite(ball.positions[start:end+1]).all() and len(set(segment.tolist()))==1:
            displacement=float(np.linalg.norm(ball.positions[end]-ball.positions[start]))
        tactical_index=bisect_right(self.tactical_frames,end)-1
        tactical=self.tactical_states[tactical_index] if tactical_index>=0 else None
        return dict(tactical_state=tactical.model_dump(mode='json') if tactical else None,
            recent_tactical_events=[e.model_dump(mode='json',exclude_none=True) for e in local if e.source=='tactical_state_engine'][-12:],schema_version='1.0',timestamp=end/r.fps,frame=end,window_seconds=window,start_frame=start,end_frame=end,
            sampled_seconds=len(p)/r.fps,frames=len(p),
            possession_percent={team:float((p.possessing_team==team).sum()/len(p)*100) for team in ('left','right')},
            unknown_percent=float(p.possessing_team.isna().sum()/len(p)*100),
            current_possession=dict(player_id=self.labels[owner].tracker_id if owner is not None else None,
                team=None if pd.isna(current.possessing_team) else current.possessing_team,state=current.state,
                streak_seconds=streak/r.fps,confidence=None if pd.isna(current.control_confidence) else float(current.control_confidence)),
            completed_passes={team:sum(e.type==T.PASS_COMPLETED and e.team==team for e in transfers) for team in ('left','right')},
            interceptions={team:sum(e.type==T.TRANSFER_INTERCEPTED and e.team==team for e in transfers) for team in ('left','right')},
            unsuccessful_transfers={team:sum(e.type!=T.PASS_COMPLETED and e.team==team for e in transfers) for team in ('left','right')},
            recent_pass_sequence=sequence,sequence_team=sequence_team,transfer_count=len(transfers),
            recent_transfers=[e.model_dump(mode='json',exclude_none=True) for e in transfers[-12:]],
            fastest_players=fastest[:3],ball_displacement_m=displacement,
            completed_pass_progression_m={team:sum(e.metrics.forward_progression_m or 0 for e in transfers if e.type==T.PASS_COMPLETED and e.team==team) for team in ('left','right')},
            validated_actions=[e.model_dump(mode='json',exclude_none=True) for e in local if e.source=='bas'][-12:],
            recent_shot_candidates=[e.model_dump(mode='json',exclude_none=True) for e in local if e.type==T.SHOT_CANDIDATE][-12:],
            notable_events=[e.model_dump(mode='json',exclude_none=True) for e in local if e.significance>=50][-20:],
            notable_event_ids=[e.event_id for e in local if e.significance>=50][-20:],
            limitations=['Offline centered smoothing uses later samples; windows exclude later event outcomes.',
                'Ball movement and speed are interpolated 2D estimates; height and strike speed are unavailable.',
                'Possession and passes are heuristic; BAS agreement is not independent validation.'])
