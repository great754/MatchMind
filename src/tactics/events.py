"""Persistence and one-shot latches; missing evidence resets continuity."""
from src.intelligence.schema import EventType,make_event,TacticalMetrics

class PersistentState:
    def __init__(self,seconds,invalidate_pending=False):
        self.invalidate_pending=invalidate_pending;self.seconds=seconds;self.candidate=None;self.since=None;self.value='unknown'
    def update(self,candidate,frame,fps):
        if candidate=='unknown' or candidate is None:
            self.candidate=None;self.since=None;self.value='unknown';return self.value,False,0.
        if candidate!=self.candidate:self.candidate=candidate;self.since=frame
        duration=(frame-self.since)/fps
        changed=False
        if duration+1e-8>=self.seconds and self.value!=candidate:self.value=candidate;changed=True
        # Formation fits can invalidate old evidence; block states retain hysteresis until confirmation.
        if self.invalidate_pending and duration+1e-8<self.seconds:self.value='unknown'
        return self.value,changed,duration

class PersistentTrigger:
    def __init__(self,seconds,cooldown):
        self.seconds=seconds;self.cooldown=cooldown;self.since=None;self.latched=False;self.last=-float('inf')
    def update(self,condition,frame,fps):
        if not condition:self.since=None;self.latched=False;return False,0.
        if self.since is None:self.since=frame
        duration=(frame-self.since)/fps
        fire=not self.latched and duration+1e-8>=self.seconds and frame/fps-self.last>=self.cooldown
        if fire:self.latched=True;self.last=frame/fps
        return fire,duration


def tactical_event(recording,frame,kind,team,metrics,player_id=None,actor_id=None,receiver_id=None,group_suffix=''):
    score={EventType.HIGH_PRESSURE_SEQUENCE:70,EventType.FORMATION_ESTIMATE_CHANGED:60,EventType.TACTICAL_BLOCK_CHANGED:55,
           EventType.TEAM_WIDTH_INCREASED:55,EventType.TEAM_COMPACTNESS_CHANGED:55,EventType.TRANSFER_NETWORK_PATTERN:50}[kind]
    key=f"{recording.sync['match_id']}:{recording.half}:{recording.source_offset}"
    return make_event(key,frame=frame,fps=recording.fps,type=kind,source='tactical_state_engine',significance=score,
        group_id=f'{key}:tactics:{kind.value}:{team}:{frame}:{group_suffix}',observation_kind='tactical_state',
        team=team,player_id=player_id,actor_id=actor_id,receiver_id=receiver_id,
        id_namespace='mot' if (actor_id in recording.mot_ids if actor_id is not None else player_id is None or player_id in recording.mot_ids.values()) else 'gsr_actor',
        tactical=TacticalMetrics(**metrics))
