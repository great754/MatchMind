"""Independent kit samples accumulate sticky, explicitly uncalibrated team belief.

Replay uses frozen entity/track associations. This module never establishes ReID.
"""
from dataclasses import dataclass,field

MIN_SAMPLE_SECONDS=.08
MIN_QUALITY=.35
MIN_MARGIN=.6


def observation_reason(kit,observed,identity_resolved,margin=None):
    if kit is None or kit.lab is None:return 'TEAM_UNKNOWN_NO_EVIDENCE'
    if kit.quality<MIN_QUALITY or kit.foreground_fraction<.2 or kit.field_context<.3:return 'TEAM_UNKNOWN_LOW_QUALITY_CROP'
    if observed is None or (margin or 0)<MIN_MARGIN:return 'TEAM_UNKNOWN_AMBIGUOUS_COLOR'
    return 'TEAM_UNKNOWN_CONFLICTING_HISTORY' if identity_resolved else 'TEAM_UNKNOWN_IDENTITY_UNRESOLVED'


@dataclass
class TeamBelief:
    team: str|None=None
    a: float=0.
    b: float=0.
    samples: int=0
    unknown: int=0
    candidate: str|None=None
    streak: int=0
    candidate_start: float|None=None
    last_sample: float|None=None
    last_strong: float|None=None
    history: list=field(default_factory=list)
    switches: list=field(default_factory=list)
    def update(self,observed,margin,kit,frame,timestamp,identity_resolved=True):
        reason=observation_reason(kit,observed,identity_resolved,margin);quality=kit.quality if kit is not None else None
        strong=bool(observed and kit is not None and kit.lab is not None and kit.quality>=MIN_QUALITY and kit.foreground_fraction>=.2 and kit.field_context>=.3 and (margin or 0)>=MIN_MARGIN)
        independent=strong and (self.last_sample is None or timestamp-self.last_sample>=MIN_SAMPLE_SECONDS-1e-8)
        event=None
        if independent:
            self.last_sample=timestamp;self.last_strong=timestamp;self.samples+=1;weight=float(margin)*kit.quality;self.a*=.98;self.b*=.98
            if observed=='team_a':self.a+=weight
            else:self.b+=weight
            if self.team and observed!=self.team:
                if self.candidate!=observed:self.candidate=observed;self.streak=0;self.candidate_start=timestamp
                self.streak+=1
            elif observed==self.team:self.candidate=None;self.streak=0;self.candidate_start=None
            winner='team_a' if self.a>=self.b else 'team_b';win=max(self.a,self.b);ratio=win/max(self.a+self.b,1e-9);old=self.team
            if self.team is None and self.samples>=3 and win>=1.2 and ratio>=.8:self.team=winner
            elif self.team and self.candidate==winner and self.streak>=6 and timestamp-self.candidate_start>=.4 and win>=1.2 and ratio>=.8:self.team=winner
            self.history=(self.history+[{'frame':frame,'timestamp':timestamp,'team':observed,'quality':quality,'margin':margin,'independent':True}])[-32:]
            if old!=self.team:
                event={'frame':frame,'timestamp':timestamp,'kind':'team_acquired' if old is None else 'team_changed','old':old,'new':self.team,'supporting_observations':self.streak if old else self.samples,'votes_before_current':{'team_a':self.a-(weight if observed=='team_a' else 0),'team_b':self.b-(weight if observed=='team_b' else 0)},'votes_after':{'team_a':self.a,'team_b':self.b},'candidate_span_seconds':timestamp-self.candidate_start if self.candidate_start is not None else 0,'quality_min_recent':min(s['quality'] for s in self.history[-max(1,self.streak):]),'reason':'Sustained independent high-quality contradictory kit evidence' if old else 'Multiple independent high-quality kit observations; anonymous team'}
                self.switches.append(event);self.candidate=None;self.streak=0;self.candidate_start=None
        elif not strong:self.unknown+=1
        support=(self.a if self.team=='team_a' else self.b)/max(self.a+self.b,1e-9) if self.team else None
        source='current_observation' if self.team and strong and observed==self.team else 'temporal_memory' if self.team else 'unknown'
        if self.team is None and self.a>0 and self.b>0:reason='TEAM_UNKNOWN_CONFLICTING_HISTORY'
        return {'current_team':self.team,'support':support,'source':source,'current_observation':observed,'current_quality':quality,'current_strong':strong,'independent_sample':bool(independent),'unknown_reason':reason if self.team is None else None,'current_observation_reason':reason if not strong else None,'evidence_a':self.a,'evidence_b':self.b,'independent_samples':self.samples,'uncertain_observations':self.unknown,'candidate_team':self.candidate,'candidate_streak':self.streak,'last_strong_observation':self.last_strong,'identity_resolved':identity_resolved,'history':list(self.history),'latest_switch':event},event


def kit_margin(kit):
    if kit is None or kit.cluster_distances is None:return None
    a,b=sorted(kit.cluster_distances);return (b-a)/max(b,1.)
