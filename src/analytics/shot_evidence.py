"""Tracking-only shot gates. BAS labels are deliberately absent from this interface."""
from dataclasses import dataclass
import math
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class ShotOptions:
    forward_velocity_fraction:float=.6
    recent_possession_seconds:float=1.
    require_recent_possession:bool=False
    require_goal_intersection:bool=True
    min_acceleration_mps2:float=0.
    direction_mode:str='context_or_velocity'

    def __post_init__(self):
        if not 0<self.forward_velocity_fraction<=1:raise ValueError('Forward fraction must be in (0,1]')
        if not math.isfinite(self.recent_possession_seconds) or self.recent_possession_seconds<0:raise ValueError('Invalid possession lookback')
        if not math.isfinite(self.min_acceleration_mps2) or self.min_acceleration_mps2<0:raise ValueError('Invalid acceleration threshold')
        if self.direction_mode not in ('context_or_velocity','velocity_only'):raise ValueError('Invalid direction mode')
        if not isinstance(self.require_recent_possession,bool) or not isinstance(self.require_goal_intersection,bool):raise ValueError('Shot gates must be boolean')


def trace_shot_frames(possession,ball,config,options=None):
    """Return each gate independently plus a documented first-failing gate.

    Flags reproduce the original detector at default options. Direction can be
    inferred from velocity when ownership is unavailable; this is disclosed.
    """
    options=options or ShotOptions();rows=[];last_actor=None;last_team=None;last_frame=-100000
    for r in possession.itertuples(index=False):
        i=int(r.frame)
        if r.state in ('unknown','out_of_play'):last_actor=last_team=None
        if not pd.isna(r.possessing_player_id):last_actor,last_team,last_frame=int(r.possessing_player_id),r.possessing_team,i
        actor=last_actor if i-last_frame<=options.recent_possession_seconds*config.fps else None
        team=last_team if actor is not None and last_team in ('left','right') else None
        p=ball.positions[i];v=ball.velocity[i];speed=ball.smoothed_speed_mps[i]
        finite=bool(np.isfinite([*p,*v,speed]).all())
        direction=(1 if v[0]>0 else -1) if finite else None
        attack=config.attack_direction(team) if team is not None else None
        goal=105. if direction==1 else 0. if direction==-1 else None
        dx=(goal-p[0])*direction if finite else None
        t=(goal-p[0])/v[0] if finite and v[0]!=0 else None
        y=p[1]+t*v[1] if t is not None else None
        forward=abs(v[0])/speed if finite and speed>0 else None
        acceleration=float(np.linalg.norm(v-ball.velocity[i-1])*config.fps) if i>0 and finite and np.isfinite(ball.velocity[i-1]).all() else None
        gates={
            'BALL_DATA_MISSING':not finite,
            'NO_INDEPENDENT_FLIGHT':r.state not in ('free','candidate','contested'),
            'POSSESSION_DIRECTION_UNKNOWN':options.require_recent_possession and team is None,
            'SPEED_TOO_LOW':not finite or speed<config.shot_min_speed_mps,
            'WRONG_DIRECTION':options.direction_mode!='velocity_only' and attack is not None and attack!=direction,
            'TOO_FAR_FROM_GOAL':dx is None or not 0<dx<=config.shot_max_goal_distance_m,
            'FORWARD_VELOCITY_TOO_LOW':forward is None or forward<options.forward_velocity_fraction,
            'TIME_TO_GOAL_TOO_LONG':t is None or not 0<t<=config.shot_max_time_to_goal_seconds,
            'GOAL_MOUTH_PROJECTION_FAILED':options.require_goal_intersection and (y is None or abs(y-34)>3.66+config.shot_goal_margin_m),
            'ACCELERATION_TOO_LOW':options.min_acceleration_mps2>0 and (acceleration is None or acceleration<options.min_acceleration_mps2)}
        failures=[name for name,failed in gates.items() if failed]
        rows.append(dict(frame=i,state=r.state,recent_possessing_player=actor,recent_possessing_team=team,
            attacking_direction=attack,trajectory_direction=direction,direction_source='recent_possession' if attack is not None else 'velocity_fallback' if finite else 'unknown',
            start_x=float(p[0]) if np.isfinite(p[0]) else None,start_y=float(p[1]) if np.isfinite(p[1]) else None,
            ball_speed=float(speed) if np.isfinite(speed) else None,vx=float(v[0]) if np.isfinite(v[0]) else None,vy=float(v[1]) if np.isfinite(v[1]) else None,
            distance_to_goal=float(dx) if dx is not None else None,goal_x=goal,projected_goal_crossing_y=float(y) if y is not None else None,
            time_to_goal=float(t) if t is not None else None,forward_velocity_fraction=float(forward) if forward is not None else None,
            acceleration_mps2=acceleration,failures=failures,primary_gate=failures[0] if failures else 'ELIGIBLE_FRAME',eligible=not failures))
    return rows
