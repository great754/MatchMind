"""Slow offline formation windows and continuous pressure display, not event tuning."""
from collections import Counter
from .schema import Formation


def pressure_level(pressure):
    if pressure.defenders_within_local is None or pressure.available_defenders < 8:
        return 'unknown'
    if (pressure.defenders_within_near or 0)>=2 or ((pressure.defenders_within_local or 0)>=2 and (pressure.closing_defenders or 0)>=1):
        return 'high'
    if (pressure.defenders_within_local or 0)>=1 or ((pressure.defenders_within_outer or 0)>=1 and (pressure.closing_defenders or 0)>=1):
        return 'moderate'
    return 'low'


def formation_windows(states, window_seconds=240., min_valid_fraction=.5, min_dominance=.6):
    """Dominant quality-gated estimates per fixed offline window.

    Unknown samples count against coverage. No backfilled forced formation.
    Uses existing clear fits, keeping rapid geometry and causal evidence intact.
    """
    buckets={}
    for s in states:buckets.setdefault(int(s.timestamp//window_seconds),[]).append(s)
    output=[]
    for window,samples in sorted(buckets.items()):
        for team in ('left','right'):
            known=[getattr(s,team).formation.estimate for s in samples
                   if getattr(s,team).formation.estimate not in ('unknown','other') and getattr(s,team).formation.fit_quality=='clear']
            counts=Counter(known);winner,count=counts.most_common(1)[0] if counts else ('unknown',0)
            coverage=len(known)/len(samples);dominance=count/len(known) if known else 0
            adequate=coverage>=min_valid_fraction and dominance>=min_dominance
            output.append(dict(team=team,start_seconds=window*window_seconds,end_seconds=(window+1)*window_seconds,
                estimate=winner if adequate else 'unknown',coverage=coverage,dominance=dominance,
                reason=None if adequate else 'insufficient_coverage' if coverage<min_valid_fraction else 'ambiguous_fit',
                samples=len(samples),window_seconds=window_seconds))
    return output


def review_pressure(replay, ownership):
    """Pressure on the attributed possessor from cached metric review geometry.

    This display layer never enters sustained-press event detection. During ball
    flight it describes opponents around the last possessor, not ball pressure.
    """
    if replay is None:return []
    import numpy as np
    from bisect import bisect_right
    from .pressure import causal_velocities,local_pressure
    from .config import TacticsConfig
    from .schema import Pressure
    config=TacticsConfig();coords=np.asarray(replay.positions,dtype=float)
    times=[r['timestamp'] for r in ownership];teams=np.asarray(replay.teams);slots={v:i for i,v in enumerate(replay.ids)}
    result=[];sample_fps=replay.fps/replay.stride
    for i,current in enumerate(coords):
        j=bisect_right(times,i/sample_fps)-1;r=ownership[j] if j>=0 else None
        owner=r['player_id'] if r else None;team=r['team'] if r else None
        if owner not in slots or team not in ('left','right') or not np.isfinite(np.asarray(replay.ball[i],float)).all():
            result.append(Pressure(reason='no_confirmed_carrier'));continue
        mask=teams==team;velocity=causal_velocities(coords,i,sample_fps,config.closing_lookback_seconds,config.closing_max_mps)
        result.append(local_pressure(current[slots[owner]],velocity[slots[owner]],current[~mask],velocity[~mask],current[mask],config,
            carrier_actor_id=owner,attacking_team=team,defending_team='right' if team=='left' else 'left',expected_defenders=int((~mask).sum())))
    return result
