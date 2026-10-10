"""Coverage and stability diagnostics, never possession ground-truth accuracy."""
import numpy as np
import pandas as pd
from .event_evaluation import PASS_CLASSES


def evaluate_possession(table,references,fps,short_seconds=.2,lookback_seconds=1.):
    owners=[None if pd.isna(o) else int(o) for o in table.possessing_player_id]
    runs=[];start=0
    for i in range(1,len(owners)+1):
        if i==len(owners) or owners[i]!=owners[start]:
            if owners[start] is not None:runs.append((start,i,owners[start]))
            start=i
    changes=sum(a is not None and b is not None and a!=b for a,b in zip(owners,owners[1:]))
    acquired=[r[2] for r in runs]
    changes_including_gaps=sum(a!=b for a,b in zip(acquired,acquired[1:]))
    durations=[(end-start)/fps for start,end,_ in runs]
    checks=[]
    for event in references:
        if event['label'] not in PASS_CLASSES:continue
        end=int(event['frame']);lo=max(0,end-int(lookback_seconds*fps))
        prior=[(i,owners[i]) for i in range(lo,end) if owners[i] is not None]
        last=prior[-1] if prior else None
        checks.append(dict(annotation_id=event['annotation_id'],label=event['label'],frame=end,
            possessing_player_id=last[1] if last else None,sample_frame=last[0] if last else None,
            sender_id=event['player_id'],sender_agreement=last[1]==event['player_id'] if last and event['player_id'] is not None else None))
    coverage={state:float((table.state==state).mean()*100) if len(table) else 0. for state in
        ('controlled','contested','free','unknown','out_of_play','candidate')}
    one=sum(end-start==1 for start,end,_ in runs);minutes=len(owners)/fps/60
    warnings=[]
    if coverage['unknown']>30:warnings.append('Unknown state exceeds 30% of frames.')
    if one>max(5,len(runs)*.1):warnings.append('Excessive single-frame ownership episodes.')
    if minutes and changes_including_gaps/minutes>30:warnings.append('More than 30 observed owner changes per minute including gaps.')
    agreements=[c['sender_agreement'] for c in checks if c['sender_agreement'] is not None]
    return dict(kind='sanity_not_ground_truth_accuracy',state_percent=coverage,ownership_episodes=len(runs),
        immediate_owner_changes=changes,possession_changes_including_gaps=changes_including_gaps,
        median_possession_duration_seconds=float(np.median(durations)) if durations else None,
        very_short_possession_count=sum(d<short_seconds for d in durations),one_frame_possession_count=one,
        bas_pre_event_checks=checks,bas_pass_like_count=len(checks),bas_with_prior_owner=sum(c['sample_frame'] is not None for c in checks),
        sender_agreement_fraction=sum(agreements)/len(agreements) if agreements else None,
        config=dict(short_seconds=short_seconds,lookback_seconds=lookback_seconds,unknown_warning_percent=30,changes_warning_per_minute=30),
        warnings=warnings,precision=None,recall=None)
