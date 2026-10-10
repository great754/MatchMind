"""Link important events to a strictly earlier tactical sample, without causal wording."""
from bisect import bisect_left
import pandas as pd
from src.intelligence.schema import EventType as T

IMPORTANT={T.PASS_COMPLETED,T.TRANSFER_INTERCEPTED,T.POSSESSION_CHANGED,T.SHOT_CANDIDATE,T.LONG_PASS,T.DIFFICULT_PASS,T.PROGRESSIVE_PASS,T.BALL_PROGRESSION}


def pre_event_evidence(events,states,passes=None,fps=25) -> list[dict]:
    """Link original release/start times to strictly earlier sampled states."""
    frames=[s.frame for s in states];result=[]
    for event in events:
        if event.type not in IMPORTANT:continue
        start=event.metrics.start_frame if event.metrics.start_frame is not None else event.frame
        index=bisect_left(frames,start)-1
        state=states[index] if index>=0 else None
        evidence=dict(event_id=event.event_id,event_type=event.type.value,event_published_frame=event.frame,event_start_frame=start,
            sample_frame=state.frame if state else None,sample_age_s=(start-state.frame)/fps if state else None,
            pre_event_context=state.model_dump(mode='json') if state else None,
            missing_reason=None if state else 'No strictly earlier tactical sample.',
            interpretation='Observed pre-event geometry, not a causal explanation. Offline ownership inputs may use centered smoothing.')
        # Explicitly separate retrospective lane/realized-target features from prior-state facts.
        if passes is not None and event.source=='transfer' and event.metrics.start_frame is not None:
            matched=passes[(passes.start_frame==event.metrics.start_frame)&(passes.sender_id==event.actor_id)]
            if len(matched):
                row=matched.iloc[0]
                evidence['retrospective_geometry']=dict(uses_realized_endpoint=True,
                    passing_lane_defenders=int(row.passing_lane_defenders) if pd.notna(row.passing_lane_defenders) else None,
                    target_nearest_defender_at_launch_m=float(row.target_nearest_defender_at_launch_m) if pd.notna(row.target_nearest_defender_at_launch_m) else None)
        result.append(evidence)
    return result
