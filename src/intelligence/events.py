"""Adapt existing deterministic analytics; never rerun or replace detectors."""
import math
import pandas as pd
from .schema import EventType as T, EventMetrics, make_event
from .labels import recording_labels,PlayerLabel
from .config import IntelligenceConfig


def finite(value):
    return None if value is None or pd.isna(value) else float(value)


def build_events(recording, result, config: IntelligenceConfig | None = None) -> list:
    config = config or IntelligenceConfig()
    labels = recording_labels(recording)
    key = f"{recording.sync['match_id']}:{recording.half}:{recording.source_offset}"
    events = []
    def identity(actor):
        if actor is None or pd.isna(actor): return dict(player_id=None,actor_id=None)
        actor=int(actor)
        return dict(player_id=labels.get(actor,PlayerLabel(actor)).tracker_id,actor_id=actor,
                    id_namespace='mot' if actor in recording.mot_ids else 'gsr_actor')
    def emit(frame,kind,source,score,group=None,**values):
        events.append(make_event(key,frame=int(frame),fps=recording.fps,type=kind,
            group_id=group or f'{key}:{source}:{frame}',source=source,significance=score,
            observation_kind={'transfer':'observable_transfer','bas':'bas_action','possession':'inferred_control','motion':'estimated_motion','trajectory':'trajectory_candidate'}[source],**values))
    previous=None; previous_team=None; last_known=None; last_known_team=None
    for row in result['possession'].itertuples():
        owner=None if pd.isna(row.possessing_player_id) else int(row.possessing_player_id)
        team=None if pd.isna(row.possessing_team) else row.possessing_team
        if owner != previous:
            prior=previous if previous is not None else last_known
            metrics=EventMetrics(possession_state=row.state,previous_player_id=labels[prior].tracker_id if prior is not None else None,previous_team=previous_team if previous is not None else last_known_team)
            if previous is not None: emit(row.frame,T.POSSESSION_LOST,'possession',20,team=previous_team,metrics=metrics,**identity(previous))
            if owner is not None: emit(row.frame,T.POSSESSION_GAINED,'possession',25,team=team,metrics=metrics,**identity(owner))
            if owner is not None and last_known is not None and owner!=last_known: emit(row.frame,T.POSSESSION_CHANGED,'possession',60,team=team,metrics=metrics,**identity(owner))
        if row.state in ('unknown','out_of_play'): last_known=last_known_team=None
        elif owner is not None: last_known,last_known_team=owner,team
        previous,previous_team=owner,team
    matches={int(r.prediction_index):r for r in result['pass_matches'].itertuples() if r.match_status=='matched'}
    outcomes={'completed':T.PASS_COMPLETED,'intercepted':T.TRANSFER_INTERCEPTED,'unresolved':T.TRANSFER_UNRESOLVED,'out_of_play':T.TRANSFER_OUT_OF_PLAY}
    for index,row in enumerate(result['passes'].itertuples()):
        metrics=EventMetrics(start_frame=int(row.start_frame),end_frame=int(row.end_frame),
            start_x_m=finite(row.start_x_m),start_y_m=finite(row.start_y_m),end_x_m=finite(row.end_x_m),end_y_m=finite(row.end_y_m),
            distance_m=finite(row.distance_m),duration_s=finite(row.duration_seconds),
            ball_speed_kmh=finite(row.ball_speed_peak_kmh),ball_speed_median_kmh=finite(row.ball_speed_median_mps*3.6),
            difficulty_score_0_100=finite(row.difficulty_score_0_100),forward_progression_m=finite(row.forward_progression_m),
            observed_step_fraction=finite(row.observed_step_fraction),outcome=row.outcome)
        receiver=None if pd.isna(row.receiver_id) else int(row.receiver_id)
        match=matches.get(index)
        values=dict(team=row.sender_team,metrics=metrics,receiver_actor_id=receiver,
                    receiver_id=labels.get(receiver,PlayerLabel(receiver)).tracker_id if receiver is not None else None,**identity(row.sender_id))
        if match: values.update(validation_state='bas_time_actor_match',validated_by_bas=True,bas_annotation_id=int(match.annotation_id))
        group=f'{key}:transfer:{row.pass_id}'
        detected_frame=int(row.end_frame)
        if receiver is not None:
            # The detector backdates contact after ownership persistence confirms it.
            owners=result['possession'].possessing_player_id.iloc[detected_frame:]
            confirmed=owners.eq(receiver).fillna(False).to_numpy(dtype=bool).nonzero()[0]
            if len(confirmed): detected_frame+=int(confirmed[0])
        elif row.termination_reason=='unknown':
            detected_frame=min(detected_frame+1,len(recording.ball)-1)
        emit(detected_frame,outcomes[row.outcome],'transfer',55 if row.outcome=='intercepted' else 45 if row.outcome=='completed' else 20,group,**values)
        if row.outcome=='completed':
            for kind,condition,score in ((T.LONG_PASS,(metrics.distance_m or 0)>=config.long_pass_m,60),
                (T.DIFFICULT_PASS,(metrics.difficulty_score_0_100 or 0)>=config.difficult_pass_score,70),
                (T.PROGRESSIVE_PASS,(metrics.forward_progression_m or 0)>=config.progressive_pass_m,55),
                (T.BALL_PROGRESSION,(metrics.forward_progression_m or 0)>=config.significant_progression_m,55)):
                if condition: emit(detected_frame,kind,'transfer',score,group,**values)
    shot_matches={int(r.prediction_index):r for r in result['shot_matches'].itertuples() if r.match_status=='matched'}
    for index,row in enumerate(result['shots'].itertuples()):
        match=shot_matches.get(index)
        validation=dict(validation_state='bas_time_match',validated_by_bas=True,bas_annotation_id=int(match.annotation_id)) if match else {}
        # Publish the speed window only once it has ended, never at its release time.
        emit(row.end_frame,T.SHOT_CANDIDATE,'trajectory',65,
            metrics=EventMetrics(start_frame=int(row.start_frame),end_frame=int(row.end_frame),ball_speed_kmh=finite(row.speed_peak_kmh),ball_speed_median_kmh=finite(row.speed_median_mps*3.6),window_coverage_fraction=finite(row.window_coverage_fraction)),team=None if pd.isna(row.team) else row.team,**identity(row.shooter_id),**validation)
    for row in recording.events:
        kind=T.GOAL if row['label']=='GOAL' else T.SHOT_VALIDATED if row['label']=='SHOT' else T.ACTION_VALIDATED
        emit(row['frame'],kind,'bas',100 if kind==T.GOAL else 85 if kind==T.SHOT_VALIDATED else 30,
             group=f"{key}:bas:{row['annotation_id']}",team=row['team'],**identity(row['player_id']),
             validation_state='bas_reference',validated_by_bas=True,bas_annotation_id=int(row['annotation_id']),metrics=EventMetrics(action_label=row['label']))
    motions=[(int(actor),team,m,config.player_speed_kmh) for actor,team,m in zip(recording.player_ids,recording.player_teams,result['players'])]
    motions.append((None,None,result['ball'],config.ball_speed_kmh))
    for actor,team,motion,threshold in motions:
        streak=0; armed=True; last=-math.inf; milestone=0
        for frame,speed in enumerate(motion.smoothed_speed_mps*3.6):
            above=math.isfinite(speed) and speed>=threshold
            streak=streak+1 if above else 0
            if not above: armed=True
            if armed and streak>=config.threshold_persistence_frames and frame/recording.fps-last>=config.threshold_cooldown_seconds:
                emit(frame,T.PLAYER_SPEED_THRESHOLD if actor is not None else T.BALL_SPEED_THRESHOLD,'motion',45 if actor is not None else 60,
                     group=f'{key}:speed:{actor}:{frame}',team=team,**identity(actor),metrics=EventMetrics(speed_kmh=float(speed),threshold_kmh=threshold))
                last=frame/recording.fps; armed=False
            distance=float(motion.distance_m[frame])
            reached=int(distance/config.distance_milestone_m) if math.isfinite(distance) else milestone
            if actor is not None and reached>milestone:
                emit(frame,T.PLAYER_DISTANCE_MILESTONE,'motion',35,group=f'{key}:distance:{actor}:{reached}',team=team,**identity(actor),metrics=EventMetrics(measured_distance_m=distance,milestone_m=reached*config.distance_milestone_m))
            milestone=max(milestone,reached)
    return sorted(events,key=lambda e:(e.frame,e.event_id))
