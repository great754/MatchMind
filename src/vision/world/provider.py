"""World snapshots -> the existing canonical observation contract."""
from src.observations.schema import RecordingObservation,PlayerObservation,BallObservation,Provenance

def canonical_observation(state,fps,source):
    players=[]
    for e in state.entities:
        visible=e.visibility=='visible'
        players.append(PlayerObservation(actor_id=e.entity_id,track_id=e.track_id,team=e.team,visibility=e.visibility,role=e.role,projection_reason=e.projection_reason,camera=e.projection_camera,
            pitch=e.current_pitch if visible else None,localization_kind=e.position_source if visible and e.current_pitch is not None else 'unknown',image_bbox=e.image_bbox if visible else None,image_foot=e.image_foot if visible else None,
            detection_confidence=e.detection_confidence if visible else None,team_confidence=e.team_confidence,team_evidence=e.team_belief,position_diagnostic=e.position_diagnostic,
            position=Provenance(provider='cv-world-m1',source=source,quality=('propagated' if e.position_source in ('camera_propagated','partial_constrained','temporal_predicted','temporal_corrected','bidirectional_interpolated') else 'inferred') if visible and e.current_pitch is not None else 'unknown',confidence=e.localization_confidence if visible else None),
            identity=Provenance(provider='cv-world-m1',source='anonymous birth / continuity / provisional association; see world.jsonl',quality='inferred',confidence=e.identity_confidence)))
    for u in state.unassigned:
        t=u.observation
        players.append(PlayerObservation(actor_id=None,track_id=t.track_id,team=t.team,visibility='visible',projection_reason=t.projection_reason,camera=t.projection_camera,pitch=t.pitch,localization_kind=state.camera.calibration_source_kind if t.pitch is not None else 'unknown',image_bbox=t.bbox,image_foot=t.foot,
            detection_confidence=t.detection_confidence,team_confidence=t.team_confidence,team_evidence=t.team_belief,
            position=Provenance(provider='cv-world-m1',source=source,quality=('propagated' if state.camera.calibration_source_kind in ('camera_propagated','partial_constrained','temporal_predicted','temporal_corrected','bidirectional_interpolated') else 'inferred') if t.pitch is not None else 'unknown',confidence=t.localization_confidence),
            identity=Provenance(provider='cv-world-m1',source='identity unresolved; hypotheses in world.jsonl',quality='unknown')))
    return RecordingObservation(recording_id=state.recording_id,frame=state.frame,timestamp=state.timestamp,fps=fps,players=players,
        ball=BallObservation(provenance=Provenance(provider='cv-world-m1',source='Ball detection not implemented',quality='unknown')),
        calibration=state.camera.calibration,visible_pitch_region=state.camera.visible_pitch_region,provenance=Provenance(provider='cv-world-m1',source=source,quality='inferred'))

def analytics_recording(observations,team_sides,half):
    """Explicit offline compatibility bridge; never infer kit-to-side or match half.

    Unknown/ambiguous identities cannot fill an existing player's coordinate row.
    No stale/world-predicted positions are ever bridged into football analytics.
    """
    import numpy as np
    from src.analytics.data import Recording
    rows=list(observations)
    if not rows or half not in (1,2):raise ValueError('Recording observations and explicit half required')
    if any(r.frame!=i or r.fps!=rows[0].fps or r.recording_id!=rows[0].recording_id for i,r in enumerate(rows)):raise ValueError('Unaligned observations')
    ids=sorted({p.actor_id for r in rows for p in r.players if p.actor_id})
    index={key:i for i,key in enumerate(ids)};teams={}
    positions=np.full((len(rows),len(ids),2),np.nan);ball=np.full((len(rows),2),np.nan)
    for i,r in enumerate(rows):
        if r.ball.pitch is not None:ball[i]=r.ball.pitch
        for p in r.players:
            if not p.actor_id:continue
            if p.team and p.team not in team_sides:raise ValueError('Explicit kit-to-side mapping missing')
            side=team_sides.get(p.team)
            if side not in ('left','right'):raise ValueError('Unknown defending side; not analytics ready')
            if p.actor_id in teams and teams[p.actor_id]!=side:raise ValueError('Unresolved team conflict')
            teams[p.actor_id]=side
            if p.visibility=='visible' and p.pitch is not None and (p.position.quality=='provided' or (p.localization_kind=='direct_calibrated' and p.position.quality in ('observed','inferred'))):positions[i,index[p.actor_id]]=p.pitch
    return Recording(positions,ball,np.where(np.isfinite(ball).all(axis=1),1,0),np.arange(1,len(ids)+1),[teams[k] for k in ids],{},[],rows[0].fps,0,half,'clip','cv-world-m1',[],{'match_id':rows[0].recording_id,'cv_actor_mapping':{str(i+1):key for i,key in enumerate(ids)}})
