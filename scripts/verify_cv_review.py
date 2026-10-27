"""Validate real cached world/canonical histories without inference or cloud writes."""
import argparse,json
from pathlib import Path
from src.vision.world.schema import WorldState
from src.observations.schema import RecordingObservation
from src.serving.cv import CVReview,digest

def verify(directory,output):
    root=Path(directory);review=CVReview(root);m=review.manifest;previous={};count=0;max_age=0;failure=None
    with (root/'frames.jsonl').open() as states,(root/'observations.jsonl').open() as observations:
        for i,(line,observation) in enumerate(zip(states,observations,strict=True)):
            state=WorldState.model_validate_json(line);row=RecordingObservation.model_validate_json(observation)
            assert state.frame==row.frame==i and abs(state.timestamp-i/m['fps'])<1e-6
            current=[e.track_id for e in state.entities if e.visibility=='visible'];assert len(current)==len(set(current))
            for e in state.entities:
                if e.visibility!='visible' or not state.camera.calibration_valid:assert e.current_pitch is None
                if e.position_source=='camera_propagated':
                    assert e.current_pitch is not None
                    if e.entity_id in previous:
                        assert e.last_reliable_pitch==previous[e.entity_id].last_reliable_pitch and e.last_localized_at==previous[e.entity_id].last_localized_at
                    else:assert e.last_reliable_pitch is None and e.last_localized_at is None
            for p in row.players:
                if p.localization_kind=='camera_propagated':assert p.position.quality=='propagated'
                if p.localization_kind=='direct_calibrated':assert p.position.quality=='inferred'
            if state.camera.calibration_valid and state.camera.calibration_source_kind=='camera_propagated':
                age=state.camera.motion_evidence['propagation_age_seconds'];assert age<=.25;max_age=max(max_age,age)
            if m['recording_id']=='test_match' and i==502:
                e=next(e for e in state.entities if e.entity_id=='ENTITY_00005');assert e.current_pitch is None and e.visibility=='off_camera';failure=e.model_dump(include={'entity_id','visibility','current_pitch','track_id'})
            previous={e.entity_id:e for e in state.entities};count+=1
    assert count==m['frames'];assert digest(review.video)==m['video_sha256']
    result={'typed_world_and_canonical_frames_validated':count,'unknown_off_camera_and_propagated_semantics':True,'propagated_never_updates_last_reliable':True,
        'maximum_measured_propagation_age_seconds':max_age,'original_failure_still_rejected_at_frame502':failure,'original_source_sha256':m['video_sha256'],'source_preserved':True}
    Path(output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--output',required=True);a=p.parse_args();verify(a.directory,a.output)
