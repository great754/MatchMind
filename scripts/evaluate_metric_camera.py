"""Validate M8 camera/footpoint/origin semantics and compare every M7 frame."""
import argparse,json,itertools
from pathlib import Path
from collections import Counter
import numpy as np
from src.vision.world.schema import WorldState
from src.observations.schema import RecordingObservation
from src.serving.cv import CVReview
from src.vision.world.metric_motion import project
from scripts.evaluate_cv_temporal import pair_displacements,numbers,run_stats

def evaluate(before,after,output):
 before,after,output=map(Path,(before,after,output));output.mkdir(parents=True,exist_ok=True);old=CVReview(before);new=CVReview(after);m=new.manifest;assert m['source_baseline_frames_sha256']==old.manifest['frames_sha256'] and m['video_sha256']==old.manifest['video_sha256'];a=[WorldState.model_validate_json(l) for l in (before/'frames.jsonl').open()];b=[WorldState.model_validate_json(l) for l in (after/'frames.jsonl').open()];assert len(a)==len(b)==m['frames'];states=Counter();placements=Counter();teams=Counter();newframes=[];lost=[];people=0;projected=0;proofs=Counter();telemetry=[];maxfooterror=0.
 with (after/'observations.jsonl').open() as observations:
  for i,(x,y,canonical) in enumerate(zip(a,b,observations,strict=True)):
   obs=RecordingObservation.model_validate_json(canonical);assert obs.frame==y.frame==x.frame==i and obs.recording_id==x.recording_id==y.recording_id;assert len(x.entities)==len(y.entities) and len(x.unassigned)==len(y.unassigned)
   if x.camera.calibration_valid:
    assert y.camera.calibration_valid;np.testing.assert_allclose(x.camera.calibration.image_to_pitch,y.camera.calibration.image_to_pitch,rtol=0,atol=1e-10)
   if not x.camera.calibration_valid and y.camera.calibration_valid:newframes.append(i)
   if x.camera.calibration_valid and not y.camera.calibration_valid:lost.append(i)
   c=y.camera;metric=c.motion_evidence['metric_camera'];states[metric['state']]+=1
   if not c.calibration_valid:assert metric['current_transform'] is None
   else:
    np.testing.assert_allclose(np.array(metric['current_transform']).ravel(),c.calibration.image_to_pitch)
    if i and b[i-1].camera.calibration_valid and b[i-1].camera.epoch==c.epoch:
     q=[[m['width']/2,m['height']/2]];delta=project(np.array(c.calibration.image_to_pitch).reshape(3,3),q)-project(np.array(b[i-1].camera.calibration.image_to_pitch).reshape(3,3),q);np.testing.assert_allclose(metric['incremental_view_shift_m'],delta[0],atol=1e-8)
   for e,f in zip(x.entities,y.entities,strict=True):
    for key in ['entity_id','track_id','image_bbox','image_foot','visibility','identity_status','identity_confidence','team','team_confidence','team_belief','team_memory','current_kit','history','track_history']:assert getattr(e,key)==getattr(f,key),(i,key)
    if f.visibility!='visible':assert f.current_pitch is None
   for e,f in zip(x.unassigned,y.unassigned,strict=True):
    for key in ['track_id','bbox','foot','team','team_confidence','team_belief','kit']:assert getattr(e.observation,key)==getattr(f.observation,key),(i,key)
   for p in obs.players:
    if p.visibility=='visible':people+=1;teams[p.team or 'unknown']+=1
    if p.pitch is not None:
     projected+=1;assert c.calibration_valid and p.image_foot is not None and p.camera is not None;computed=project(np.array(c.calibration.image_to_pitch).reshape(3,3),[p.image_foot])[0];error=float(np.linalg.norm(computed-p.pitch));maxfooterror=max(maxfooterror,error);assert error<1e-6;placements[p.camera.placement_provenance]+=1;assert p.camera.placement_provenance!='UNKNOWN'
    elif p.camera is not None:assert p.camera.placement_provenance=='UNKNOWN'
   proof=c.motion_evidence['temporal_camera'].get('independent_anchor_proof')
   if proof:
    proofs[proof['kind']]+=1
    if proof['kind']=='M8_CURRENT_METRIC_LANDMARK':
     record=c.motion_evidence['temporal_camera'].get('m8_independent_anchor') or b[proof['frame']].camera.motion_evidence['temporal_camera'].get('landmark_m8');assert record and record['accepted'];np.testing.assert_allclose(np.array(proof['transform']).reshape(3,3),record['selected_transform'])
   telemetry.append({'frame':i,'timestamp':y.timestamp,'state':metric['state'],'accepted':c.calibration_valid,'reason':c.source,'metric':metric,'landmark_attempt':c.motion_evidence['temporal_camera'].get('landmark_m8'),'field_motion':c.motion_evidence['temporal_camera']['field_motion'],'footpoints':[{'id':e.entity_id,'image_foot':e.image_foot,'pitch':e.current_pitch,'projection':e.projection_reason} for e in y.entities if e.visibility=='visible']+[{'id':u.observation.track_id,'image_foot':u.observation.foot,'pitch':u.observation.pitch,'projection':u.observation.projection_reason} for u in y.unassigned]})
 assert people==sum(sum(e.visibility=='visible' for e in x.entities)+len(x.unassigned) for x in a)
 before_pairs=pair_displacements(a);after_pairs=pair_displacements(b);common=sorted(before_pairs.keys()&after_pairs.keys());threshold=12/m['fps']+.5;diag=json.loads((after/'diagnostics.json').read_text())
 report={'recording_id':m['recording_id'],'baseline_frames_sha256':old.manifest['frames_sha256'],'frames_sha256':m['frames_sha256'],'video_sha256':m['video_sha256'],'frames':len(b),'before_camera':sum(x.camera.calibration_valid for x in a),'camera':sum(x.camera.calibration_valid for x in b),'camera_percent':100*sum(x.camera.calibration_valid for x in b)/len(b),'people':people,'projected':projected,'placement_percent':100*projected/people,'states':dict(states),'placement_provenance':dict(placements),'newly_recovered':newframes,'previously_accepted_lost':lost,'before_runs':run_stats(a,m['fps']),'after_runs':run_stats(b,m['fps']),'matched_before':numbers([before_pairs[k] for k in common],threshold),'matched_after':numbers([after_pairs[k] for k in common],threshold),'all_pairs_before':numbers(list(before_pairs.values()),threshold),'all_pairs_after':numbers(list(after_pairs.values()),threshold),'teams':dict(teams),'team_and_identity_frozen':True,'maximum_foot_projection_error_m':maxfooterror,'origin_proofs':dict(proofs),'timing':diag['timing'],'wall_seconds':diag['wall_seconds'],'cuts':sum(x.camera.continuity=='cut' for x in b),'epochs':sorted({x.camera.epoch for x in b}),'landmark_relocalizations':[x['frame'] for x in diag['temporal_camera']['landmark_candidates'] if x['accepted']],'landmark_kinds':dict(Counter(x['kind'] for x in diag['temporal_camera']['landmark_candidates'] if x['accepted'])),'keyframe_recoveries':[x['frame'] for x in diag['temporal_camera'].get('keyframe_remeasurement_attempts',[]) if x['accepted']],'accuracy':'Dependent diagnostics only; independent real metric accuracy unavailable'}
 (output/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False));(output/'timeline.jsonl').write_text(''.join(json.dumps(x,allow_nan=False)+'\n' for x in telemetry));(output/'center_window.jsonl').write_text(''.join(json.dumps(x,allow_nan=False)+'\n' for x in telemetry if 151<=x['frame']<=455));print(json.dumps({k:v for k,v in report.items() if k not in ['newly_recovered','landmark_relocalizations','keyframe_recoveries','before_runs','after_runs']},indent=2));return report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--before',required=True);p.add_argument('--after',required=True);p.add_argument('--output',required=True);a=p.parse_args();evaluate(a.before,a.after,a.output)
