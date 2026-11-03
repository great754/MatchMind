"""M7 offline two-channel measured field localization and downstream projection.

Original-luminance and contrast-normalized feature tracks use identical residual
and uncertainty gates. Prefer original-luminance trajectory; recover only its
unknown frames from a validated contrast route. The routes are correlated image
evidence, NOT two independent metric cameras. Every selection is auditable.
"""
import argparse,json,time
from pathlib import Path
from collections import Counter
from .temporal_review import run
from .schema import WorldState
from .provider import canonical_observation
from .review import digest
from .temporal_camera import transform,disagreement


def process(video,baseline,poses,output):
 output=Path(output)
 if output.exists():raise ValueError('Choose a fresh output; snapshots are immutable')
 output.mkdir(parents=True);started=time.perf_counter();reports={}
 for mode in ('sparse','dense'):reports[mode]=run(video,baseline,poses,output/'routes'/mode,'m7',mode)
 rows={mode:[WorldState.model_validate_json(l) for l in (output/'routes'/mode/'frames.jsonl').open()] for mode in reports};m=json.loads((output/'routes/sparse/manifest.json').read_text());counts=Counter();offsets=[];timeline=[];selections=[];transitions=[];last_status=None
 with (output/'frames.jsonl').open('wb') as target,(output/'observations.jsonl').open('w') as obs:
  for i,(a,b) in enumerate(zip(rows['sparse'],rows['dense'],strict=True)):
   chosen=a;mode='sparse';delta=None
   if not a.camera.calibration_valid and b.camera.calibration_valid:chosen=b;mode='dense'
   if a.camera.calibration_valid and b.camera.calibration_valid:delta=disagreement(transform(a.camera),transform(b.camera),m['width'],m['height'])
   assert a.frame==b.frame==i and a.recording_id==b.recording_id
   # Team and image-space people are frozen, regardless of selected camera.
   assert [(e.entity_id,e.image_bbox,e.image_foot,e.team,e.team_belief) for e in a.entities]==[(e.entity_id,e.image_bbox,e.image_foot,e.team,e.team_belief) for e in b.entities]
   c=chosen.camera;d=c.motion_evidence['temporal_camera'];d['measurement_route']={'selected':mode,'sparse_valid':a.camera.calibration_valid,'dense_valid':b.camera.calibration_valid,'route_disagreement_p90_m':delta,'selection':'original luminance preferred; contrast route may fill unknown; both must independently pass fixed validation gates','independence':'correlated image evidence; not independent metric accuracy'}
   offsets.append(target.tell());target.write((chosen.model_dump_json()+'\n').encode());obs.write(canonical_observation(chosen,m['fps'],str(Path(video).resolve())).model_dump_json()+'\n')
   loc=c.motion_evidence['localization'];state=loc['status'];counts['camera_'+state]+=1;counts['camera_accepted']+=c.calibration_valid;counts['route_'+mode]+=1
   for e in chosen.entities:
    if e.visibility=='visible':counts['players_eligible']+=1;counts['players_projected']+=e.current_pitch is not None;counts['team_'+str(e.team)]+=1;counts['projection_'+e.projection_reason]+=1
   for u in chosen.unassigned:counts['players_eligible']+=1;counts['players_projected']+=u.observation.pitch is not None;counts['team_'+str(u.observation.team)]+=1;counts['projection_'+u.observation.projection_reason]+=1
   timeline.append({'frame':i,'timestamp':chosen.timestamp,'state':state,'cut':c.continuity=='cut','reason':loc['outcome'],'last_direct_age_seconds':loc.get('last_absolute_age_seconds'),'support':loc.get('support_score'),'landmarks':loc['semantic_classes']});selections.append({'frame':i,**d['measurement_route']})
   if state!=last_status:transitions.append({'frame':i,'timestamp':chosen.timestamp,'label':'Calibration lost' if state=='UNKNOWN' else 'Calibration valid' if state in ('DIRECT','RELOCALIZED') else 'Camera propagated','transition':loc['transition'],'status':state});last_status=state
 result={**m,'frame_offsets':offsets,'frames_sha256':digest(output/'frames.jsonl'),'camera_timeline':timeline,'transitions':transitions,'camera_model_version':'m7','provenance':'Experimental M7 persistent world-field tracking, measured temporal windows and explicit two-channel selection; immutable M6 people/teams; no football analytics'}
 (output/'manifest.json').write_text(json.dumps(result));report={'classification':'EXPERIMENTAL / LOCALLY VERIFIED; independent real metric accuracy unavailable','frames':m['frames'],'metrics':dict(counts),'camera_coverage_percent':100*counts['camera_accepted']/m['frames'],'player_placement_percent':100*counts['players_projected']/counts['players_eligible'],'routes':reports,'route_selections':selections,'wall_seconds':time.perf_counter()-started,'source_baseline_frames_sha256':m['source_baseline_frames_sha256'],'source_video_sha256':m['video_sha256'],'original_luminance_preferred':True,'teams_and_identity_frozen':True}
 (output/'diagnostics.json').write_text(json.dumps(report,indent=2,allow_nan=False));print(json.dumps({k:v for k,v in report.items() if k not in ('routes','route_selections')},indent=2));return report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--video',required=True);p.add_argument('--baseline',required=True);p.add_argument('--poses',required=True);p.add_argument('--output',required=True);a=p.parse_args();process(a.video,a.baseline,a.poses,a.output)
