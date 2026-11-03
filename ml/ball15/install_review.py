"""Validate additive M15 streams and prove original M8/M9 world fields remain identical."""
import json,os
from pathlib import Path
from src.serving.cv import CVReview
from src.vision.world.schema import WorldState
from .common import ROOT,save,verify

def run():
 verify();registry={};report={}
 for name in ['test_match','team_goal','southampton']:
  base=Path(f'outputs/world_model/{name}_m9_ready');sidecar=Path(f'outputs/world_model/ball_m15/{name}');manifest_path=sidecar/'manifest.json';manifest=json.loads(manifest_path.read_text());manifest['pipeline_components']={'motion':json.loads((ROOT/'motion_lock.json').read_text())['selected'],'learned_verifier_enabled':json.loads((ROOT/'pipeline_lock.json').read_text())['verification_enabled'],'local_search_enabled':json.loads((ROOT/'search_lock.json').read_text())['local_enabled'],'independent_ground_validation':False};manifest_path.write_text(json.dumps(manifest,indent=2)+'\n');review=CVReview(base);review.attach_ball(sidecar)
  if review.ball_review is None:raise ValueError('M15 validation failed: '+name)
  cameras=0
  for i,line in enumerate((base/'frames.jsonl').open()):
   old=WorldState.model_validate_json(line);joined=review.frame((i+.001)/review.manifest['fps'])
   if old.model_dump(exclude={'ball_world'})!=joined.model_dump(exclude={'ball_world'}):raise ValueError('Existing world fields changed')
   cameras+=old.camera.calibration_valid
  report[name]={'all_frames_compared':review.manifest['frames'],'old_fields_identical':True,'m8_accepted_camera_frames':cameras,'camera_model':review.manifest['camera_model_version'],'player_model':review.manifest['player_model_version'],'ball_model':review.metadata()['ball_model_version']};registry[name]=str(sidecar)
 save('review_preservation.json',report);p=Path('configs/serving/cv_ball_m15.json');temp=p.with_suffix('.tmp');temp.write_text(json.dumps(registry,indent=2)+'\n');os.replace(temp,p);print(report)
if __name__=='__main__':run()
