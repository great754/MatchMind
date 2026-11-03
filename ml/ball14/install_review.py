"""Validate every additive frame, then atomically enable local experimental review."""
import json,os
from pathlib import Path
from src.serving.cv import CVReview
from src.vision.world.schema import WorldState
from .common import ROOT,save,verify

def run():
 registry={};report={}
 for name in ['test_match','team_goal','southampton']:
  base=Path('outputs/world_model')/(name+'_m9_ready');sidecar=Path('outputs/world_model/ball_m14')/name;review=CVReview(base);review.attach_ball(sidecar)
  if review.ball_review is None:raise ValueError('Ball validation failed: '+name+' '+str(review.ball_diagnostic))
  cameras=0
  for i,line in enumerate((base/'frames.jsonl').open()):
   original=WorldState.model_validate_json(line);joined=review.frame((i+.001)/review.manifest['fps'])
   if original.model_dump(exclude={'ball_world'})!=joined.model_dump(exclude={'ball_world'}):raise ValueError('M8/M9 content changed during ball join')
   cameras+=original.camera.calibration_valid
  report[name]={'all_frames_compared':review.manifest['frames'],'old_world_fields_unchanged':True,'m8_camera_accepted_frames':cameras,'player_model_version':review.manifest['player_model_version'],'ball_model_version':review.metadata()['ball_model_version'],'metric_ball_frames':review.ball_review.manifest['statistics']['metric_supported']};registry[name]=str(sidecar)
 verify();save('review_preservation.json',report);path=Path('configs/serving/cv_ball.json');temp=path.with_suffix('.tmp');temp.write_text(json.dumps(registry,indent=2)+'\n');os.replace(temp,path);print(report,flush=True)
if __name__=='__main__':run()
