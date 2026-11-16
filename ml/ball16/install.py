"""Validate an optional M16 review; preserve default M14 and old world content."""
import json,os
from pathlib import Path
from src.serving.cv import CVReview
from src.vision.world.schema import WorldState
from .common import save,verify

def run():
 registry={};report={};verify()
 for name in ['test_match','team_goal','southampton']:
  base=Path(f'outputs/world_model/{name}_m9_ready');out=Path(f'outputs/world_model/ball_m16/{name}');review=CVReview(base);review.attach_ball(out)
  if review.ball_review is None:raise ValueError('M16 invalid: '+name)
  counts=0
  for i,line in enumerate((base/'frames.jsonl').open()):
   before=WorldState.model_validate_json(line);after=review.frame((i+.001)/review.manifest['fps'])
   if before.model_dump(exclude={'ball_world'})!=after.model_dump(exclude={'ball_world'}):raise ValueError('Old fields changed')
   counts+=before.camera.calibration_valid
  report[name]={'frames':review.manifest['frames'],'old_world_fields_unchanged':True,'m8_camera_accepted':counts,'ball_version':review.metadata()['ball_model_version']};registry[name]=str(out)
 p=Path('configs/serving/cv_ball_m16.json');temp=p.with_suffix('.tmp');temp.write_text(json.dumps(registry,indent=2)+'\n');os.replace(temp,p);save('review_preservation.json',report);print(report)
if __name__=='__main__':run()
