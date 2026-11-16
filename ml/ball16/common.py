"""Isolated M16 research with immutable historical inputs."""
import os,json
from pathlib import Path
from ml.ball15.common import sha
ROOT=Path(os.environ.get('TACTIQAI_BALL16_ROOT','outputs/ml/ball_m16'))
SEED=20261116
os.environ.setdefault('YOLO_CONFIG_DIR','/tmp/tactiq-m16-ultralytics')
def save(n,v):ROOT.mkdir(exist_ok=True,parents=True);(ROOT/n).write_text(json.dumps(v,indent=2)+'\n')
def load(n):return json.loads((ROOT/n).read_text())
def freeze():
 if (ROOT/'frozen.json').exists():return verify()
 from ml.ball15.common import verify as previous
 previous();paths=[]
 for directory in ['outputs/ml/ball_m15','outputs/world_model/ball_m15','ml/ball15','src/vision/ball15']:
  paths += [p for p in Path(directory).rglob('*') if p.is_file() and p.suffix in ['.json','.jsonl','.pt','.pth','.py']]
 paths += [Path('configs/serving/cv_ball_m15.json'),Path('docs/cv_ball_tracking_m14.md'),Path('docs/cv_ball_tracking_m15.md')]
 save('frozen.json',{'sha256':{str(p):sha(p) for p in paths}})
def verify():
 from ml.ball15.common import verify as previous
 previous()
 for p,h in load('frozen.json')['sha256'].items():
  if sha(p)!=h:raise ValueError('Frozen historical content changed: '+p)
 return True
if __name__=='__main__':freeze()
