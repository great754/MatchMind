"""M15 isolated experiments and immutable predecessor checks."""
import os,json,hashlib
from pathlib import Path
ROOT=Path(os.environ.get('TACTIQAI_BALL15_ROOT','outputs/ml/ball_m15'))
SEED=20261115
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def save(n,v):ROOT.mkdir(exist_ok=True,parents=True);(ROOT/n).write_text(json.dumps(v,indent=2)+'\n')
def load(n):return json.loads((ROOT/n).read_text())
def freeze():
 if (ROOT/'frozen.json').exists():return verify()
 from ml.ball14.common import verify as oldverify
 oldverify();paths=[]
 for root in ['outputs/ml/ball_m14','outputs/world_model/ball_m14']:
  paths += [p for p in Path(root).rglob('*') if p.is_file() and p.suffix in ['.json','.jsonl','.pth']]
 paths += list(Path('ml/ball14').glob('*.py'))+[Path('src/vision/ball/tracker.py'),Path('src/vision/ball/geometry.py'),Path('configs/serving/cv.json'),Path('configs/serving/cv_ball.json')]
 save('frozen.json',{'sha256':{str(p):sha(p) for p in paths}})
def verify():
 from ml.ball14.common import verify as oldverify
 oldverify()
 for p,h in load('frozen.json')['sha256'].items():
  if sha(p)!=h:raise ValueError('Frozen predecessor changed: '+p)
 return True
if __name__=='__main__':freeze()
