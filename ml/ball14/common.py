import os,json
from pathlib import Path
from ml.reid11.common import sha
ROOT=Path(os.environ.get('TACTIQAI_BALL14_ROOT','outputs/ml/ball_m14'))
SEED=20261105

def save(name,value):ROOT.mkdir(parents=True,exist_ok=True);(ROOT/name).write_text(json.dumps(value,indent=2)+'\n')
def load(name):return json.loads((ROOT/name).read_text())
def freeze():
 from ml.reid13.common import verify
 verify();paths=[]
 for root in ['outputs/ml/soccer_reid_m10','outputs/ml/soccer_reid_m11','outputs/ml/soccer_reid_m13']:
  p=Path(root);paths+=list(p.glob('*.json'))+list(p.glob('*.pth'))+list(p.glob('run_*/*.pth'))+list(p.glob('expanded*/*.pth'))+list(p.glob('old_data*/*.pth'))
 for name in ['test_match','team_goal','southampton']:
  p=Path('outputs/world_model')/(name+'_m9_ready');paths+=list(p.glob('*.json'))+list(p.glob('*.jsonl'))
 paths+=[Path('configs/serving/cv.json')];paths+=list(Path('src/analytics').glob('*.py'))+list(Path('src/tactics').glob('*.py'))+list(Path('src/observations').glob('*.py'))
 if (ROOT/'frozen_baselines.json').exists():raise ValueError('Already frozen')
 save('frozen_baselines.json',{'sha256':{str(p):sha(p) for p in paths},'extension_policy':'new M14 ball field and separate canonical stream only; old player/camera/ball placeholder streams immutable'})
def verify():
 for path,h in load('frozen_baselines.json')['sha256'].items():
  if sha(path)!=h:raise ValueError('Baseline changed: '+path)
 save('baseline_preservation.json',{'old_snapshots_and_reid_artifacts_unchanged':True,'analytics_and_tactics_unchanged':True})
if __name__=='__main__':freeze()
