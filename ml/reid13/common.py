import json,os
from pathlib import Path
from ml.reid11.common import sha
ROOT=Path(os.environ.get('TACTIQAI_REID13_RUN_ROOT','outputs/ml/soccer_reid_m13'))
M10=Path('outputs/ml/soccer_reid_m10');M11=Path('outputs/ml/soccer_reid_m11');SEED=20261104

def save(name,value):
 ROOT.mkdir(parents=True,exist_ok=True);(ROOT/name).write_text(json.dumps(value,indent=2)+'\n')
def load(name):return json.loads((ROOT/name).read_text())
def freeze():
 from ml.reid11.common import verify
 verify();paths=[]
 for root in [M10,M11]:
  paths+=list(root.glob('*.json'))+list(root.glob('*.pth'))+list(root.glob('run_*/*.pth'))
 paths+=list(Path('ml/reid').glob('*.py'))+list(Path('ml/reid11').glob('*.py'))
 paths+=[Path('docs')/n for n in ['current_pipeline_status.md','current_pipeline_explorer.html','cv_soccer_reid_m10.md','cv_temporal_soccer_reid_m11.md']]
 if (ROOT/'frozen_baselines.json').exists():raise ValueError('Already frozen')
 save('frozen_baselines.json',{'sha256':{str(p):sha(p) for p in paths},'historical_test_not_for_selection':True})
def verify():
 from ml.reid.registry import verify_baseline
 verify_baseline();d=load('frozen_baselines.json')
 for p,h in d['sha256'].items():
  if p in ['docs/current_pipeline_status.md','docs/current_pipeline_explorer.html']:continue
  if sha(p)!=h:raise ValueError('Frozen input changed: '+p)
 save('baseline_preservation.json',{'m8_m9_unchanged':True,'m10_m11_frozen_inputs_unchanged':True,'production_activated':False})
if __name__=='__main__':freeze()
