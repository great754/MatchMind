"""Frozen experiment boundaries and content-addressed provenance."""
import hashlib,json,os
from pathlib import Path
M10=Path('outputs/ml/soccer_reid_m10')
ROOT=Path(os.environ.get('TACTIQAI_REID11_RUN_ROOT','outputs/ml/soccer_reid_m11'))
SEED=20261103

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()

def save(name,value):
 ROOT.mkdir(parents=True,exist_ok=True);(ROOT/name).write_text(json.dumps(value,indent=2)+'\n')

def load(name):return json.loads((ROOT/name).read_text())

def freeze():
 from ml.reid.registry import verify_baseline
 verify_baseline()
 paths=list(Path('ml/reid').glob('*.py'))+list(Path('configs/ml').glob('soccer-reid-v1*.json'))
 paths += [M10/p for p in ['curated_manifest.json','dataset_manifest.json','identity_audit.json','selection_lock.json','model_registry.json','epoch_history.json','heldout_comparison.json','shadow_comparison.json','integration_decision.json']]
 paths += [Path('docs')/p for p in ['current_pipeline_status.md','current_pipeline_explorer.html','cv_player_world_model_m9.md','cv_soccer_reid_m10.md']]
 paths += list((M10/'run_moderate').glob('*.pth'))+list((M10/'run_none').glob('*.pth'))
 data={'sha256':{str(p):sha(p) for p in paths},'m10_test_previously_published':True,'test_policy':'M10 TEST reused only as locked final comparison; no new model selection against TEST. Only two unseen matches; not a fresh prospective benchmark.'}
 if (ROOT/'frozen_m10.json').exists():raise ValueError('Already frozen; use a new run root')
 save('frozen_m10.json',data);return data

def verify():
 from ml.reid.registry import verify_baseline
 verify_baseline();d=load('frozen_m10.json');changed=[]
 for p,h in d['sha256'].items():
  if p in ['docs/current_pipeline_status.md','docs/current_pipeline_explorer.html']:continue # mandatory final status update
  if sha(p)!=h:changed.append(p)
 if changed:raise ValueError(f'Frozen M10 changed: {changed}')
 save('baseline_preservation.json',{'frozen_m10_unchanged':True,'m8_m9_unchanged':True,'production_unchanged':True});return True
if __name__=='__main__':freeze()
