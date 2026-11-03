"""Version identity is separate from mutable evaluation/integration status."""
import hashlib,json
from pathlib import Path
from src.vision.world.player_appearance import digest

def fingerprint(entry):
 fields=['model_name','architecture','checkpoint_sha256','training_manifest_sha256','training_configuration_sha256','created_date'];metadata={k:entry[k] for k in fields};return hashlib.sha256(json.dumps(metadata,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def verify_hashes(expected):
 for path,sha in expected.items():
  if digest(path)!=sha:raise ValueError(f'Frozen input changed: {path}')
 return True

def verify_baseline(root=Path('outputs/ml/soccer_reid_m10')):
 frozen=json.loads((root/'frozen_m9.json').read_text());expected={**frozen['source_code_sha256'],'configs/serving/cv.json':frozen['registry_sha256'],'outputs/models/m9/osnet_x0_25_msmt17.pth':frozen['generic_checkpoint_sha256']}
 for name in ['test_match','team_goal','southampton']:
  base=Path('outputs/world_model')/(name+'_m9_ready');expected[str(base/'frames.jsonl')]=frozen[name]['frames_sha256'];expected[str(base/'manifest.json')]=frozen[name]['manifest_sha256'];expected[str(base/'reid_candidates.json')]=frozen[name]['candidates_sha256']
 verify_hashes(expected);result={'m9_and_m8_snapshots_unchanged':True,'m9_team_reachability_assignment_code_unchanged':True,'analytics_code_unchanged':True,'generic_checkpoint_unchanged':True,'active_registry_unchanged':True};(root/'baseline_preservation.json').write_text(json.dumps(result,indent=2));return result
if __name__=='__main__':print(verify_baseline())
