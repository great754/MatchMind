"""Choose between the two bounded validation-only trials, then freeze the campaign."""
import json
from pathlib import Path
from .common import ROOT,load,save,sha
from src.vision.ball.tracker import BallTrackerConfig

def run():
 if (ROOT/'campaign_selection.json').exists():raise ValueError('Campaign already sealed; no post-TEST tuning')
 trials=[ROOT,ROOT/'trial_small'];candidates=[]
 for trial in trials:
  settings=json.loads((trial/'trial_settings.json').read_text()) if (trial/'trial_settings.json').exists() else {}
  for c in json.loads((trial/'validation_candidates.json').read_text()):
   candidates.append({**c,'checkpoint':c.get('checkpoint',str(ROOT/'generic_fasterrcnn_mobilenet320.pth')),'checkpoint_sha256':c.get('checkpoint_sha256',load('initializer.json')['sha256']),'tile_size':settings.get('inference_tile',480),'trial_root':str(trial),'trial_settings':settings})
 key=lambda c:(c['metrics']['f1'],c['metrics']['precision'],not c['tiled'])
 chosen=max(candidates,key=key);fine=max([c for c in candidates if c['kind']=='finetuned'],key=key);cfg=BallTrackerConfig(minimum_confidence=chosen['threshold']);save('tracker_config.json',cfg.model_dump());result={'selected':chosen,'best_finetuned':fine,'validation_only':True,'test_seen':False,'trial_selection_sha256':[sha(t/'selection_lock.json') for t in trials],'tracker_config_sha256':sha(ROOT/'tracker_config.json'),'dataset_sha256':sha(ROOT/'dataset.json'),'benchmark':'all200TESTframes, match-disjoint; detector full/tile480/tile256 and best fine selected only with20VALframes; tracker-only, camera,physics ablations on identical samples','primary_detector_iou':.25,'strict_detector_iou':.5,'ground_truth_limits':'original GT visibility/occlusion/airborne/in-play identity unannotated; do not convert absent labels into invisible truth'};save('campaign_selection.json',result);Path('configs/ml/ball-m14-tracker.json').write_text(json.dumps(cfg.model_dump(),indent=2)+'\n');print('CAMPAIGN',chosen['kind'],chosen['tile_size'],chosen['threshold'],'VALF1',chosen['metrics']['f1'],flush=True)
if __name__=='__main__':run()
