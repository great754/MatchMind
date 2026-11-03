"""Mandatory validation safeguard: retain frozen M14 if new detector trials regress."""
import json,shutil
from pathlib import Path
from .common import ROOT,load,save,sha
from .benchmark import old_predictions
from ml.ball14.evaluate import detection_metrics

def run():
 if (ROOT/'search_lock.json').exists() or (ROOT/'test_results.json').exists():raise ValueError('Cannot expand selection after final seal/TEST')
 pilot=ROOT/'pilot_yolo';pilot.mkdir(exist_ok=True)
 for name in ['detector_lock.json','pipeline_lock.json','association_model.json','confidence_calibration.json']:
  p=ROOT/name
  if p.exists():shutil.copyfile(p,pilot/name)
 rows=[r for r in load('dataset.json')['samples'] if r['split']=='validation'][::4];ps=old_predictions(rows,'validation_comparison');metrics=detection_metrics(rows,[p['candidates'] for p in ps],.5,.25);lock=load('detector_lock.json');old=json.loads(Path('outputs/ml/ball_m14/campaign_selection.json').read_text())['selected'];candidate={'kind':'m14','checkpoint':old['checkpoint'],'checkpoint_sha256':old['checkpoint_sha256'],'tile':256,'threshold':.5,'metrics':metrics,'inference_seconds':sum(p['seconds'] for p in ps)};winner=max([lock['selected'],candidate],key=lambda c:(c['metrics']['f1'],c['metrics']['precision']));lock.update(selected=winner,frozen_m14_validation_comparison=candidate,selection_extension='New YOLO trials failed validation; mandatory frozen baseline safeguard evaluated before final TEST; no new detector tuning',final_test_seen=False)
 save('detector_lock.json',lock)
 for name in ['train_selected.json','validation_selected.json']:
  p=ROOT/'cache'/name
  if p.exists():p.rename(ROOT/'cache'/('pilot_yolo_'+name))
 # The pilot verifier uses a different detector; preserve it and rebuild before TEST.
 for name in ['pipeline_lock.json','association_model.json','confidence_calibration.json']:
  p=ROOT/name
  if p.exists():p.unlink()
 from .select_detector import predictions
 from .detector import make_model
 model=make_model(winner['checkpoint'])
 for split in ['train','validation']:
  rs=[r for r in load('dataset.json')['samples'] if r['split']==split];predictions(model,rs,split+'_selected',winner['tile'],winner['kind']=='generic')
 print('BASELINE_SAFEGUARD',winner['kind'],winner['metrics'],flush=True)
if __name__=='__main__':run()
