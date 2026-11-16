"""Quantify candidate generation, threshold, localization and temporal losses on VAL only."""
import json,math
from pathlib import Path
from collections import Counter
import numpy as np,cv2
from ml.ball14.evaluate import iou
from .common import save

def run():
 root=Path('outputs/ml/ball_m15');read=lambda n:json.loads((root/n).read_text());rows=[r for r in read('dataset.json')['samples'] if r['split']=='validation'];ps=read('cache/validation_selected.json');local=read('cache/validation_guided.json')['predictions'];threshold=read('detector_lock.json')['selected']['threshold'];counts=Counter();scores=[];grid={};near=[]
 for row,p,loc in zip(rows,ps,local,strict=True):
  for b in row['balls']:
   cs=p['candidates'];good=[c for c in cs if iou(c['bbox'],b['bbox'])>=.25];oper=[c for c in good if c['confidence']>=threshold];counts['gt']+=1;counts['raw_candidate_match']+=bool(good);counts['operating_match']+=bool(oper);counts['tiny_under12']+=max(b['bbox'][2:])<12
   center=[b['bbox'][0]+b['bbox'][2]/2,b['bbox'][1]+b['bbox'][3]/2];close=any(math.dist([c['bbox'][0]+c['bbox'][2]/2,c['bbox'][1]+c['bbox'][3]/2],center)<=12 for c in cs)
   counts['no_candidate']+=not good and not close;counts['localization_failure_near_center']+=not good and close;counts['threshold_loss']+=bool(good) and not oper
   localgood=[c for c in loc['candidates'] if iou(c['bbox'],b['bbox'])>=.25];counts['local_raw_match']+=bool(localgood);counts['local_operating_match']+=any(c['confidence']>=threshold for c in localgood);counts['global_recovered_local_lost']+=bool(good) and not localgood;counts['local_recovered_global_lost']+=not good and bool(localgood)
   if good:scores.append(max(c['confidence'] for c in good))
 for t in [.02,.1,.2,.35,.5,.65]:
  from ml.ball14.evaluate import detection_metrics
  m=detection_metrics(rows,[p['candidates'] for p in ps],t,.25);grid[str(t)]={k:m[k] for k in ['tp','fp','fn','precision','recall','f1']}
 train=[r for r in read('dataset.json')['samples'] if r['split']=='train'];hashes=[];diameters=[]
 for row in train:
  image=cv2.imread(row['path'])
  for b in row['balls']:
   x,y,w,h=b['bbox'];crop=image[max(0,int(y)):int(y+h),max(0,int(x)):int(x+w)];diameters.append(max(w,h))
   if crop.size:small=cv2.resize(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY),(8,8));hashes.append(np.packbits(small>small.mean()).tobytes().hex())
 save('failure_taxonomy.json',{'validation_counts':dict(counts),'threshold_grid':grid,'training':{'frames':len(train),'clips':len({r['recording_id'] for r in train}),'games':len({r['match_id'] for r in train}),'ball_boxes':len(diameters),'unique_ball_appearance_8x8_hashes':len(set(hashes)),'median_ball_extent_px':float(np.median(diameters))},'measured_local_failure':'Compare raw candidate recall before scoring; M15 replaced global proposals with local-only proposals and reused high visual thresholds; local crops change apparent scale and lack verified crop-specific calibration. Global recovery prevents permanent lock but cannot recover visually unproposed balls. Exact counters above quantify losses without assigning unverified causes.','gru_failure':'VAL mean endpoint error worse at all horizons; TEST region radii extremely broad. Only199 train histories from one game, overlapping frames, no visual input, exposure/domain shift; these are design limitations, not experimentally isolated causes.','ground':'M14 accepted416 image observations, zero near-foot contact; no independent metric references/height calibration. Camera alone cannot determine ball height.','training_quality':'One game/two short clips; positive repeated stationary balls; random and person crops are not verified negatives. M16 must not repeat unknown-visibility negative labeling.','larger_experiment_authorized_after_diagnosis':True});print(dict(counts),grid,flush=True)
if __name__=='__main__':run()
