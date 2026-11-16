"""Measure additive crop recovery against cached global candidates on VAL only."""
import cv2
from ml.ball14.evaluate import iou
from src.vision.ball16.tracker import MultiHypothesisTracker
from .common import load,save
from .detector import infer,recovery_region,merge_additive

def audit(rows,ps,model,resolution,threshold):
 last=None;tracker=None;before=after=0;calls=0;extra_seconds=0
 for r,p in zip(rows,ps,strict=True):
  if r['recording_id']!=last:tracker=MultiHypothesisTracker(r['width'],r['height'],threshold);last=r['recording_id']
  point,sigma=tracker.predict(r['timestamp']);region=recovery_region(point,sigma,r['width'],r['height']);candidates=p['candidates']
  if region:
   local,t=infer(model,cv2.imread(r['path']),resolution,region);candidates=merge_additive(candidates,local);calls+=1;extra_seconds+=t['seconds']
  for b in r['balls']:
   before+=any(iou(c['bbox'],b['bbox'])>=.25 for c in p['candidates']);after+=any(iou(c['bbox'],b['bbox'])>=.25 for c in candidates)
  tracker.step(r['frame'],r['timestamp'],candidates,r.get('camera',{'epoch':0,'calibration_valid':False}))
 result={'frames':len(rows),'global_raw_gt_matches':before,'additive_raw_gt_matches':after,'extra_local_calls':calls,'extra_local_seconds':extra_seconds,'selected':False,'reason':'Proposal ceiling diagnostic only; not selected without separate calibrated accepted-tracking coverage improvement. Global candidates preserved by construction.'};save('local_recovery_audit.json',result);return result
