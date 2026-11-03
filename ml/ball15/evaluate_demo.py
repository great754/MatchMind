"""Sparse source-only reference evaluation of actual full-video M14/M15 causal sidecars."""
import json,math
from pathlib import Path
from collections import Counter
from .common import load,save
from ml.ball14.evaluate import iou
from .metrics import wilson

def run():
 annotated=[r for r in load('demo_annotations.json')['rows'] if r['scorable']];result={}
 for name in ['test_match','team_goal','southampton','all']:
  report={}
  for version in ['m14','m15']:
   counts=Counter();errors=[];frames=[];cache={};camera_cache={}
   for r in annotated:
    if name!='all' and r['recording_id']!=name:continue
    recording=r['recording_id']
    if recording not in cache:cache[recording]=[json.loads(l) for l in Path(f'outputs/world_model/ball_{version}/{recording}/ball_frames.jsonl').open()]
    s=cache[recording][r['frame']];gt=r['balls'][0]['bbox'];center=[gt[0]+gt[2]/2,gt[1]+gt[3]/2];counts['frames']+=1;point=s['image_position'];counts['observed']+=s['directly_observed']
    if s['directly_observed']:
     good=iou(s['image_bbox'],gt)>=.25;counts['observed_iou_correct']+=good;counts['observed_iou_incorrect']+=not good;err=math.dist(point,center);errors.append(err);counts['observed_center_correct']+=err<=12
    if s['predicted_image_position'] is not None:counts['predicted']+=1;counts['predicted_gt_inconsistent']+=math.dist(s['predicted_image_position'],center)>12
    counts['metric_supported']+=s['pitch_position'] is not None;counts['airborne_metric_acceptance']+=r['height']=='AIRBORNE' and s['pitch_position'] is not None;frames.append({'frame':r['frame'],'recording_id':recording,'tracking_status':s['tracking_status'],'observed':s['directly_observed'],'source_only_height_label':r['height'],'metric':s['pitch_position']})
   report[version]={'counts':dict(counts),'observed_precision_iou':counts['observed_iou_correct']/counts['observed'] if counts['observed'] else None,'observed_visible_recall_iou':counts['observed_iou_correct']/counts['frames'],'observed_center_consistency':counts['observed_center_correct']/counts['observed'] if counts['observed'] else None,'observed_center_wilson95':wilson(counts['observed_center_correct'],counts['observed']),'prediction_inconsistency':counts['predicted_gt_inconsistent']/counts['predicted'] if counts['predicted'] else None,'median_observed_center_error_px':sorted(errors)[len(errors)//2] if errors else None,'metric_error_m':None,'frames':frames}
  result[name]=report
 save('demo_sidecar_evaluation.json',{'results':result,'scope':'27 source-only manually/color-assisted visible boxes, 14 uncertain frames excluded. Single Codex annotator, not external adjudication; no surveyed ground truth. Exact full-video sidecar clocks. Sparse sample cannot measure true occlusion/reacquisition recall or dense tracking accuracy.'});print({n:{v:r['counts'] for v,r in x.items()} for n,x in result.items()})
if __name__=='__main__':run()
