"""All three versions on identical sealed source-only references; no labels from predictions."""
import json,math
from pathlib import Path
from collections import Counter
from ml.ball14.evaluate import iou
from .common import save,load

def run():
 audit=load('demo_annotations.json');rows=[r for r in audit['rows'] if r['scorable']];results={}
 for name in ['test_match','team_goal','southampton','all']:
  report={}
  for version in ['m14','m15','m16']:
   counts=Counter();cache={}
   for r in rows:
    n=r['recording_id']
    if name!='all' and n!=name:continue
    if n not in cache:cache[n]=[json.loads(l) for l in Path(f'outputs/world_model/ball_{version}/{n}/ball_frames.jsonl').open()]
    s=cache[n][r['frame']];gt=r['balls'][0]['bbox'];center=[gt[0]+gt[2]/2,gt[1]+gt[3]/2];counts['frames']+=1;counts['observed']+=s['directly_observed'];counts['metric']+=s['pitch_position'] is not None
    if s['directly_observed']:counts['correct_iou']+=iou(s['image_bbox'],gt)>=.25;counts['correct_center']+=math.dist(s['image_position'],center)<=12
    if s['predicted_image_position'] is not None:counts['predicted']+=1;counts['prediction_inconsistent']+=math.dist(s['predicted_image_position'],center)>12
   report[version]=dict(counts)
  results[name]=report
 save('demo_reference_results.json',{'results':results,'annotation_source':'M16 sealed source-only40clear boxes (27historical+13new),uncertainframesexcluded. Single Codex visual/color-assisted audit, not independently adjudicated human GT. No verified metric coordinates or new labels inferred from detections.','expanded_visibility_or_ground_reference_available':False})
 print(results)
if __name__=='__main__':run()
