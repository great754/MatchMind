"""Reproducible source extracts and unselected error review, with M14/M15 comparison artifacts."""
import json
from pathlib import Path
import cv2,numpy as np
from ml.ball14.evaluate import iou
from .common import ROOT,load,save,sha

def source_frames():
 for name in ['test_match','team_goal','southampton']:
  rows=[r for r in json.loads(Path('configs/ml/ball-m15-demo-annotations.json').read_text())['rows'] if r['recording_id']==name];cap=cv2.VideoCapture(f'data/raw/{name}.mp4')
  for r in rows:
   cap.set(cv2.CAP_PROP_POS_FRAMES,r['frame']);ok,image=cap.read();assert ok;p=Path(r['path']);p.parent.mkdir(exist_ok=True,parents=True);cv2.imwrite(str(p),image)
   if sha(p)!=r['source_sha256']:raise ValueError('Source JPEG differs from original annotation source')
  cap.release()

def run():
 rows=[r for r in load('dataset.json')['samples'] if r['split']=='test'];ps=load('cache/test_selected.json');threshold=load('detector_lock.json')['selected']['threshold'];examples=[];counts={};dest=ROOT/'failure_examples';dest.mkdir(exist_ok=True)
 for r,p in zip(rows,ps,strict=True):
  cs=[c for c in p['candidates'] if c['confidence']>=threshold];matches=[max([iou(c['bbox'],b['bbox']) for b in r['balls']],default=0)>=.25 for c in cs];kind='miss' if r['balls'] and not any(matches) else 'false_proposal' if any(not b for b in matches) else 'matched'
  if counts.get(kind,0)>=3:continue
  im=cv2.imread(r['path'])
  for b in r['balls']:
   x,y,w,h=map(int,b['bbox']);cv2.rectangle(im,(x,y),(x+w,y+h),(255,255,0),2)
  for c,match in zip(cs,matches):
   x,y,w,h=map(int,c['bbox']);cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0) if match else (0,0,255),1)
  cv2.putText(im,f'M15 {kind} {r["sample_id"]}; cyan organizer GT',(10,25),0,.65,(255,255,255),2);p=dest/f'{kind}_{counts.get(kind,0)}.jpg';cv2.imwrite(str(p),im);examples.append({'kind':kind,'sample_id':r['sample_id'],'path':str(p)});counts[kind]=counts.get(kind,0)+1
 comparisons={}
 for name in ['test_match','team_goal','southampton']:
  old=[json.loads(l) for l in Path(f'outputs/world_model/ball_m14/{name}/ball_frames.jsonl').open()];new=[json.loads(l) for l in Path(f'outputs/world_model/ball_m15/{name}/ball_frames.jsonl').open()];path=ROOT/f'{name}_m14_m15_comparison.jsonl'
  path.write_text(''.join(json.dumps({'frame':i,'timestamp':b['timestamp'],'m14':{k:a[k] for k in ['tracking_status','image_position','predicted_image_position','pitch_position','directly_observed']},'m15':{k:b[k] for k in ['tracking_status','image_position','predicted_image_position','pitch_position','directly_observed','candidate_verification','search_stage']}})+'\n' for i,(a,b) in enumerate(zip(old,new,strict=True))));comparisons[name]={'path':str(path),'sha256':sha(path)}
 save('visual_review.json',{'examples':examples,'frame_comparisons':comparisons,'warning':'IoU mismatches do not prove every unmatched proposal is a different object; organizer visibility unknown. Real videos have no independent surveyed metric GT.'})
if __name__=='__main__':run()
