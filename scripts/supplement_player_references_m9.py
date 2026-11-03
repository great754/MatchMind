"""Zoom/framing and long-gap source-context reference panels (not identity truth)."""
import json
from pathlib import Path
import cv2,numpy as np
from src.vision.world.player_appearance import raw_people

def run():
 refs=[]
 for name in ['test_match','team_goal','southampton']:
  base=Path('outputs/world_model')/(name+'_m9_ready');m=json.loads((base/'manifest.json').read_text());rows=[json.loads(l) for l in (base/'frames.jsonl').open()];cap=cv2.VideoCapture(m['source_video']);out=Path('outputs/world_model/m9_verification')/name/'reference';scores=[]
  for i,r in enumerate(rows):
   zoom=r['camera']['motion_evidence']['metric_camera'].get('zoom_singular_values')
   if zoom and 8<i<len(rows)-8:scores.append((max(abs(np.log(x)) for x in zoom if x>0),i))
  chosen=max(scores)[1];candidates=[];before=raw_people(rows[chosen-6]);after=raw_people(rows[chosen+6]);common={p['track_id'] for p in before}&{p['track_id'] for p in after}
  if common:candidates.append(('zoom_framing',chosen-6,chosen+6,sorted(common)[0],sorted(common)[0]))
  evid=json.loads((base/'reid_candidates.json').read_text());long=[(e['appearance_similarity'],target,e) for target,es in evid.items() for e in es if e['elapsed_seconds']>6 and (e['appearance_similarity'] or 0)>.9]
  if long:
   _,target,e=max(long,key=lambda x:x[0]);track=e['segment_id'].split(':')[1];last=max(i for i,r in enumerate(rows) if i<int(target.split(':')[2]) and any(p['track_id']==track for p in raw_people(r)));candidates.append(('long_disappearance',last,int(target.split(':')[2]),track,target.split(':')[1]))
  for kind,a,b,ta,tb in candidates:
   images=[]
   for frame,track in [(a,ta),(b,tb)]:
    cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,image=cap.read();assert ok;p=next(p for p in raw_people(rows[frame]) if p['track_id']==track);x,y,w,h=map(round,p['bbox']);cv2.rectangle(image,(x,y),(x+w,y+h),(90,80,255),3);image=cv2.resize(image,(640,360));cv2.putText(image,f'{name} {kind} {frame} {track}',(8,24),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1);images.append(image)
   path=out/(kind+'.jpg');cv2.imwrite(str(path),np.hstack(images));refs.append({'recording_id':name,'kind':kind,'frames':[a,b],'tracks':[ta,tb],'source_sha256':m['video_sha256'],'image':str(path),'assessment':'NOT_YET_REVIEWED','independent_ground_truth':False})
  cap.release()
 Path('outputs/world_model/m9_verification/supplementary_reference.json').write_text(json.dumps(refs,indent=2))
if __name__=='__main__':run()
