"""Render source context/crops for a small manual M9 reference review."""
import json
from pathlib import Path
import cv2,numpy as np

def run():
 for name in ['test_match','team_goal','southampton']:
  source=Path('outputs/world_model')/(name+'_m9_ready');out=Path('outputs/world_model/m9_verification')/name/'reference';out.mkdir(parents=True,exist_ok=True);m=json.loads((source/'manifest.json').read_text());features=json.loads((Path('outputs/world_model/m9_cache')/name/'features.json').read_text());events=json.loads((source/'reassociation_events.json').read_text());candidates=json.loads((source/'reid_candidates.json').read_text());cases=[]
  for e in events:cases.append({'kind':'ACCEPTED_RECONNECT','a_frame':e['disappeared_at_frame']-1,'b_frame':e['reappeared_at_frame'],'a_track':e['source_track'],'b_track':e['target_track'],'decision':e})
  for target,values in candidates.items():
   if len(cases)>=max(6,len(events)+2):break
   if any(e['to_segment']==target for e in events):continue
   valid=[c for c in values if c.get('appearance_similarity') is not None and c['appearance_similarity']>=.82]
   if not valid:continue
   c=valid[0];a=c['segment_id'];a_track=a.split(':')[1];a_frame=max(i for i,ps in enumerate(features) if i<int(target.split(':')[2]) and any(p['track_id']==a_track for p in ps));cases.append({'kind':'REJECTED_OR_UNRESOLVED','a_frame':a_frame,'b_frame':int(target.split(':')[2]),'a_track':a_track,'b_track':target.split(':')[1],'decision':c})
  cap=cv2.VideoCapture(m['source_video']);tiles=[]
  for k,c in enumerate(cases):
   panels=[];crops=[]
   for tag in ['a','b']:
    frame=c[tag+'_frame'];track=c[tag+'_track'];cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,image=cap.read();assert ok;p=next(p for p in features[frame] if p['track_id']==track);x,y,w,h=map(round,p['bbox']);crop=image[max(0,y):min(image.shape[0],y+h),max(0,x):min(image.shape[1],x+w)];cv2.rectangle(image,(x,y),(x+w,y+h),(70,70,255),3);panel=cv2.resize(image,(640,360));cv2.putText(panel,f'{name} frame{frame} {track}',(8,24),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2);panels.append(panel);crops.append(cv2.resize(crop,(120,180),interpolation=cv2.INTER_NEAREST))
   header=np.full((40,1280,3),25,np.uint8);cv2.putText(header,f'Case{k} {c["kind"]} -- full source context; red box is candidate',(8,26),cv2.FONT_HERSHEY_SIMPLEX,.55,(240,240,240),1);canvas=np.vstack([header,np.hstack(panels)]);cv2.imwrite(str(out/f'case_{k:02d}.jpg'),canvas);cv2.imwrite(str(out/f'case_{k:02d}_crops.jpg'),np.hstack(crops));c['case_id']=f'{name}_{k:02d}';c['manual_label']='NOT_YET_REVIEWED';c['challenge']='Requires source-context review; crop similarity alone is insufficient';tiles.append(cv2.resize(canvas,(640,200)))
  cap.release();(out/'reference_cases.json').write_text(json.dumps(cases,indent=2))
  if len(tiles)%2:tiles.append(np.zeros_like(tiles[0]))
  cv2.imwrite(str(out/'contact_sheet.jpg'),np.vstack([np.hstack(tiles[i:i+2]) for i in range(0,len(tiles),2)]))
if __name__=='__main__':run()
