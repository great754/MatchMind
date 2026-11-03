"""Render changed assignment source context without rewriting M9 references."""
import json
from pathlib import Path
import cv2,numpy as np
from .protocol import ROOT

def run():
 out=ROOT/'changed_reconnect_review';out.mkdir(exist_ok=True);cases=[]
 for name in ['test_match','team_goal','southampton']:
  src=Path('outputs/world_model')/(name+'_m9_ready');m=json.loads((src/'manifest.json').read_text());f=json.loads((Path('outputs/world_model/m9_cache')/name/'features.json').read_text());old=json.loads((ROOT/'m9_shadow'/name/'generic_events.json').read_text());new=json.loads((ROOT/'m9_shadow'/name/'soccer_events.json').read_text());oldkeys={(e['from_segment'],e['to_segment']) for e in old};newkeys={(e['from_segment'],e['to_segment']) for e in new};cap=cv2.VideoCapture(m['source_video'])
  for tag,events,other in [('added',new,oldkeys),('removed',old,newkeys)]:
   for e in events:
    if (e['from_segment'],e['to_segment']) in other:continue
    panels=[]
    for frame,track in [(e['disappeared_at_frame']-1,e['source_track']),(e['reappeared_at_frame'],e['target_track'])]:
     cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,image=cap.read();assert ok;p=next(p for p in f[frame] if p['track_id']==track);x,y,w,h=map(round,p['bbox']);cv2.rectangle(image,(x,y),(x+w,y+h),(80,80,255),3);image=cv2.resize(image,(640,360));cv2.putText(image,f'{name} {tag} {track} frame{frame}',(8,24),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1);panels.append(image)
    path=out/f'{name}_{tag}_{e["source_track"]}_{e["target_track"]}.jpg';cv2.imwrite(str(path),np.hstack(panels));cases.append({'recording_id':name,'change':tag,'event':e,'image':str(path),'assessment':'NOT_YET_REVIEWED'})
  cap.release()
 (out/'cases.json').write_text(json.dumps(cases,indent=2))
if __name__=='__main__':run()
