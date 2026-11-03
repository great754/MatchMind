"""M8 read-only torso foreground evidence for a future team milestone."""
import json
from pathlib import Path
import cv2,numpy as np
from scripts.diagnose_camera_m8 import SOURCES

def run():
 for name,(baseline,poses) in SOURCES.items():
  root=Path(baseline);m=json.loads((root/'manifest.json').read_text());out=Path('outputs/world_model/m8_verification')/name/'team_evidence';out.mkdir(parents=True,exist_ok=True);rows=[json.loads(l) for l in (root/'frames.jsonl').open()];cap=cv2.VideoCapture(m['source_video']);records=[];events=[]
  for row in rows:
   for e in row['entities']:
    for event in e['history']:
     if event['frame']==row['frame'] and event['kind'].startswith('team_'):events.append({'entity':e['entity_id'],**event})
  for index in sorted({0,30,100,216,350,min(500,len(rows)-1),len(rows)-1}):
   cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,image=cap.read();assert ok
   row=rows[index];people=[{'id':e['entity_id'],'kit':e['current_kit'],'team':e['team'],'belief':e.get('team_belief'),'votes':e['team_memory'],'events':e['history']} for e in row['entities'] if e['visibility']=='visible']+[{'id':u['observation']['track_id'],'kit':u['observation'].get('kit'),'team':u['observation']['team'],'belief':u['observation'].get('team_belief')} for u in row['unassigned']]
   picked=[];seen=set()
   for p in people:
    kit=p['kit']
    if not kit or not kit.get('torso_bbox'):continue
    x,y,w,h=map(int,kit['torso_bbox']);crop=image[y:y+h,x:x+w]
    if not crop.size:continue
    hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV);category=(p['team'],bool(np.median(hsv[:,:,2])<85))
    if category in seen and len(picked)>=3:continue
    seen.add(category);prefix=f'{index:04d}_{p["id"]}';lab=cv2.cvtColor(crop,cv2.COLOR_BGR2LAB);mask=(hsv[:,:,2]>30)&(hsv[:,:,1]>45)
    if kit.get('background_lab') is not None:mask&=np.linalg.norm(lab.astype(float)-kit['background_lab'],axis=2)>22
    cv2.imwrite(str(out/(prefix+'_crop.png')),crop);cv2.imwrite(str(out/(prefix+'_foreground.png')),mask.astype(np.uint8)*255);np.savez_compressed(out/(prefix+'_pixels.npz'),foreground_lab=lab[mask],foreground_hsv=hsv[mask],foreground_bgr=crop[mask],foreground_mask=mask)
    records.append({'frame':index,**p,'files_prefix':prefix,'reconstructed_foreground_fraction':float(mask.mean()),'note':'Exact existing V>30/S>45 and local LAB-distance>22 foreground rule; classifier not changed'});picked.append(p)
    if len(picked)>=5:break
  cap.release();team=json.loads((root/'manifest.json').read_text()).get('team_model',{})
  if not team and name=='southampton':team=json.loads(Path('outputs/world_model/southampton/players/diagnostics.json').read_text())['team_model']
  (out/'evidence.json').write_text(json.dumps({'baseline':baseline,'classifier_unchanged':True,'cluster_model':team,'samples':records,'team_events':events,'independent_team_ground_truth':None},indent=2));print(name,len(records),'preserved crops',flush=True)
if __name__=='__main__':run()
