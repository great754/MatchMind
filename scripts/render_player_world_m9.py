"""Source/kit reference panels; labels remain experimental, not ground truth."""
import json
from pathlib import Path
import cv2,numpy as np
from src.vision.world.player_appearance import raw_people,foreground

COLORS={'TEAM_A':(120,130,255),'TEAM_B':(255,170,110),'OFFICIAL':(100,235,255),'OTHER':(235,150,190),'UNKNOWN':(230,230,230)}
def run():
 for name in ['test_match','team_goal','southampton']:
  source=Path('outputs/world_model')/(name+'_m9_ready');m=json.loads((source/'manifest.json').read_text());rows=[json.loads(l) for l in (source/'frames.jsonl').open()];out=Path('outputs/world_model/m9_verification')/name/'team_reference';out.mkdir(exist_ok=True,parents=True);cap=cv2.VideoCapture(m['source_video']);records=[];tiles=[]
  for frame in [30,100,216,350,min(500,len(rows)-1)]:
   cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,image=cap.read();assert ok;raw=raw_people(rows[frame]);world={p['current_track_id']:p for p in rows[frame]['player_world']['players'] if p['current_track_id']};panel=image.copy()
   for p in raw:
    x,y,w,h=map(round,p['bbox']);new=world.get(p['track_id']);label=new['team']['label'] if new else 'UNKNOWN';cv2.rectangle(panel,(x,y),(x+w,y+h),COLORS[label],max(1,m['width']//600));cv2.putText(panel,label,(max(0,x),max(12,y-3)),cv2.FONT_HERSHEY_SIMPLEX,max(.25,m['width']/2500),COLORS[label],1)
    f,mask=foreground(image,p['bbox']);x1,y1,w1,h1=f['torso_bbox']
    if mask is not None and mask.size:
     crop=image[y1:y1+h1,x1:x1+w1];prefix=f'{frame}_{p["track_id"]}';cv2.imwrite(str(out/(prefix+'_crop.png')),crop);cv2.imwrite(str(out/(prefix+'_mask.png')),mask.astype(np.uint8)*255);lab=cv2.cvtColor(crop,cv2.COLOR_BGR2LAB);hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV);np.savez_compressed(out/(prefix+'_pixels.npz'),lab=lab[mask],hsv=hsv[mask],bgr=crop[mask]);records.append({'frame':frame,'track_id':p['track_id'],'legacy_team':p['legacy_team'],'m9_team':label,'m9_visual_team':new['team']['current_visual_label'] if new else 'UNKNOWN','entity':new['entity_id'] if new else None,'feature':f,'manual_label':'NOT_YET_REVIEWED'})
   tile=cv2.resize(panel,(640,round(panel.shape[0]*640/panel.shape[1])));cv2.imwrite(str(out/f'frame_{frame}.jpg'),tile);tiles.append(tile)
  cap.release();(out/'samples.json').write_text(json.dumps(records,indent=2));cv2.imwrite(str(out/'contact_sheet.jpg'),np.vstack(tiles))
if __name__=='__main__':run()
