"""Small before/after source/debug JPEGs from measured snapshots; no video copies."""
import argparse,json
from pathlib import Path
import cv2,numpy as np
from src.vision.world.pitch_geometry import PitchGeometry


def render(before,after,output,frames):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);a=[json.loads(l) for l in (Path(before)/'frames.jsonl').open()];b=[json.loads(l) for l in (Path(after)/'frames.jsonl').open()];m=json.loads((Path(after)/'manifest.json').read_text());cap=cv2.VideoCapture(m['source_video']);geo=PitchGeometry();tiles=[]
    for index in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,image=cap.read();assert ok;panels=[]
        for label,row in [('M5',a[index]),('M6',b[index])]:
            panel=image.copy();c=row['camera'];loc=c['motion_evidence']['localization']
            if c['calibration_valid']:
                for line in geo.overlay(np.array(c['calibration']['image_to_pitch']).reshape(3,3),m['width'],m['height']):cv2.line(panel,tuple(line['start']),tuple(line['end']),(255,220,40),max(1,m['width']//540),cv2.LINE_AA)
            if label=='M6':
                motion=c['motion_evidence']['temporal_camera']['field_motion']
                for v in motion.get('vectors',[]):
                    p=tuple(map(round,v['previous']));q=tuple(map(round,v['current']));color=(140,240,110) if v['inlier'] else (100,80,255);cv2.line(panel,p,q,color,1);cv2.circle(panel,q,2,color,1)
                for l in motion.get('lines',[]):
                    if l['kind']=='Side line top':cv2.line(panel,tuple(map(round,l['current'][0])),tuple(map(round,l['current'][1])),(80,255,255),max(1,m['width']//360))
            people=[(e['image_bbox'],e['team'],e['current_pitch']) for e in row['entities'] if e['visibility']=='visible']+[(u['observation']['bbox'],u['observation']['team'],u['observation']['pitch']) for u in row['unassigned']]
            for box,team,point in people:
                x,y,w,h=box;color=(100,130,255) if team=='team_a' else (255,165,110) if team=='team_b' else (220,220,220);cv2.rectangle(panel,(round(x),round(y)),(round(x+w),round(y+h)),color,1)
                if point is not None:cv2.circle(panel,(round(x+w/2),round(y+h)),3,(50,50,255),1)
            panel=cv2.resize(panel,(640,round(m['height']*640/m['width'])));header=np.full((64,640,3),20,np.uint8);cv2.putText(header,f"{label} {m['recording_id']} frame {index} / {index/m['fps']:.3f}s / {loc['status']}",(6,22),cv2.FONT_HERSHEY_SIMPLEX,.46,(235,235,235),1);cv2.putText(header,loc['outcome'][:80],(6,44),cv2.FONT_HERSHEY_SIMPLEX,.4,(235,235,235),1);panels.append(np.vstack([header,panel]))
        pair=np.hstack(panels);cv2.imwrite(str(output/f'frame_{index:03d}_comparison.jpg'),pair,[cv2.IMWRITE_JPEG_QUALITY,90]);tiles.append(cv2.resize(pair,(640,round(pair.shape[0]/2))))
    cap.release()
    while len(tiles)%2:tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(output/'contact_sheet.jpg'),np.vstack([np.hstack(tiles[i:i+2]) for i in range(0,len(tiles),2)]),[cv2.IMWRITE_JPEG_QUALITY,90])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--before',required=True);p.add_argument('--after',required=True);p.add_argument('--output',required=True);p.add_argument('--frames',type=int,nargs='+',required=True);a=p.parse_args();render(a.before,a.after,a.output,a.frames)
