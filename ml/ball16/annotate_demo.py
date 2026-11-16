"""Additional source-only visual references, sealed before M16 model selection."""
import json,cv2,numpy as np
from pathlib import Path
from .common import ROOT,save,sha
LABELS={'team_goal':{30:(451,205),150:(435,240),450:(300,190)},'southampton':{30:(516,289),210:(263,397),270:(506,444),330:(604,508),450:(846,567),510:(1038,553),570:(1102,478)},'test_match':{30:(60,793),90:(810,692),150:(727,567),210:(526,451),270:(453,327),330:(726,348),390:(686,467),450:(784,559)}}
def propose():
 rows=[];tiles=[]
 for name,frames in LABELS.items():
  for frame,(cx,cy) in frames.items():
   path=ROOT/'demo_review'/f'{name}_{frame:04d}.jpg';image=cv2.imread(str(path));x=max(0,cx-24);y=max(0,cy-24);patch=image[y:cy+24,x:cx+24].copy();hsv=cv2.cvtColor(patch,cv2.COLOR_BGR2HSV);mask=((hsv[:,:,2]>175)&((hsv[:,:,1]<110)|((hsv[:,:,0]<40)&(hsv[:,:,1]>70)))).astype('uint8')*255;n,labels,stats,centers=cv2.connectedComponentsWithStats(mask);options=[]
   for k in range(1,n):
    xx,yy,w,h,area=stats[k];distance=np.linalg.norm(centers[k]+[x,y]-[cx,cy])
    if 2<=area and max(w,h)<30 and distance<14:options.append((distance,k))
   if options:
    _,k=min(options);xx,yy,w,h,area=stats[k];bbox=[int(xx+x),int(yy+y),int(w),int(h)];cv2.rectangle(patch,(xx,yy),(xx+w,yy+h),(0,0,255),1)
   else:bbox=None
   tile=cv2.resize(patch,(192,192),interpolation=cv2.INTER_NEAREST);cv2.putText(tile,f'{name} {frame}',(3,17),0,.4,(0,255,255),1);tiles.append(tile);rows.append({'recording_id':name,'frame':frame,'path':str(path),'source_sha256':sha(path),'bbox':bbox,'visibility':'VISIBLE_CANDIDATE_REQUIRES_VISUAL_AUDIT','height':'UNKNOWN','metric_reference':None})
 while len(tiles)%6:tiles.append(np.zeros_like(tiles[0]))
 cv2.imwrite(str(ROOT/'demo_review/proposed_boxes.jpg'),np.vstack([np.hstack(tiles[i:i+6]) for i in range(0,len(tiles),6)]));save('demo_proposed.json',rows)
if __name__=='__main__':propose()
