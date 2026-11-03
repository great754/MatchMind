"""Saved benchmark misses/unmatched proposals and real-video reference contact sheets."""
import json
from pathlib import Path
import cv2,numpy as np
from .common import ROOT,load,save
from .evaluate import iou

def run():
 c=load('campaign_selection.json');rows=[r for r in load('dataset.json')['samples'] if r['split']=='test'];pred=load('test_cache/generic_tile256.json');output=ROOT/'failure_examples';output.mkdir(exist_ok=True);examples=[]
 for kind in ['miss','unmatched_proposal','multiple_candidates']:
  count=0
  for row,p in zip(rows,pred):
   ps=[d for d in p['candidates'] if d['confidence']>=c['selected']['threshold']];matched=[d for d in ps if any(iou(d['bbox'],b['bbox'])>=.25 for b in row['balls'])];wrong=[d for d in ps if not any(iou(d['bbox'],b['bbox'])>=.25 for b in row['balls'])]
   if kind=='miss' and matched or kind=='unmatched_proposal' and not wrong or kind=='multiple_candidates' and len(ps)<2:continue
   image=cv2.imread(row['path']);crops=[]
   for b in row['balls']:
    x,y,w,h=map(round,b['bbox']);cv2.rectangle(image,(x,y),(x+w,y+h),(255,220,60),2);cx=x+w//2;cy=y+h//2;crop=image[max(0,cy-100):min(1080,cy+100),max(0,cx-150):min(1920,cx+150)];crops.append(cv2.resize(crop,(450,300)))
   for d in ps:
    x,y,w,h=map(round,d['bbox']);cv2.rectangle(image,(x,y),(x+w,y+h),(70,255,70) if d in matched else (60,60,255),2)
   small=cv2.resize(image,(960,540));cv2.putText(small,kind+' | '+row['sample_id']+' | cyan=originalGT,green=IoUmatch,red=unmatched',(8,20),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1);path=output/f'{kind}_{count}.jpg';cv2.imwrite(str(path),small)
   if crops:cv2.imwrite(str(output/f'{kind}_{count}_gt_crop.jpg'),np.hstack(crops))
   examples.append({'type':kind,'sample_id':row['sample_id'],'path':str(path),'source_sha256':row['source_sha256'],'annotation_sha256':row['annotation_sha256'],'visibility_not_independently_labelled':True});count+=1
   if count>=3:break
 save('failure_review.json',{'examples':examples,'note':'Unmatched by IoU is a benchmark failure, not proof that every proposal is semantically false; ball visibility and GT interpolation remain limitations.'})
if __name__=='__main__':run()
