"""Held-out visual examples from fixed measured categories, never converted to new GT."""
from pathlib import Path
import cv2
from ml.ball14.evaluate import iou
from .common import ROOT,load,save

def run():
 rows=[r for r in load('dataset.json')['samples'] if r['split']=='test'];ps=load('cache/test_m16.json');threshold=load('selection.json')['detector']['threshold'];counts={};examples=[];out=ROOT/'failure_examples';out.mkdir(exist_ok=True)
 for row,p in zip(rows,ps,strict=True):
  cs=[c for c in p['candidates'] if c['confidence']>=threshold];matches=[max([iou(c['bbox'],b['bbox']) for b in row['balls']],default=0)>=.25 for c in cs];kind='miss' if row['balls'] and not any(matches) else 'unmatched' if any(not b for b in matches) else 'matched'
  if counts.get(kind,0)>=3:continue
  image=cv2.imread(row['path'])
  for b in row['balls']:
   x,y,w,h=map(int,b['bbox']);cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,0),2)
  for c,good in zip(cs,matches):
   x,y,w,h=map(int,c['bbox']);cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0) if good else (0,0,255),1)
  cv2.putText(image,f'M16 {kind} | SoccerNet {row["sample_id"]} | cyan organizer annotation',(10,26),0,.6,(255,255,255),2);path=out/f'{kind}_{counts.get(kind,0)}.jpg';cv2.imwrite(str(path),image);examples.append({'kind':kind,'sample_id':row['sample_id'],'path':str(path),'visibility_not_independently_verified':True});counts[kind]=counts.get(kind,0)+1
 save('visual_examples.json',{'examples':examples,'note':'Source annotation comparisons, not independent newly labelled truth; unmatched may reflect localization or missing reserve-ball annotation.'})
if __name__=='__main__':run()
