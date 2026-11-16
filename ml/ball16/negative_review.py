"""Propose hard-negative crops; never label them until source-only visual review."""
import json,math
from pathlib import Path
import cv2,numpy as np
from ml.ball14.evaluate import iou
from .common import ROOT,save,sha

def run():
 rows=[r for r in json.loads(Path('outputs/ml/ball_m15/dataset.json').read_text())['samples'] if r['split']=='train'];ps=json.loads(Path('outputs/ml/ball_m15/cache/train_selected.json').read_text());out=[];tiles=[]
 for r,p in zip(rows,ps,strict=True):
  for c in p['candidates']:
   if c['confidence']<.2 or max([iou(c['bbox'],b['bbox']) for b in r['balls']],default=0)>.01:continue
   b=c['bbox'];x,y=b[0]+b[2]/2,b[1]+b[3]/2
   if min([math.dist([x,y],[a['bbox'][0]+a['bbox'][2]/2,a['bbox'][1]+a['bbox'][3]/2]) for a in r['balls']],default=0)<100:continue
   im=cv2.imread(r['path']);xx=max(0,int(x)-16);yy=max(0,int(y)-16);crop=im[yy:yy+32,xx:xx+32]
   if crop.shape[:2]!=(32,32):continue
   index=len(out);out.append({'id':index,'source_sample':r['sample_id'],'match_id':r['match_id'],'path':r['path'],'source_sha256':r['source_sha256'],'crop':[xx,yy,32,32],'suggestion':'neural FP proposal, not a label until reviewed'});tile=cv2.resize(crop,(128,128));cv2.putText(tile,str(index),(4,18),0,.6,(0,255,255),2);tiles.append(tile)
   if len(out)>=48:break
  if len(out)>=48:break
 while len(tiles)%8:tiles.append(np.zeros_like(tiles[0]))
 cv2.imwrite(str(ROOT/'negative_candidates.jpg'),np.vstack([np.hstack(tiles[i:i+8]) for i in range(0,len(tiles),8)]));save('negative_candidates.json',out)
if __name__=='__main__':run()
