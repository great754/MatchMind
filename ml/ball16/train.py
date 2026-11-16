"""Positive-only, multi-match ball fine-tune; no unknown-visibility negative frames."""
import json,time,random,resource
from pathlib import Path
import cv2,numpy as np,requests,torch
from .common import ROOT,load,save,sha,SEED

def prepare():
 rows=load('dataset.json')['samples'];rng=random.Random(SEED);counts={};audit=[]
 for split in ['train','validation']:
  count=0
  for r in [r for r in rows if r['split']==split]:
   im=cv2.imread(r['path']);h,w=im.shape[:2];ball=rng.choice(r['balls']);b=ball['bbox'];size=rng.choice([256,320,480]) if split=='train' else 320;x=int(np.clip(b[0]+b[2]/2-rng.uniform(.25,.75)*size,0,w-size));y=int(np.clip(b[1]+b[3]/2-rng.uniform(.25,.75)*size,0,h-size));crop=im[y:y+size,x:x+size].copy();labels=[]
   for b in r['balls']:
    a,bb,bw,bh=b['bbox'];x1=max(0,a-x);y1=max(0,bb-y);x2=min(size,a+bw-x);y2=min(size,bb+bh-y)
    if x2>x1 and y2>y1:labels.append(f'0 {(x1+x2)/(2*size)} {(y1+y2)/(2*size)} {(x2-x1)/size} {(y2-y1)/size}')
   if not labels:raise ValueError('Positive crop lost labels')
   if split=='train':
    crop=np.clip(crop.astype(float)*rng.uniform(.7,1.3),0,255).astype('uint8')
    if rng.random()<.3:
     kernel=np.zeros((5,5),np.float32);kernel[2]=.2;crop=cv2.filter2D(crop,-1,kernel)
    if rng.random()<.15:
     _,cx,cy,bw,bh=map(float,labels[0].split());xx=max(0,int((cx-bw/2)*size));yy=max(0,int((cy-bh/2)*size));crop[yy:yy+max(1,int(bh*size*.2)),xx:xx+max(1,int(bw*size*.2))]=110
   stem=r['sample_id'].replace(':','_');p=ROOT/'crops/images'/split/(stem+'.jpg');q=ROOT/'crops/labels'/split/(stem+'.txt');p.parent.mkdir(parents=True,exist_ok=True);q.parent.mkdir(parents=True,exist_ok=True);cv2.imwrite(str(p),crop,[cv2.IMWRITE_JPEG_QUALITY,rng.randint(45,95) if split=='train' else 95]);q.write_text('\n'.join(labels)+'\n');count+=1;audit.append({'sample_id':r['sample_id'],'crop':[x,y,size,size],'labels':len(labels)})
  counts[split]=count
 save('crop_audit.json',{'counts':counts,'samples':audit,'negative_frames':0,'sampling':'one positive crop/frame, sizes256/320/480, small source balls enlarged; no arbitrary random/person crops labeled negative'});p=ROOT/'ball.yaml';p.write_text(f'path: {(ROOT/"crops").resolve()}\ntrain: images/train\nval: images/validation\nnames:\n  0: soccer_ball\n');return p

def run():
 load('failure_taxonomy.json')
 from ultralytics import YOLO
 p=ROOT/'yolo11s.pt'
 if not p.exists():
  url='https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt';r=requests.get(url,timeout=60);r.raise_for_status()
  if len(r.content)>30000000:raise ValueError('Checkpoint size exceeds bounded limit')
  p.write_bytes(r.content)
 save('initializer.json',{'architecture':'YOLO11s COCO, soccer fine-tune','url':'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt','sha256':sha(p),'bytes':p.stat().st_size,'license':'Ultralytics AGPL-3.0 local experiment, no commercial deployment right asserted'})
 data=prepare();config={'epochs':4,'patience':2,'batch':8,'imgsz':640,'optimizer':'AdamW','lr0':.0005,'lrf':.1,'weight_decay':.0005,'seed':SEED,'workers':0,'amp':False,'mosaic':.1,'close_mosaic':1,'hsv_h':.01,'hsv_s':.2,'hsv_v':.2,'scale':.35,'fliplr':.5,'plots':False,'cache':False};save('training_config.json',config);start=time.perf_counter();model=YOLO(str(p));model.train(data=str(data),device='mps' if torch.backends.mps.is_available() else 'cpu',project=str(ROOT/'training'),name='soccer11s',exist_ok=False,**config);best=ROOT/'training/soccer11s/weights/best.pt';save('training_result.json',{'seconds':time.perf_counter()-start,'checkpoint':str(best),'checkpoint_sha256':sha(best),'dataset_sha256':sha(ROOT/'dataset.json'),'mps_allocated_snapshot_bytes':torch.mps.current_allocated_memory() if torch.backends.mps.is_available() else None,'process_max_rss_bytes_macos':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'rss_scope':'whole isolated training process on macOS; MPS unified-memory allocations separately reported'})
if __name__=='__main__':run()
