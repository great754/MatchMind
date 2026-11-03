"""Bounded soccer fine-tuning; no final TEST images are loaded here."""
import random,time,json
from pathlib import Path
import cv2
from .common import ROOT,SEED,load,save,sha
from ml.ball14.evaluate import iou

def crops():
 rows=load('dataset.json')['samples'];rng=random.Random(SEED);counts={};audit=[]
 for split in ['train','validation']:
  count=0
  for row in [r for r in rows if r['split']==split]:
   image=cv2.imread(row['path']);h,w=image.shape[:2];regions=[]
   for ball in row['balls']:
    b=ball['bbox']
    for _ in range(2 if split=='train' else 1):regions.append((max(0,min(w-320,int(b[0]+b[2]/2-rng.uniform(.25,.75)*320))),max(0,min(h-320,int(b[1]+b[3]/2-rng.uniform(.25,.75)*320))),'positive'))
   people=row['people'];person=rng.choice(people) if people else {'bbox':[w/2,h/2,0,0]};b=person['bbox'];regions += [(max(0,min(w-320,int(b[0]+b[2]/2-160))),max(0,min(h-320,int(b[1]+b[3]-160))),'player_foot_hard_negative'),(rng.randrange(w-319),rng.randrange(h-319),'random_background')]
   for j,(x,y,kind) in enumerate(regions):
    labels=[]
    for ball in row['balls']:
     a,b,bw,bh=ball['bbox'];x1=max(0,a-x);y1=max(0,b-y);x2=min(320,a+bw-x);y2=min(320,b+bh-y)
     if x2-x1>=2 and y2-y1>=2:labels.append(f'0 {(x1+x2)/640} {(y1+y2)/640} {(x2-x1)/320} {(y2-y1)/320}')
    stem=row['sample_id'].replace(':','_')+f'_{j}';dest=ROOT/'crops';ip=dest/'images'/split/(stem+'.jpg');lp=dest/'labels'/split/(stem+'.txt');ip.parent.mkdir(parents=True,exist_ok=True);lp.parent.mkdir(parents=True,exist_ok=True);im=image[y:y+320,x:x+320].copy()
    if split=='train':
     im=(im.astype(float)*rng.uniform(.8,1.2)).clip(0,255).astype('uint8')
     if rng.random()<.25:im=cv2.GaussianBlur(im,(3,3),rng.uniform(.4,1.2))
     if rng.random()<.2:
      small=rng.choice([160,224]);im=cv2.resize(cv2.resize(im,(small,small)),(320,320))
     if labels and rng.random()<.12:
      _,cx,cy,bw,bh=map(float,labels[0].split());xx=int((cx-bw/2)*320);yy=int((cy-bh/2)*320);im[yy:yy+max(1,int(bh*80)),xx:xx+max(1,int(bw*80))]=110
    cv2.imwrite(str(ip),im,[cv2.IMWRITE_JPEG_QUALITY,rng.randint(50,95) if split=='train' else 95]);lp.write_text('\n'.join(labels)+'\n');count+=1;audit.append({'sample':row['sample_id'],'kind':kind,'labels':len(labels),'crop':[x,y,320,320],'path':str(ip)})
  counts[split]=count
 data=ROOT/'soccer_ball.yaml';data.write_text(f'path: {(ROOT/"crops").resolve()}\ntrain: images/train\nval: images/validation\nnames:\n  0: soccer_ball\n');save('crop_audit.json',{'counts':counts,'samples':audit,'hard_negative_caution':'Player-foot and background crops retain all organizer ball labels; GT incompleteness may leave unlabeled balls. Not certified invisible-ball labels.'});return data

def run():
 import torch
 from ultralytics import YOLO
 data=crops();start=time.perf_counter();model=YOLO(str(ROOT/'yolo11n.pt'))
 model.train(data=str(data),epochs=12,imgsz=640,batch=8,device='mps' if torch.backends.mps.is_available() else 'cpu',workers=0,project=str(ROOT/'training'),name='soccer11n',exist_ok=False,seed=SEED,deterministic=True,optimizer='AdamW',lr0=.001,lrf=.1,weight_decay=.0005,patience=12,save=True,plots=False,cache=False,amp=False,mosaic=.25,close_mosaic=3,mixup=0.,hsv_h=.01,hsv_s=.2,hsv_v=.2,translate=.1,scale=.25,fliplr=.5,verbose=False)
 p=ROOT/'training/soccer11n/weights/best.pt';save('detector_training.json',{'seconds':time.perf_counter()-start,'best':str(p),'sha256':sha(p),'initializer_sha256':sha(ROOT/'yolo11n.pt'),'dataset_sha256':sha(ROOT/'dataset.json'),'mps_allocated_bytes_snapshot':torch.mps.current_allocated_memory() if torch.backends.mps.is_available() else None,'peak_memory_measured':False})
if __name__=='__main__':run()
