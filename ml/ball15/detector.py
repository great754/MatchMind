"""Offline AGPL YOLO research backend, with uncertainty-guided high-resolution crops."""
import time,math
import numpy as np
from .common import ROOT,load
from ml.ball14.model import tiles

def make_model(path=None):
 if path and 'generic_fasterrcnn_mobilenet320' in str(path):
  import torch
  from ml.ball14.model import make_model as oldmodel
  model=oldmodel(device='mps' if torch.backends.mps.is_available() else 'cpu').eval();model._tactiq_m14=True;return model
 from ultralytics import YOLO
 return YOLO(str(path or ROOT/'yolo11n.pt'))

def infer(model,image,tile=640,region=None,generic=False):
 from torchvision.ops import nms
 import torch
 if getattr(model,'_tactiq_m14',False):
  from ml.ball14.model import infer as oldinfer
  device='mps' if torch.backends.mps.is_available() else 'cpu'
  if region is None:return oldinfer(model,image,False,True,device,tile)
  x,y,w,h=region;cs,timing=oldinfer(model,image[y:y+h,x:x+w],False,False,device,tile)
  for c in cs:c['bbox'][0]+=x;c['bbox'][1]+=y;c['inference_view']='local_prediction'
  return cs,timing
 h,w=image.shape[:2];regions=[(0,0,w,h)]+tiles(w,h,tile) if region is None else [region];regions=list(dict.fromkeys(regions));boxes=[];scores=[];views=[];start=time.perf_counter()
 for begin in range(0,len(regions),8):
  rs=regions[begin:begin+8];images=[image[y:y+hh,x:x+ww] for x,y,ww,hh in rs];results=model.predict(images,imgsz=640,conf=.02,iou=.4,classes=[32] if generic else None,device='mps' if torch.backends.mps.is_available() else 'cpu',verbose=False,max_det=30)
  for r,out in zip(rs,results,strict=True):
   for b,s in zip(out.boxes.xyxy.cpu().numpy(),out.boxes.conf.cpu().numpy()):
    x1,y1,x2,y2=b+np.array([r[0],r[1],r[0],r[1]]);x1=max(0,float(x1));y1=max(0,float(y1));x2=min(w,float(x2));y2=min(h,float(y2))
    if x2-x1<1 or y2-y1<1 or max(x2-x1,y2-y1)>.15*max(w,h):continue
    boxes.append([x1,y1,x2,y2]);scores.append(float(s));views.append('local_prediction' if region is not None else 'full' if r==(0,0,w,h) else f'tile:{r[0]}:{r[1]}')
 keep=nms(torch.tensor(boxes),torch.tensor(scores),.3).tolist()[:30] if boxes else []
 return [{'candidate_id':f'ball_{i}','bbox':[boxes[k][0],boxes[k][1],boxes[k][2]-boxes[k][0],boxes[k][3]-boxes[k][1]],'confidence':scores[k],'inference_view':views[k],'source':'VIDEO_CV_DETECTOR'} for i,k in enumerate(keep)],{'seconds':time.perf_counter()-start,'views':len(regions)}

def search_region(point,sigma,age,width,height):
 if point is None or sigma is None:return None
 radius=max(64,3*sigma+age*max(width,height)*.25);size=min(max(width,height),int(math.ceil(radius*2)));x=max(0,min(width-size,int(point[0]-size/2)));y=max(0,min(height-size,int(point[1]-size/2)));return x,y,min(size,width-x),min(size,height-y)
