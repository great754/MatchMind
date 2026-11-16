"""M16 higher-resolution soccer detector and additive local recovery."""
import time,math
import cv2,numpy as np,torch
from ml.ball14.model import tiles
from .common import ROOT,load,sha

def make_model(path=None):
 from ultralytics import YOLO
 return YOLO(str(path or load('training_result.json')['checkpoint']))
def infer(model,image,resolution=640,region=None):
 from torchvision.ops import nms
 start=time.perf_counter();h,w=image.shape[:2];regions=([region] if region else [(0,0,w,h)]+tiles(w,h,640));regions=list(dict.fromkeys(regions));boxes=[];scores=[];views=[]
 for start_i in range(0,len(regions),4):
  rs=regions[start_i:start_i+4];outputs=model.predict([image[y:y+hh,x:x+ww] for x,y,ww,hh in rs],imgsz=resolution,conf=.02,iou=.4,verbose=False,device='mps' if torch.backends.mps.is_available() else 'cpu',max_det=40)
  for r,out in zip(rs,outputs,strict=True):
   for b,s in zip(out.boxes.xyxy.cpu().numpy(),out.boxes.conf.cpu().numpy()):
    a,bb,c,d=b+np.array([r[0],r[1],r[0],r[1]]);a=max(0,float(a));bb=max(0,float(bb));c=min(w,float(c));d=min(h,float(d))
    if c-a<1 or d-bb<1 or max(c-a,d-bb)>.15*max(w,h):continue
    boxes.append([a,bb,c,d]);scores.append(float(s));views.append('local_additive' if region else 'global_tiled')
 keep=nms(torch.tensor(boxes),torch.tensor(scores),.3).tolist()[:40] if boxes else []
 return [{'candidate_id':f'c{i}','bbox':[boxes[k][0],boxes[k][1],boxes[k][2]-boxes[k][0],boxes[k][3]-boxes[k][1]],'confidence':scores[k],'inference_view':views[k],'source':'VIDEO_CV_DETECTOR'} for i,k in enumerate(keep)],{'seconds':time.perf_counter()-start,'views':len(regions)}
def merge_additive(global_candidates,local_candidates):
 from ml.ball14.evaluate import iou
 result=[dict(c) for c in global_candidates]
 for c in local_candidates:
  if not any(iou(c['bbox'],g['bbox'])>.3 for g in global_candidates):result.append(dict(c))
 for i,c in enumerate(result):c['candidate_id']=f'c{i}'
 return result

def recovery_region(point,radius,width,height):
 if point is None:return None
 size=int(min(max(width,height),max(256,min(480,2*(radius or 64)))));x=max(0,min(width-size,int(point[0]-size/2)));y=max(0,min(height-size,int(point[1]-size/2)));return x,y,min(size,width-x),min(size,height-y)
