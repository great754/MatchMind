"""Pretrained lightweight two-stage detector with fine ball/foreground head."""
import random,time
import cv2,numpy as np,torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator
from .common import ROOT,load,sha

def make_model(fine=False,checkpoint=None,device='cpu'):
 settings=load('trial_settings.json') if (ROOT/'trial_settings.json').exists() else {}
 payload=torch.load(checkpoint,map_location='cpu',weights_only=True) if checkpoint else None
 if payload is not None:settings=payload.get('trial_settings',settings)
 p=ROOT/'generic_fasterrcnn_mobilenet320.pth'
 if sha(p)!=load('initializer.json')['sha256']:raise ValueError('Pretrained detector hash mismatch')
 m=fasterrcnn_mobilenet_v3_large_320_fpn(weights=None,weights_backbone=None,min_size=384,max_size=384,box_score_thresh=.01,box_nms_thresh=.35,box_detections_per_img=120,rpn_pre_nms_top_n_test=300,rpn_post_nms_top_n_test=100,rpn_pre_nms_top_n_train=600,rpn_post_nms_top_n_train=200,box_batch_size_per_image=settings.get('roi_batch',128),rpn_batch_size_per_image=128)
 m.load_state_dict(torch.load(p,map_location='cpu',weights_only=True),strict=True)
 if fine:
  old=m.roi_heads.box_predictor;new=FastRCNNPredictor(old.cls_score.in_features,2)
  with torch.no_grad():
   for target,source in [(0,0),(1,37)]:new.cls_score.weight[target]=old.cls_score.weight[source];new.cls_score.bias[target]=old.cls_score.bias[source];new.bbox_pred.weight[4*target:4*target+4]=old.bbox_pred.weight[4*source:4*source+4];new.bbox_pred.bias[4*target:4*target+4]=old.bbox_pred.bias[4*source:4*source+4]
  m.roi_heads.box_predictor=new
  # Keep pretrained RPN tensor shapes (15anchors/location), shift scales for tiny balls.
  if not settings.get('preserve_anchor_sizes',False):m.rpn.anchor_generator=AnchorGenerator(((4,8,16,32,64),(8,16,32,64,128),(16,32,64,128,256)),((.5,1.,2.),)*3)
 if checkpoint:
  m.load_state_dict(payload['model'],strict=True)
 return m.to(device)

def image_tensor(image):return torch.from_numpy(cv2.cvtColor(image,cv2.COLOR_BGR2RGB).transpose(2,0,1).copy()).float()/255

def tiles(width,height,size=480,overlap=.2):
 def starts(n):
  if n<=size:return [0]
  points=list(range(0,n-size+1,max(1,int(size*(1-overlap)))))
  if points[-1]!=n-size:points.append(n-size)
  return points
 return [(x,y,min(size,width-x),min(size,height-y)) for y in starts(height) for x in starts(width)]

def infer(model,image,fine=False,tiled=False,device='cpu',tile_size=480,minimum=.02):
 from torchvision.ops import nms
 h,w=image.shape[:2];regions=[(0,0,w,h)]+(tiles(w,h,tile_size) if tiled else []);regions=list(dict.fromkeys(regions));boxes=[];scores=[];views=[];start=time.perf_counter()
 with torch.inference_mode():
  for i in range(0,len(regions),4):
   rs=regions[i:i+4];out=model([image_tensor(image[y:y+hh,x:x+ww]).to(device) for x,y,ww,hh in rs])
   for j,(r,p) in enumerate(zip(rs,out)):
    b=p['boxes'].cpu().numpy();s=p['scores'].cpu().numpy();labels=p['labels'].cpu().numpy()
    for box,score,label in zip(b,s,labels):
     if label!=(1 if fine else 37) or score<minimum:continue
     x1,y1,x2,y2=box+np.array([r[0],r[1],r[0],r[1]]);x1=max(0,float(x1));y1=max(0,float(y1));x2=min(w,float(x2));y2=min(h,float(y2))
     if x2-x1<1 or y2-y1<1 or max(x2-x1,y2-y1)>.15*max(w,h):continue
     boxes.append([x1,y1,x2,y2]);scores.append(float(score));views.append('full' if r==(0,0,w,h) else f'tile:{r[0]}:{r[1]}')
 if boxes:
  keep=nms(torch.tensor(boxes),torch.tensor(scores),.3).tolist()[:30]
 else:keep=[]
 result=[{'candidate_id':f'ball_{i}','bbox':[boxes[k][0],boxes[k][1],boxes[k][2]-boxes[k][0],boxes[k][3]-boxes[k][1]],'confidence':scores[k],'inference_view':views[k],'source':'VIDEO_CV_DETECTOR'} for i,k in enumerate(keep)]
 return result,{'seconds':time.perf_counter()-start,'views':len(regions)}

def training_item(row,rng):
 image=cv2.imread(row['path']);h,w=image.shape[:2];settings=load('trial_settings.json') if (ROOT/'trial_settings.json').exists() else {};size=settings.get('training_tile',320)
 if row['balls'] and rng.random()<.8:
  b=rng.choice(row['balls'])['bbox'];cx=b[0]+b[2]/2;cy=b[1]+b[3]/2;x=int(np.clip(cx-rng.uniform(.2,.8)*size,0,w-size));y=int(np.clip(cy-rng.uniform(.2,.8)*size,0,h-size))
 else:x=rng.randrange(w-size+1);y=rng.randrange(h-size+1)
 crop=image[y:y+size,x:x+size].copy();boxes=[]
 for ball in row['balls']:
  a,b,ww,hh=ball['bbox'];x1=max(0,a-x);y1=max(0,b-y);x2=min(size,a+ww-x);y2=min(size,b+hh-y)
  if x2-x1>=2 and y2-y1>=2:boxes.append([x1,y1,x2,y2])
 crop=np.clip(crop.astype(float)*rng.uniform(.8,1.2)+rng.uniform(-12,12),0,255).astype(np.uint8)
 if rng.random()<.25:
  k=rng.choice([3,5]);kernel=np.zeros((k,k),np.float32);kernel[k//2,:]=1/k;crop=cv2.filter2D(crop,-1,kernel)
 if rng.random()<.25:
  small=rng.choice([160,224,288]);crop=cv2.resize(cv2.resize(crop,(small,small)),(size,size))
 if rng.random()<.25:
  ok,encoded=cv2.imencode('.jpg',crop,[cv2.IMWRITE_JPEG_QUALITY,rng.randint(40,90)]);crop=cv2.imdecode(encoded,cv2.IMREAD_COLOR)
 if boxes and rng.random()<.15:
  # Occlude only a small part of a labelled box, retaining its annotated location.
  a,b,c,d=rng.choice(boxes);xx=int(a);yy=int(b);crop[yy:yy+max(1,int((d-b)*.25)),xx:xx+max(1,int((c-a)*.25))]=110
 if rng.random()<.5:
  crop=cv2.flip(crop,1);boxes=[[size-c,b,size-a,d] for a,b,c,d in boxes]
 return image_tensor(crop),{'boxes':torch.tensor(boxes,dtype=torch.float32).reshape(-1,4),'labels':torch.ones(len(boxes),dtype=torch.int64)}
