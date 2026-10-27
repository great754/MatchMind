"""Optional offline semantic line inference using TVCalib's approved checkpoint.

Architecture/preprocessing follow the upstream MIT-licensed TVCalib inference
module. No training-stack code, auto-downloads or pickle execution are required.
"""
import hashlib
from pathlib import Path
import cv2
import numpy as np
from .pitch import PitchEvidence,LineEvidence

CLASSES=['Big rect. left bottom','Big rect. left main','Big rect. left top','Big rect. right bottom','Big rect. right main','Big rect. right top','Circle central','Circle left','Circle right','Goal left crossbar','Goal left post left ','Goal left post right','Goal right crossbar','Goal right post left','Goal right post right','Goal unknown','Line unknown','Middle line','Side line bottom','Side line left','Side line right','Side line top','Small rect. left bottom','Small rect. left main','Small rect. left top','Small rect. right bottom','Small rect. right main','Small rect. right top']
# Canonical corner-metres, with no inference about team defending sides.
GEOMETRY={'Big rect. left bottom':(1,54.16),'Big rect. left main':(0,16.5),'Big rect. left top':(1,13.84),
'Big rect. right bottom':(1,54.16),'Big rect. right main':(0,88.5),'Big rect. right top':(1,13.84),
'Middle line':(0,52.5),'Side line bottom':(1,68.),'Side line left':(0,0.),'Side line right':(0,105.),'Side line top':(1,0.),
'Small rect. left bottom':(1,43.16),'Small rect. left main':(0,5.5),'Small rect. left top':(1,24.84),
'Small rect. right bottom':(1,43.16),'Small rect. right main':(0,99.5),'Small rect. right top':(1,24.84)}

def semantic_geometry(mask,confidence,width,height):
    """Class masks -> robust line fits -> guarded identified intersections.

    Curve/goal classes are visible evidence only, never fitted as ground lines.
    Model class scores are heuristic support, not localization probabilities.
    """
    mh,mw=mask.shape;sx=width/mw;sy=height/mh;fits={};lines=[];curves=[]
    for index,name in enumerate(CLASSES,1):
        ys,xs=np.where((mask==index)&(confidence>=.5))
        if len(xs)<12:continue
        if name.startswith('Circle'):
            selected=np.linspace(0,len(xs)-1,min(48,len(xs)),dtype=int)
            curves.append({'kind':name,'points':np.column_stack([xs[selected]*sx,ys[selected]*sy]).tolist(),'support':float(confidence[ys,xs].mean())})
        points=np.column_stack((xs,ys)).astype(np.float32)
        if name not in GEOMETRY:continue
        v=cv2.fitLine(points,cv2.DIST_HUBER,0,.01,.01).reshape(-1);direction=v[:2];center=v[2:]
        residual=np.abs((points[:,0]-center[0])*direction[1]-(points[:,1]-center[1])*direction[0])
        good=residual<2.5
        if good.sum()<12 or good.mean()<.65:continue
        proj=(points[good]-center)@direction;lo,hi=np.quantile(proj,[.02,.98])
        if hi-lo<25:continue
        a=center+lo*direction;b=center+hi*direction
        a=a*np.array([sx,sy]);b=b*np.array([sx,sy]);line=np.cross([*a,1],[*b,1]);line=line/np.linalg.norm(line[:2])
        fits[name]=(line,a,b);lines.append(LineEvidence(start=tuple(a),end=tuple(b),kind=name,support=float(confidence[ys[good],xs[good]].mean())))
    correspondences=[]
    for left,(l,a,b) in fits.items():
        for right,(r,c,d) in fits.items():
            if GEOMETRY[left][0]!=0 or GEOMETRY[right][0]!=1:continue
            p=np.cross(l,r)
            if abs(p[2])<1e-8:continue
            p=p[:2]/p[2]
            if not (0<=p[0]<width and 0<=p[1]<height):continue
            # Intersection must be supported near the observed segments.
            slack=.08*np.hypot(width,height)
            if any(np.linalg.norm(p-s)+np.linalg.norm(p-t)>np.linalg.norm(s-t)+2*slack for s,t in [(a,b),(c,d)]):continue
            correspondences.append((tuple(p),(GEOMETRY[left][1],GEOMETRY[right][1])))
    if len(correspondences)>=4:
        hull=cv2.convexHull(np.array([p[0] for p in correspondences],np.float32))
        if cv2.contourArea(hull)<width*height*.02:correspondences=[]
    return PitchEvidence(status='ambiguous' if lines or curves else 'unavailable',lines=lines,curves=curves,grass_fraction=None,
        reason='Semantic line hypotheses; metric fit requires four spread, consistent intersections'),correspondences

class SemanticPitch:
    def __init__(self,checkpoint,expected_sha256,device='cpu'):
        import torch
        from torchvision.models.segmentation import deeplabv3_resnet101
        from torchvision.transforms import Compose,Resize,ToTensor,Normalize
        h=hashlib.sha256()
        with Path(checkpoint).open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
        if h.hexdigest()!=expected_sha256:raise ValueError('Semantic pitch model hash mismatch')
        if device not in ('cpu','mps'):raise ValueError('Supported experimental devices: cpu/mps')
        self.torch=torch;self.device=device;torch.set_num_threads(4)
        self.model=deeplabv3_resnet101(weights=None,weights_backbone=None,num_classes=29,aux_loss=True)
        from argparse import Namespace
        with torch.serialization.safe_globals([Namespace]):
            state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        self.model.load_state_dict(state['model'],strict=True);self.model.to(device).eval()
        self.preprocess=Compose([Resize(256),ToTensor(),Normalize([.485,.456,.406],[.229,.224,.225])])
    def masks(self,image):
        from PIL import Image
        tensor=self.preprocess(Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            output=self.model(tensor)['out'].softmax(1);confidence,mask=output.max(1)
        return mask[0].cpu().numpy(),confidence[0].cpu().numpy()
    def detect(self,image):
        mask,confidence=self.masks(image)
        return semantic_geometry(mask,confidence,image.shape[1],image.shape[0])
