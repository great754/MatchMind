"""Controlled OSNet widths with person-ReID initialization; small learned pooling."""
import random,cv2,numpy as np,torch
from torch import nn
from torch.nn import functional as F
from ml.reid.model import SoccerReID,image_tensor
from src.vision.world.vendor.osnet import osnet_x1_0
from .common import ROOT,load,sha

class ReID(SoccerReID):
 def __init__(self,classes,width='x0_25'):
  self.width=width
  if width=='x0_25':super().__init__(classes,'full_masked');return
  nn.Module.__init__(self);self.strategy='full_masked';p=ROOT/'osnet_x1_0_msmt17.pth'
  if sha(p)!=load('alternative_initializer.json')['sha256']:raise ValueError('Alternative initializer hash mismatch')
  self.backbone=osnet_x1_0(num_classes=4101,pretrained=False);state=torch.load(p,map_location='cpu',weights_only=True);state=state.get('state_dict',state);self.backbone.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()},strict=True);self.backbone.classifier=nn.Identity();self.classifier=nn.Linear(512,classes)

class Pooler(nn.Module):
 def __init__(self):super().__init__();self.attention=nn.Sequential(nn.Linear(512,64),nn.Tanh(),nn.Linear(64,1))
 def forward(self,x,quality,mask=None):
  if mask is None:mask=torch.ones_like(quality,dtype=torch.bool)
  if not mask.any(1).all():raise ValueError('Empty tracklet')
  logits=.5*torch.tanh(self.attention(x).squeeze(-1))+quality.clamp_min(.01).log();weights=logits.masked_fill(~mask,-torch.inf).softmax(1);return F.normalize((x*weights.unsqueeze(-1)).sum(1),dim=-1),weights

def supcon(z,labels,teams,temperature=.1):
 z=F.normalize(z,dim=-1);scores=z@z.T/temperature;diagonal=torch.eye(len(z),device=z.device,dtype=torch.bool);positive=(labels[:,None]==labels[None,:])&~diagonal
 # Same-team negatives retained alongside all positives; unrelated kits cannot dominate.
 eligible=((teams[:,None]==teams[None,:])|positive)&~diagonal;valid=positive.any(1)&eligible.any(1)
 if not valid.any():return z.sum()*0
 logp=scores-scores.masked_fill(~eligible,-torch.inf).logsumexp(1,keepdim=True);terms=torch.where(positive,logp,0).sum(1)/positive.sum(1).clamp_min(1);return -terms[valid].mean()

def augmented_tensor(row,rng):
 x=image_tensor(row,'full_masked',True,rng)
 # Mild scale variation on already head-masked normalized body layout, no new source pixels.
 if rng.random()<.3:
  scale=rng.uniform(.85,1.);a=x[0].transpose(1,2,0);h,w=a.shape[:2];small=cv2.resize(a,(max(1,int(w*scale)),max(1,int(h*scale))));canvas=np.zeros_like(a);y=(h-small.shape[0])//2;x0=(w-small.shape[1])//2;canvas[y:y+small.shape[0],x0:x0+small.shape[1]]=small;x=canvas.transpose(2,0,1)[None]
 return x.astype(np.float32)
