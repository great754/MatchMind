"""Pretrained DINO ViT-S/16, late-block adaptation and quality-aware temporal attention."""
import cv2,numpy as np,torch
from torch import nn
from torch.nn import functional as F
from .common import ROOT,load,sha

def image_array(image):
 image=image.copy();image[:int(.18*image.shape[0])]=127
 # Preserve 2:1 body aspect in a neutral 224-square canvas.
 h,w=image.shape[:2];scale=min(224/h,224/w);a=cv2.resize(image,(max(1,round(w*scale)),max(1,round(h*scale))));canvas=np.full((224,224,3),127,np.uint8);y=(224-a.shape[0])//2;x=(224-a.shape[1])//2;canvas[y:y+a.shape[0],x:x+a.shape[1]]=a
 a=cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB).transpose(2,0,1).astype(np.float32)/255
 return (a-np.array([.485,.456,.406],np.float32)[:,None,None])/np.array([.229,.224,.225],np.float32)[:,None,None]

def image_tensor(row):
 image=cv2.imread(row['crop_path'])
 if image is None:raise ValueError('Missing crop')
 return image_array(image)

class Visual(nn.Module):
 def __init__(self,classes=51):
  super().__init__()
  import timm
  self.net=timm.create_model('vit_small_patch16_224',pretrained=False,num_classes=0)
  p=ROOT/'dino_deitsmall16_pretrain.pth'
  if sha(p)!=load('initializer.json')['sha256']:raise ValueError('Initializer hash mismatch')
  self.net.load_state_dict(torch.load(p,map_location='cpu',weights_only=True),strict=True)
  self.classifier=nn.Linear(384,classes)
 def prefix(self,x):
  x=self.net.patch_embed(x);x=self.net._pos_embed(x);x=self.net.patch_drop(x);x=self.net.norm_pre(x)
  for block in self.net.blocks[:-1]:x=block(x)
  return x
 def finish(self,tokens):return F.normalize(self.net.norm(self.net.blocks[-1](tokens))[:,0],dim=-1)
 def forward(self,x):return self.finish(self.prefix(x))
 def late(self):
  for p in self.net.parameters():p.requires_grad=False
  for layer in [self.net.blocks[-1],self.net.norm]:
   for p in layer.parameters():p.requires_grad=True

class Temporal(nn.Module):
 def __init__(self,dim=384,classes=51):
  super().__init__();layer=nn.TransformerEncoderLayer(dim,4,dim*2,dropout=.1,batch_first=True,norm_first=True)
  self.encoder=nn.TransformerEncoder(layer,1,enable_nested_tensor=False);self.time=nn.Linear(1,dim);self.attention=nn.Linear(dim,1);self.classifier=nn.Linear(dim,classes)
 def forward(self,x,quality,times,mask=None):
  if mask is None:mask=torch.ones_like(quality,dtype=torch.bool)
  if not mask.any(1).all():raise ValueError('Empty tracklet')
  t=(times-times[:,0:1])/240.;h=self.encoder(x+self.time(t.unsqueeze(-1)),src_key_padding_mask=~mask)
  # Monotone quality prior and bounded learned correction: a frame of quality .1
  # cannot outweigh quality 1 solely through the learned logit (ratio <= exp(1)).
  logits=.5*torch.tanh(self.attention(h).squeeze(-1))+quality.clamp_min(.01).log();logits=logits.masked_fill(~mask,-torch.inf);weights=logits.softmax(1)
  pooled=(weights.unsqueeze(-1)*(x+.1*h)).sum(1);z=F.normalize(pooled,dim=-1)
  return self.classifier(z),z,weights
