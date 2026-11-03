"""OSNet initialization and body-only crop strategies; offline training only."""
import random
import cv2,numpy as np,torch
from torch import nn
from src.vision.world.vendor.osnet import osnet_x0_25
from src.vision.world.player_appearance import MODEL_SHA256,digest
from .dataset import crop_regions

class SoccerReID(nn.Module):
 def __init__(self,classes=1,strategy='m9_body',generic_checkpoint='outputs/models/m9/osnet_x0_25_msmt17.pth'):
  super().__init__();self.strategy=strategy
  if digest(generic_checkpoint)!=MODEL_SHA256:raise ValueError('Generic initializer hash mismatch')
  self.backbone=osnet_x0_25(num_classes=4101,pretrained=False);self.backbone.load_state_dict(torch.load(generic_checkpoint,map_location='cpu',weights_only=True),strict=True);self.backbone.classifier=nn.Identity();self.classifier=nn.Linear(512,classes)
 def forward(self,x):
  # x: batch x region x RGB x H x W; share backbone across regions.
  b,r,c,h,w=x.shape;v=self.backbone.featuremaps(x.reshape(b*r,c,h,w));v=self.backbone.global_avgpool(v).view(b*r,-1);v=self.backbone.fc(v);v=v.reshape(b,r,-1).mean(1);return self.classifier(v),torch.nn.functional.normalize(v,dim=1)
 def train_late(self):
  for p in self.backbone.parameters():p.requires_grad=False
  for layer in [self.backbone.conv4,self.backbone.conv5,self.backbone.fc]:
   for p in layer.parameters():p.requires_grad=True
  self.train();self.backbone.conv1.eval();self.backbone.conv2.eval();self.backbone.conv3.eval()

def image_tensor(row,strategy,augment=False,rng=None):
 image=cv2.imread(row['crop_path'])
 if image is None:raise ValueError('Missing crop')
 if augment:
  rng=rng or random;image=np.clip(image.astype(float)*rng.uniform(.85,1.15)+rng.uniform(-8,8),0,255).astype(np.uint8);hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV);hsv[:,:,1]=np.clip(hsv[:,:,1].astype(float)*rng.uniform(.85,1.15),0,255);image=cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
  if rng.random()<.15:image=cv2.GaussianBlur(image,(3,3),0)
  if rng.random()<.25:
   h,w=image.shape[:2];factor=rng.uniform(.6,.95);image=cv2.resize(cv2.resize(image,(max(3,int(w*factor)),max(3,int(h*factor)))),(w,h))
  if rng.random()<.5:image=cv2.flip(image,1)
  if rng.random()<.15:
   h,w=image.shape[:2];y=rng.randrange(max(1,int(.8*h)));x=rng.randrange(max(1,int(.8*w)));image[y:y+max(1,int(.1*h)),x:x+max(1,int(.2*w))]=127
 values=[]
 for region in crop_regions(image,strategy):
  a=cv2.resize(cv2.cvtColor(region,cv2.COLOR_BGR2RGB),(128,256)).transpose(2,0,1).astype(np.float32)/255;values.append((a-np.array([.485,.456,.406],np.float32)[:,None,None])/np.array([.229,.224,.225],np.float32)[:,None,None])
 return np.stack(values)

def choose_device():return 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
def seed_all(seed):random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(4)
