"""Tiny visual-pair CNN experiment using source crops, not coordinate-only labels."""
import json,random,time
from pathlib import Path
import cv2,numpy as np,torch
from torch import nn
from .common import ROOT,save,SEED,load,sha
class VisualPair(nn.Module):
 def __init__(self):
  super().__init__();self.encoder=nn.Sequential(nn.Conv2d(3,12,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(12,24,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(1));self.head=nn.Sequential(nn.Linear(72,24),nn.ReLU(),nn.Linear(24,1))
 def forward(self,x):
  a=self.encoder(x[:,:3]).flatten(1);b=self.encoder(x[:,3:]).flatten(1);return self.head(torch.cat([a,b,abs(a-b)],1)).squeeze(1)
def crop(image,bbox,size=48):
 x,y,w,h=bbox;radius=max(10,max(w,h));cx=x+w/2;cy=y+h/2;x1=max(0,int(cx-radius));y1=max(0,int(cy-radius));x2=min(image.shape[1],int(cx+radius));y2=min(image.shape[0],int(cy+radius));patch=image[y1:y2,x1:x2]
 if not patch.size:raise ValueError('Empty visual crop')
 return cv2.resize(cv2.cvtColor(patch,cv2.COLOR_BGR2RGB),(size,size))
def pair(previous,current):return torch.from_numpy(np.concatenate([previous,current],2).transpose(2,0,1).copy()).float()/255

def run():
 torch.manual_seed(SEED);torch.set_num_threads(2);r=json.loads(Path('outputs/ml/ball_m15/dataset.json').read_text())['samples'];train=[x for x in r if x['split']=='train'];lookup={(x['recording_id'],x['frame']):x for x in train};positives=[];negatives=[]
 for row in train:
  old=lookup.get((row['recording_id'],row['frame']-1))
  if not old or len(row['balls'])!=1 or len(old['balls'])!=1:continue
  a=crop(cv2.imread(old['path']),old['balls'][0]['bbox']);b=crop(cv2.imread(row['path']),row['balls'][0]['bbox']);positives.append(pair(a,b))
 reviewed=load('verified_negative_regions.json')['rows']
 for n in reviewed:
  source=next(x for x in train if x['sample_id']==n['source_sample']);old=lookup.get((source['recording_id'],source['frame']-1))
  if old is None or len(old['balls'])!=1:continue
  im=cv2.imread(source['path']);prior=crop(cv2.imread(old['path']),old['balls'][0]['bbox']);x,y,w,h=n['crop'];negative=cv2.resize(cv2.cvtColor(im[y:y+h,x:x+w],cv2.COLOR_BGR2RGB),(48,48));negatives.append(pair(prior,negative))
 # Source-match split for verified negative visual validation is absent: do not calibrate/activate on held-out TEST.
 model=VisualPair();optimizer=torch.optim.AdamW(model.parameters(),lr=.001);rng=random.Random(SEED);losses=[];start=time.perf_counter()
 for epoch in range(20):
  xx=[];yy=[]
  for _ in range(128):xx.extend([rng.choice(positives),rng.choice(negatives)]);yy.extend([1.,0.])
  for start_batch in range(0,len(xx),32):
   x=torch.stack(xx[start_batch:start_batch+32]);y=torch.tensor(yy[start_batch:start_batch+32]);loss=nn.functional.binary_cross_entropy_with_logits(model(x),y);optimizer.zero_grad();loss.backward();optimizer.step();losses.append(float(loss.detach()))
 p=ROOT/'temporal_visual.pth';torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'seed':SEED,'architecture':'shared CNN12/24 + concatenation(current,previous,absdiff) + MLP24','verified_negative_manifest_sha256':sha(ROOT/'verified_negative_regions.json')},p);save('temporal_result.json',{'positive_pairs':len(positives),'reviewed_negative_regions':len(negatives),'seconds':time.perf_counter()-start,'checkpoint_sha256':sha(p),'activated':False,'blocker':'No independent match-disjoint verified-negative temporal validation labels; organizer positives have unverified visibility. Cannot certify validation F1/coverage improvement. Training loss is not accuracy.','final_training_loss':losses[-1]});print('VISUAL_PAIR',len(positives),len(negatives),'not activated',flush=True)
if __name__=='__main__':run()

def visual_replay(rows,predictions,threshold):
 from src.vision.ball16.tracker import MultiHypothesisTracker
 model=VisualPair();model.load_state_dict(torch.load(ROOT/'temporal_visual.pth',map_location='cpu',weights_only=True)['model']);model.eval();states=[];previous=None;last=None;tracker=None
 for r,p in zip(rows,predictions,strict=True):
  if r['recording_id']!=last:tracker=MultiHypothesisTracker(r['width'],r['height'],threshold);previous=None;last=r['recording_id']
  image=cv2.imread(r['path'])
  def verifier(candidate):
   current=crop(image,candidate['bbox']);prior=previous if previous is not None else current
   with torch.no_grad():return float(model(pair(prior,current)[None]).sigmoid()[0])
  state=tracker.step(r['frame'],r['timestamp'],p['candidates'],r.get('camera',{'epoch':0,'calibration_valid':False}),verifier=verifier)
  if state.directly_observed:previous=crop(image,state.image_bbox)
  if r.get('camera',{}).get('continuity')=='cut':previous=None
  states.append(state.model_dump())
 return states
