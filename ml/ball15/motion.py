"""Verified organizer trajectories, compact GRU versus image CV/CA, sealed validation calibration."""
import copy,time
from collections import defaultdict
import numpy as np,torch
from torch import nn
from src.vision.ball15.motion import features,extrapolate
from .common import ROOT,SEED,load,save,sha
HORIZONS=[1,3,5,10]
class Predictor(nn.Module):
 def __init__(self):
  super().__init__();self.gru=nn.GRU(8,24,batch_first=True);self.head=nn.Linear(24,12)
 def forward(self,x):
  _,h=self.gru(x);out=self.head(h[-1]);return out[:,:8].reshape(-1,4,2),out[:,8:].clamp(-3,5)

def samples(rows):
 sequences=defaultdict(list)
 for r in rows:
  if len(r['balls'])==1:sequences[r['recording_id']].append(r)
 out=[]
 for seq,rs in sequences.items():
  lookup={r['frame']:r for r in rs}
  for r in rs:
   ids=[r['frame']-i for i in range(7,-1,-1)]+[r['frame']+h for h in HORIZONS]
   if any(i not in lookup for i in ids):continue
   selected=[lookup[i] for i in ids];balls=[s['balls'][0] for s in selected]
   if len({b['track_id'] for b in balls})!=1:continue
   points=np.array([[b['bbox'][0]+b['bbox'][2]/2,b['bbox'][1]+b['bbox'][3]/2] for b in balls]);diag=np.hypot(r['width'],r['height'])
   if np.max(np.linalg.norm(np.diff(points[:8],axis=0),axis=1))>diag*.08:continue
   times=np.array([s['timestamp'] for s in selected]);x=features(points[:8],times[:8],np.ones(8),diag);y=(points[8:]-points[7])/diag*1000
   out.append({'sample_id':r['sample_id'],'recording_id':seq,'frame':r['frame'],'x':x,'y':y.astype('float32'),'points':points,'times':times,'diag':diag})
 return out

def errors(model,ss,kind):
 if kind=='gru':
  with torch.no_grad():delta,log=model(torch.tensor(np.stack([s['x'] for s in ss])));delta=delta.numpy();sigma=np.exp(log.numpy())
 else:delta=sigma=None
 result=[]
 for i,s in enumerate(ss):
  estimates=s['points'][7]+delta[i]/1000*s['diag'] if kind=='gru' else np.array([extrapolate(s['points'][:8],s['times'][:8],s['times'][8+j],kind) for j in range(4)]);e=np.linalg.norm(estimates-s['points'][8:],axis=1);v1=s['points'][7]-s['points'][6];v2=s['points'][8]-s['points'][7];turn=float(np.dot(v1,v2)/max(np.linalg.norm(v1)*np.linalg.norm(v2),1e-9))<.5
  result.append({'sample_id':s['sample_id'],'errors':e.tolist(),'sigma':(sigma[i]/1000*s['diag']).tolist() if kind=='gru' else None,'sudden_direction_change':bool(turn)})
 return result

def run():
 if (ROOT/'motion_lock.json').exists():raise ValueError('Motion selection sealed')
 torch.manual_seed(SEED);np.random.seed(SEED);torch.set_num_threads(2);rows=load('dataset.json')['samples'];train=samples([r for r in rows if r['split']=='train']);val=samples([r for r in rows if r['split']=='validation']);model=Predictor();opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.0001);x=torch.tensor(np.stack([s['x'] for s in train]));y=torch.tensor(np.stack([s['y'] for s in train]));best=float('inf');state=None;history=[];start=time.perf_counter()
 for epoch in range(80):
  model.train();order=torch.randperm(len(x))
  for ix in order.split(32):
   mean,log=model(x[ix]);loss=((mean-y[ix]).square().sum(-1)/(2*torch.exp(2*log))+2*log).mean();opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
  model.eval();e=errors(model,val,'gru');score=float(np.mean([r['errors'] for r in e]));history.append({'epoch':epoch+1,'validation_mean_error_px':score})
  if score<best:best=score;state=copy.deepcopy(model.state_dict());bestepoch=epoch+1
 model.load_state_dict(state);p=ROOT/'motion_gru.pth';torch.save({'model':state,'architecture':'GRU8x24->12, 4horizons xy + isotropic sigma','epoch':bestepoch,'seed':SEED,'dataset_sha256':sha(ROOT/'dataset.json')},p);reports={}
 for kind in ['velocity','acceleration','gru']:
  es=errors(model,val,kind);arr=np.array([r['errors'] for r in es]);sigma=np.array([r['sigma'] for r in es]) if kind=='gru' else np.ones_like(arr);q=np.quantile(arr/np.maximum(sigma,1),.9,axis=0);reports[kind]={'samples':len(es),'mean_error_px':arr.mean(0).tolist(),'median_error_px':np.median(arr,0).tolist(),'p90_error_px':np.quantile(arr,.9,axis=0).tolist(),'radius90_scale':q.tolist(),'validation_rows':es}
 chosen=min(reports,key=lambda k:np.mean(reports[k]['mean_error_px']));save('motion_lock.json',{'selected':chosen,'horizons_frames':HORIZONS,'validation':reports,'checkpoint_sha256':sha(p),'train_samples':len(train),'validation_samples':len(val),'training_seconds':time.perf_counter()-start,'history':history,'ground_truth_limits':'original GT coordinates, visibility unannotated; no predicted labels or manual demo data in training','uncertainty':'90% radius calibrated on VAL per horizon, coverage must be tested prospectively','motion_state':'direction change from image derivatives, not a football action label'});print('MOTION',chosen,{k:v['mean_error_px'] for k,v in reports.items()},flush=True)
if __name__=='__main__':run()

def learned_callback():
 model=Predictor();model.load_state_dict(torch.load(ROOT/'motion_gru.pth',map_location='cpu',weights_only=True)['model']);model.eval()
 def predict(history,timestamp,width,height):
  diagonal=float(np.hypot(width,height));x=features(history.points,history.times,history.confidence,diagonal)
  with torch.no_grad():delta,_=model(torch.tensor(x)[None]);d=delta[0].numpy()
  steps=(timestamp-history.times[-1])/.04;dx=np.array([np.interp(steps,[0]+HORIZONS,[0]+d[:,k].tolist()) for k in range(2)]);return np.asarray(history.points[-1])+dx/1000*diagonal
 return predict
