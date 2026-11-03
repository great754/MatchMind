"""Bounded data/objective/backbone ablations; fresh VAL selects, TEST remains sealed."""
import random,time,json,math
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from ml.reid.model import choose_device,seed_all,image_tensor
from ml.reid.sampling import pk_batches
from ml.reid.losses import hard_triplet
from ml.reid.train import cpu_tree
from .common import ROOT,SEED,save,load,sha
from .model import ReID,Pooler,supcon,augmented_tensor
from .protocol import evaluate,retrieval
CONFIG={'version':'soccer-reid-v2-m13','seed':SEED,'epochs':6,'steps_per_epoch':100,'batch_identities':8,'observations_per_identity':4,'optimizer':'AdamW','lr':.0003,'weight_decay':.0005,'gradient_clip':5,'loss':'CE + same-team-prioritized batch-hard triplet margin0.3; optional0.2 supervised contrastive temperature0.1','trainable':'conv4,conv5,fc + new identity head; earlier layers/BN frozen','augmentation':'moderate illumination/saturation, blur, down-up resolution, flip, partial occlusion, mild scale padding; head masked','selection':'fresh broadcast VAL0.5same-team crop mAP+0.5weighted tracklet mAP','patience':2,'variants':[{'name':'old_data_triplet','data':'old','width':'x0_25','supcon':False},{'name':'expanded_triplet','data':'expanded','width':'x0_25','supcon':False},{'name':'expanded_supcon','data':'expanded','width':'x0_25','supcon':True},{'name':'expanded_x1_supcon','data':'expanded','width':'x1_0','supcon':True}],'pool_epochs':8,'pool_steps':40,'pool_lr':.0001,'threshold':'visual0.82 fixed descriptive; separate validation-only precision/recall operating point; no TEST calibration','test_policy':'Only new SN_game9 TEST, once after all model/pooling selections. Historical M10/M11 TEST not used for selection.','primary_beat':'same-team crop mAP and selected-pool tracklet mAP each >=M10+0.03, crop Rank1 nonregression; no automatic production activation','supervision_limit':'original sequence GT; source match/team+numeric alias exclusion, not an independently verified worldwide player registry'}

def embed(rows,model,device):
 out=[]
 with torch.inference_mode():
  for i in range(0,len(rows),48):
   x=torch.tensor(np.stack([image_tensor(r,'full_masked') for r in rows[i:i+48]]),device=device);out.extend(model(x)[1].cpu().numpy())
 return np.array(out)

def load_model(path,device):
 payload=torch.load(path,map_location='cpu',weights_only=True);m=ReID(payload['classes'],payload['width']);m.load_state_dict(payload['model'],strict=True);m.eval();return m.to(device)

def train():
 if (ROOT/'training_history.json').exists() or (ROOT/'selection_lock.json').exists():raise ValueError('Run already started; new version/root required')
 data=load('dataset.json');validation=[r for r in data['samples'] if r['split']=='validation'];alltrain=[r for r in data['samples'] if r['split']=='train'];save('config.json',CONFIG);Path('configs/ml/soccer-reid-v2-m13.json').write_text(json.dumps(CONFIG,indent=2)+'\n');device=choose_device();history=[];variants=[];wall=time.perf_counter()
 for cfg in CONFIG['variants']:
  rows=[r for r in alltrain if cfg['data']=='expanded' or r['source']=='soccertrack_v2'];ids={x:i for i,x in enumerate(sorted({r['player_identity'] for r in rows}))};teams={x:i for i,x in enumerate(sorted({r['team_identity'] for r in rows}))};seed_all(SEED);m=ReID(len(ids),cfg['width']).to(device);m.train_late();opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=CONFIG['lr'],weight_decay=CONFIG['weight_decay']);directory=ROOT/cfg['name'];directory.mkdir(exist_ok=False);best=-1;stale=0;chosen=None
  for epoch in range(CONFIG['epochs']):
   m.train_late();rng=random.Random(SEED+epoch);losses=[];tick=time.perf_counter()
   for ix in pk_batches(rows,CONFIG['steps_per_epoch'],SEED+epoch):
    rs=[rows[i] for i in ix];x=torch.tensor(np.stack([augmented_tensor(r,rng) for r in rs]),device=device);labels=torch.tensor([ids[r['player_identity']] for r in rs],device=device);team=torch.tensor([teams[r['team_identity']] for r in rs],device=device);logits,z=m(x);loss=F.cross_entropy(logits,labels)+hard_triplet(z,labels,team,.3)
    if cfg['supcon']:loss=loss+.2*supcon(z,labels,team)
    opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step();losses.append(float(loss.detach().cpu()))
   train_seconds=time.perf_counter()-tick;m.eval();tick=time.perf_counter();v=embed(validation,m,device);metrics=evaluate(validation,v);validation_seconds=time.perf_counter()-tick;p=directory/f'epoch_{epoch+1:02d}.pth';torch.save({'model':cpu_tree(m.state_dict()),'classes':len(ids),'width':cfg['width'],'strategy':'full_masked','optimizer':cpu_tree(opt.state_dict()),'torch_rng_state':torch.get_rng_state(),'epoch':epoch+1,'config_sha256':sha(ROOT/'config.json'),'dataset_sha256':sha(ROOT/'dataset.json')},p);entry={'variant':cfg['name'],'epoch':epoch+1,'train_loss':float(np.mean(losses)),'validation_score':metrics['selection_score'],'validation_crop_map':metrics['same_team_crop']['map'],'validation_tracklet_map':metrics['tracklets']['quality_weighted']['map'],'training_seconds':train_seconds,'validation_seconds':validation_seconds,'checkpoint':str(p),'sha256':sha(p),'classes':len(ids),'samples':len(rows),'device':device};history.append(entry);save('training_history.json',history);print('EPOCH',entry,flush=True)
   if entry['validation_score']>best+1e-8:best=entry['validation_score'];chosen=entry;stale=0
   else:stale+=1
   if stale>=CONFIG['patience']:print('EARLY_STOP',cfg['name'],flush=True);break
  variants.append({**cfg,'selected':chosen});del m
  if device=='mps':torch.mps.empty_cache()
 save('variant_selections.json',variants);selected=max([v for v in variants if v['data']=='expanded'],key=lambda v:v['selected']['validation_score']);save('visual_selection.json',{'selected':selected,'control':next(v for v in variants if v['data']=='old'),'test_seen':False,'wall_seconds':time.perf_counter()-wall,'device':device});return selected

def pool_data(rows):
 out=[]
 for identity in sorted({r['player_identity'] for r in rows}):
  items=sorted([r for r in rows if r['player_identity']==identity],key=lambda r:r['timestamp']);chunks=np.array_split(np.array(items,dtype=object),4)
  for chunk in chunks:
   if len(chunk)>=4:out.append(list(chunk[np.linspace(0,len(chunk)-1,min(8,len(chunk)),dtype=int)]))
 return out

def fit_pool():
 if (ROOT/'selection_lock.json').exists():raise ValueError('Selection locked')
 seed_all(SEED);selection=load('visual_selection.json');chosen=selection['selected']['selected'];device=choose_device();m=load_model(chosen['checkpoint'],device);data=load('dataset.json')['samples'];rows=[r for r in data if r['split']=='train'];val=[r for r in data if r['split']=='validation'];v=embed(rows,m,device);vv=embed(val,m,device);np.save(ROOT/'train_selected_vectors.npy',v);np.save(ROOT/'validation_selected_vectors.npy',vv);indices={r['sample_id']:i for i,r in enumerate(rows)};tracks=pool_data(rows);byid={identity:[t for t in tracks if t[0]['player_identity']==identity] for identity in sorted({r['player_identity'] for r in rows})};ids={k:i for i,k in enumerate(byid)};teams={k:i for i,k in enumerate(sorted({r['team_identity'] for r in rows}))};pooler=Pooler().to(device);opt=torch.optim.AdamW(pooler.parameters(),lr=CONFIG['pool_lr'],weight_decay=.0005);history=[];best=-1;selected=None
 for epoch in range(CONFIG['pool_epochs']):
  rng=random.Random(SEED+epoch);losses=[];pooler.train()
  for _ in range(CONFIG['pool_steps']):
   team=rng.choice(sorted(teams));options=[k for k,t in byid.items() if t and t[0][0]['team_identity']==team];chosenids=rng.sample(options,min(8,len(options)));ts=[]
   for k in chosenids:ts+=rng.sample(byid[k],2) if len(byid[k])>=2 else byid[k]*2
   if len(chosenids)<2:continue
   n=max(map(len,ts));x=np.zeros((len(ts),n,512),np.float32);q=np.zeros((len(ts),n),np.float32);mask=np.zeros((len(ts),n),bool)
   for i,t in enumerate(ts):
    for j,r in enumerate(t):x[i,j]=v[indices[r['sample_id']]];q[i,j]=r['quality']['score'];mask[i,j]=True
   z,_=pooler(torch.tensor(x,device=device),torch.tensor(q,device=device),torch.tensor(mask,device=device));y=torch.tensor([ids[t[0]['player_identity']] for t in ts],device=device);tm=torch.tensor([teams[t[0]['team_identity']] for t in ts],device=device);loss=hard_triplet(z,y,tm,.3)+.1*supcon(z,y,tm);opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach().cpu()))
  pooler.eval();metrics=evaluate(val,vv,pooler,device);p=ROOT/f'pool_epoch_{epoch+1:02d}.pth';torch.save({'model':cpu_tree(pooler.state_dict()),'visual_sha256':chosen['sha256'],'epoch':epoch+1},p);entry={'epoch':epoch+1,'loss':float(np.mean(losses)),'validation_learned_map':metrics['tracklets']['learned']['map'],'checkpoint':str(p),'sha256':sha(p)};history.append(entry);save('pool_history.json',history)
  if entry['validation_learned_map']>best:best=entry['validation_learned_map'];selected=p
 pooler.load_state_dict(torch.load(selected,map_location=device,weights_only=True)['model']);pooler.eval();metrics=evaluate(val,vv,pooler,device);method=max(metrics['tracklets'],key=lambda k:metrics['tracklets'][k]['map']);lock={'visual_checkpoint':chosen['checkpoint'],'visual_sha256':chosen['sha256'],'visual_variant':selection['selected']['name'],'pool_checkpoint':str(selected),'pool_sha256':sha(selected),'selected_aggregation':method,'validation_aggregation_map':{k:v['map'] for k,v in metrics['tracklets'].items()},'variants':load('variant_selections.json'),'config_sha256':sha(ROOT/'config.json'),'dataset_sha256':sha(ROOT/'dataset.json'),'test_seen':False,'selection_complete':True,'current_code_sha256':{str(p):sha(p) for p in Path('ml/reid13').glob('*.py')}};save('selection_lock.json',lock);print('SELECTION_LOCKED',lock,flush=True)
if __name__=='__main__':
 import sys
 fit_pool() if '--pool' in sys.argv else train()
