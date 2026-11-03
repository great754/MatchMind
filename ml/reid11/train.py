"""Two-stage tracklet-supervised ViT and temporal training, selected on VAL only."""
import json,time,random,math
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from ml.reid.model import choose_device,seed_all
from ml.reid.losses import hard_triplet
from ml.reid.evaluate import evaluate_vectors,metrics_for
from ml.reid.protocol import retrieval_sets
from .common import ROOT,SEED,save,load,sha
from .data import diverse
from .model import Visual,Temporal
from .features import cache,fine_vectors
CONFIG={'version':'tactiqai-temporal-soccer-reid-m11-v1','seed':SEED,'backbone':'DINO ViT-S/16, ImageNet self-supervised,384D','image_size':224,'crop':'full body, top18% masked, aspect-preserving neutral padding','visual_finetune':'last transformer block and final norm; first11 blocks frozen','visual_epochs':6,'visual_steps':80,'visual_lr':.00003,'temporal_epochs':12,'temporal_steps':40,'temporal_lr':.0001,'identities_per_batch':4,'tracklets_per_identity':2,'frames_per_training_tracklet':8,'frames_per_evaluation_tracklet':12,'loss':'tracklet identity cross-entropy + same-team hard triplet margin0.3','optimizer':'AdamW weight_decay0.0005 gradient_clip5','augmentation':'temporal sub-selection plus embedding dropout0.05; no pixel augmentation after frozen-prefix cache','visual_selection':'0.7 VAL same-team weighted-tracklet mAP +0.3 VAL same-team crop mAP','temporal_selection':'VAL same-team 12-frame temporal-tracklet mAP','scorer':'L2 logistic regression, train only; VAL threshold max recall under empirical same-team FMR<=0.01; no TEST tuning','geometry':'existing provided nonlinear keypoint interpolation; unknown uncertainty and true visibility represented as missing','evaluation':'M10 crop protocol unchanged, plus identical M11 8–12-frame tracklets for all models','integration':'requires representation gains without unsafe shadow merges; otherwise leave production M9 active'}

def batch_tracks(tracklets,rng):
 teams=sorted({t['team_identity'] for t in tracklets});team=rng.choice(teams);ids=sorted({t['player_identity'] for t in tracklets if t['team_identity']==team});other=sorted({t['player_identity'] for t in tracklets if t['team_identity']!=team});chosen=rng.sample(ids,min(3,len(ids)))+rng.sample(other,1);out=[]
 for identity in chosen:
  options=[t for t in tracklets if t['player_identity']==identity];out.extend(rng.sample(options,2) if len(options)>=2 else options*2)
 return out

def sequence_arrays(tracklets,rows,indices,training=False,rng=None):
 by={r['sample_id']:r for r in rows};selected=[]
 for t in tracklets:
  items=[by[k] for k in t['samples']]
  if training:items=sorted(rng.sample(items,min(8,len(items))),key=lambda r:r['timestamp'])
  selected.append(items)
 n=max(map(len,selected));ix=np.zeros((len(selected),n),int);q=np.zeros((len(selected),n),np.float32);times=q.copy();mask=np.zeros_like(q,bool)
 for j,items in enumerate(selected):
  for k,r in enumerate(items):ix[j,k]=indices[r['sample_id']];q[j,k]=r['quality']['score'];times[j,k]=r['timestamp'];mask[j,k]=True
 return ix,q,times,mask

def temporal_vectors(model,tracklets,rows,vectors,device):
 indices={r['sample_id']:i for i,r in enumerate(rows)};out=[];weights=[]
 with torch.inference_mode():
  for start in range(0,len(tracklets),32):
   ts=tracklets[start:start+32];ix,q,t,mask=sequence_arrays(ts,rows,indices);_,z,w=model(torch.tensor(vectors[ix],device=device),torch.tensor(q,device=device),torch.tensor(t,device=device),torch.tensor(mask,device=device));out.extend(z.cpu().numpy());weights.extend([a[:len(b['samples'])].tolist() for a,b in zip(w.cpu().numpy(),ts)])
 return np.asarray(out),weights

def track_metrics(tracklets,vectors):
 index={t['player_identity']:i for i,t in enumerate(tracklets) if t['window']==1};pairs=[(i,index[t['player_identity']]) for i,t in enumerate(tracklets) if t['window']==0 and t['player_identity'] in index];qi=[a for a,b in pairs];gi=[b for a,b in pairs]
 def meta(i):return {**tracklets[i],'sample_id':tracklets[i]['tracklet_id'],'quality':{'size_bucket':'unknown','sharpness':0,'occlusion_proxy':0}}
 return {key:metrics_for([meta(i) for i in qi],[meta(i) for i in gi],vectors[qi],vectors[gi],same,subgroups=False) for key,same in [('overall',False),('same_team',True)]}

def run():
 if (ROOT/'selection_lock.json').exists() or (ROOT/'training_history.json').exists():raise ValueError('Run already started; use a new version/root')
 save('config.json',CONFIG);Path('configs/ml/temporal-soccer-reid-v1.json').write_text(json.dumps(CONFIG,indent=2)+'\n');seed_all(SEED);device=choose_device();data=load('dataset.json');trtokens,tr=cache('train');vatokens,va=cache('validation');tracklets=data['tracklets']['train'];vatracks=data['tracklets']['validation'];ids={s:i for i,s in enumerate(sorted({r['player_identity'] for r in tr}))};teams={s:i for i,s in enumerate(sorted({r['team_identity'] for r in tr}))};indices={r['sample_id']:i for i,r in enumerate(tr)};model=Visual().to(device);model.late();opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=CONFIG['visual_lr'],weight_decay=.0005);history=[];best=-1;start=time.perf_counter();samplepairs={'same_team':0,'different_team':0}
 for epoch in range(CONFIG['visual_epochs']):
  model.eval();model.net.blocks[-1].train();model.classifier.train();rng=random.Random(SEED+epoch);losses=[];tick=time.perf_counter()
  for step in range(CONFIG['visual_steps']):
   ts=batch_tracks(tracklets,rng);ix,q,t,mask=sequence_arrays(ts,tr,indices,True,rng);tokens=torch.tensor(np.array(trtokens[ix.reshape(-1)],dtype=np.float32),device=device);z=model.finish(tokens).reshape(len(ts),-1,384);weights=torch.tensor(q,device=device);pooled=F.normalize((z*weights.unsqueeze(-1)).sum(1)/weights.sum(1,keepdim=True),dim=-1);y=torch.tensor([ids[t['player_identity']] for t in ts],device=device);team=torch.tensor([teams[t['team_identity']] for t in ts],device=device);loss=F.cross_entropy(model.classifier(pooled),y)+hard_triplet(pooled,y,team,.3);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();losses.append(float(loss.detach().cpu()))
   for i,a in enumerate(ts):
    for b in ts[i+1:]:
     if a['player_identity']!=b['player_identity']:samplepairs['same_team' if a['team_identity']==b['team_identity'] else 'different_team']+=1
  elapsed=time.perf_counter()-tick;model.eval();v=fine_vectors(model,vatokens,device);metric=evaluate_vectors(va,v);p=ROOT/f'visual_epoch_{epoch+1:02d}.pth';torch.save({'last_block':model.net.blocks[-1].state_dict(),'norm':model.net.norm.state_dict(),'classifier':model.classifier.state_dict(),'epoch':epoch+1,'initializer_sha256':load('initializer.json')['sha256']},p);entry={'stage':'visual','epoch':epoch+1,'loss':float(np.mean(losses)),'validation_score':metric['selection_score'],'validation_same_team_map':metric['same_team_map'],'validation_tracklet_map':metric['tracklet_map'],'training_seconds':elapsed,'checkpoint':str(p),'sha256':sha(p)};history.append(entry);save('training_history.json',history);print('VISUAL',entry,flush=True)
  if metric['selection_score']>best:best=metric['selection_score'];selected=p
 payload=torch.load(selected,map_location=device,weights_only=True);model.net.blocks[-1].load_state_dict(payload['last_block']);model.net.norm.load_state_dict(payload['norm']);model.eval();trv=fine_vectors(model,trtokens,device);vav=fine_vectors(model,vatokens,device);np.save(ROOT/'train_finetuned.npy',trv);np.save(ROOT/'validation_finetuned.npy',vav);visual_selected=selected
 temporal=Temporal().to(device);opt=torch.optim.AdamW(temporal.parameters(),lr=CONFIG['temporal_lr'],weight_decay=.0005);best=-1
 for epoch in range(CONFIG['temporal_epochs']):
  temporal.train();rng=random.Random(SEED+100+epoch);losses=[];tick=time.perf_counter()
  for step in range(CONFIG['temporal_steps']):
   ts=batch_tracks(tracklets,rng);ix,q,t,mask=sequence_arrays(ts,tr,indices,True,rng);x=F.dropout(torch.tensor(trv[ix],device=device),.05,training=True);logits,z,_=temporal(x,torch.tensor(q,device=device),torch.tensor(t,device=device),torch.tensor(mask,device=device));y=torch.tensor([ids[t['player_identity']] for t in ts],device=device);team=torch.tensor([teams[t['team_identity']] for t in ts],device=device);loss=F.cross_entropy(logits,y)+hard_triplet(z,y,team,.3);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(temporal.parameters(),5);opt.step();losses.append(float(loss.detach().cpu()))
  temporal.eval();v,_=temporal_vectors(temporal,vatracks,va,vav,device);metrics=track_metrics(vatracks,v);p=ROOT/f'temporal_epoch_{epoch+1:02d}.pth';torch.save({'model':temporal.state_dict(),'epoch':epoch+1,'visual_checkpoint_sha256':sha(visual_selected)},p);entry={'stage':'temporal','epoch':epoch+1,'loss':float(np.mean(losses)),'validation_score':metrics['same_team']['map'],'validation_rank1':metrics['same_team']['rank1'],'training_seconds':time.perf_counter()-tick,'checkpoint':str(p),'sha256':sha(p)};history.append(entry);save('training_history.json',history);print('TEMPORAL',entry,flush=True)
  if metrics['same_team']['map']>best:best=metrics['same_team']['map'];selected=p
 selection={'visual_checkpoint':str(visual_selected),'visual_sha256':sha(visual_selected),'temporal_checkpoint':str(selected),'temporal_sha256':sha(selected),'config_sha256':sha(ROOT/'config.json'),'dataset_sha256':sha(ROOT/'dataset.json'),'test_seen':False,'scorer_not_yet_locked':True,'device':device,'model_selection_seconds':time.perf_counter()-start,'same_team_negative_counts':samplepairs}
 save('visual_temporal_selection.json',selection);return selection
if __name__=='__main__':run()
