"""Bounded, preregistered OSNet fine-tuning; validation-only checkpoint selection."""
import argparse,json,time,random,math,platform
from pathlib import Path
import cv2,numpy as np,torch
from torch.nn import functional as F
from .protocol import ROOT,curate,validate_splits
from .model import SoccerReID,image_tensor,choose_device,seed_all
from .losses import hard_triplet
from .sampling import pk_batches
from .evaluate import evaluate_split,evaluate_vectors,embed
from src.vision.world.player_appearance import digest

CONFIG={'version':'tactiqai-soccer-reid-v1','seed':20261102,'epochs':6,'batch_identities':8,'crops_per_identity':4,'learning_rate':.0003,'weight_decay':.0005,'optimizer':'AdamW','loss':'identity_cross_entropy + same-team-prioritized batch-hard cosine triplet','triplet_margin':.3,'patience':3,'trainable':'conv4, conv5, fc + new identity head; earlier layers and their BN statistics frozen','augmentation_runs':['none','moderate'],'selection':'0.7 validation same-team quality-weighted tracklet mAP +0.3 validation same-team crop mAP','integration_gate':{'same_team_rank1_gain':.03,'same_team_map_gain':.03,'tracklet_map_gain':.03,'overall_map_max_regression':.005},'test_policy':'No TEST embeddings/metrics before selection_lock; evaluate generic and selected model once after lock'}

def benchmark(strategy='m9_body'):
 seed_all(CONFIG['seed']);device=choose_device();model=SoccerReID(51,strategy).to(device);model.train_late();opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0003);regions=3 if strategy=='multi_region' else 1;x=torch.randn(32,regions,3,256,128,device=device);y=torch.arange(8,device=device).repeat_interleave(4);teams=y//4
 times=[]
 for i in range(4):
  t=time.perf_counter();opt.zero_grad();logits,z=model(x);loss=F.cross_entropy(logits,y)+hard_triplet(z,y,teams);loss.backward();opt.step()
  if device=='mps':torch.mps.synchronize()
  times.append(time.perf_counter()-t)
 result={'device':device,'batch':32,'regions':regions,'step_seconds':times,'steady_step_seconds':float(np.median(times[1:])),'torch':torch.__version__,'numpy':np.__version__,'opencv':cv2.__version__,'hardware':'Apple M4, 16GB unified memory; CUDA unavailable'};del model
 if device=='mps':torch.mps.empty_cache()
 (ROOT/'hardware_benchmark.json').write_text(json.dumps(result,indent=2));print(result,flush=True);return result

def validation_triplet(rows,vectors):
 ids={k:i for i,k in enumerate(sorted({r['player_identity'] for r in rows}))};teams={k:i for i,k in enumerate(sorted({r['team_identity'] for r in rows}))};values=[]
 for ix in pk_batches(rows,10,CONFIG['seed']):
  values.append(float(hard_triplet(torch.tensor(vectors[ix]),torch.tensor([ids[rows[i]['player_identity']] for i in ix]),torch.tensor([teams[rows[i]['team_identity']] for i in ix]),CONFIG['triplet_margin'])))
 return float(np.mean(values))

def cpu_tree(value):
 if isinstance(value,torch.Tensor):return value.detach().cpu()
 if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
 if isinstance(value,(list,tuple)):return type(value)(cpu_tree(v) for v in value)
 return value

def run():
 if (ROOT/'selection_lock.json').exists():raise ValueError('Selection already locked; use a new experiment directory/version, never overwrite')
 if not (ROOT/'curated_manifest.json').exists():curate()
 manifest=ROOT/'curated_manifest.json';all_rows=json.loads(manifest.read_text())['samples'];rows=[r for r in all_rows if r['split']=='train'];val=[r for r in all_rows if r['split']=='validation'];validate_splits(all_rows);cfg={**CONFIG,'manifest_sha256':digest(manifest),'sample_count':len(rows),'steps_per_epoch':math.ceil(len(rows)/32),'packages':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'opencv':cv2.__version__},'dataset_version':'soccer-reid-v1-curated','generic_checkpoint_sha256':digest('outputs/models/m9/osnet_x0_25_msmt17.pth')};config_path=Path('configs/ml/soccer-reid-v1-experiment.json')
 if not config_path.exists():config_path.write_text(json.dumps(cfg,indent=2))
 (ROOT/'training_config.json').write_text(json.dumps(cfg,indent=2));config_hash=digest(ROOT/'training_config.json');baseline={}
 for strategy in ['m9_body','full_masked','multi_region']:
  baseline[strategy]=evaluate_split('validation',strategy=strategy,output=ROOT/f'generic_validation_{strategy}.json');print('BASELINE_VALIDATION',strategy,{k:baseline[strategy][k] for k in ['same_team_rank1','same_team_map','tracklet_map','selection_score']},flush=True)
 strategy=max(baseline,key=lambda s:baseline[s]['selection_score']);(ROOT/'crop_selection.json').write_text(json.dumps({'selected':strategy,'basis':'generic validation score only; TEST not read','scores':{s:v['selection_score'] for s,v in baseline.items()}},indent=2));device=choose_device();ids={k:i for i,k in enumerate(sorted({r['player_identity'] for r in rows}))};teams={k:i for i,k in enumerate(sorted({r['team_identity'] for r in rows}))};history=[];candidates=[]
 for augmentation in CONFIG['augmentation_runs']:
  seed_all(CONFIG['seed']);model=SoccerReID(len(ids),strategy).to(device);model.train_late();opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);best=-1;stale=0;directory=ROOT/('run_'+augmentation);directory.mkdir(exist_ok=False);pair_counts={'same_team_negatives':0,'different_team_negatives':0,'same_match_same_team_negatives':0,'positives':0}
  for epoch in range(cfg['epochs']):
   model.train_late();losses=[];ces=[];triplets=[];start=time.perf_counter();rng=random.Random(CONFIG['seed']+epoch)
   for indices in pk_batches(rows,cfg['steps_per_epoch'],CONFIG['seed']+epoch):
    samples=[rows[i] for i in indices];x=torch.from_numpy(np.stack([image_tensor(r,strategy,augmentation=='moderate',rng) for r in samples])).to(device);y=torch.tensor([ids[r['player_identity']] for r in samples],device=device);team=torch.tensor([teams[r['team_identity']] for r in samples],device=device);opt.zero_grad();logits,z=model(x);ce=F.cross_entropy(logits,y);triplet=hard_triplet(z,y,team,cfg['triplet_margin']);loss=ce+triplet;loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();losses.append(float(loss.detach().cpu()));ces.append(float(ce.detach().cpu()));triplets.append(float(triplet.detach().cpu()))
    for i,a in enumerate(samples):
     for b in samples[i+1:]:
      if a['player_identity']==b['player_identity']:pair_counts['positives']+=1
      elif a['team_identity']==b['team_identity']:pair_counts['same_team_negatives']+=1;pair_counts['same_match_same_team_negatives']+=a['match_id']==b['match_id']
      else:pair_counts['different_team_negatives']+=1
   train_seconds=time.perf_counter()-start;model.eval();v,eval_seconds=embed(val,model,strategy,device);metrics=evaluate_vectors(val,v);epochfile=directory/f'epoch_{epoch+1:02d}.pth';torch.save({'model':{k:t.detach().cpu() for k,t in model.state_dict().items()},'strategy':strategy,'classes':len(ids),'version':CONFIG['version'],'dataset_sha256':digest(manifest),'config_sha256':config_hash,'epoch':epoch+1,'optimizer_state':cpu_tree(opt.state_dict()),'torch_rng_state':torch.get_rng_state()},epochfile);entry={'augmentation':augmentation,'epoch':epoch+1,'train_loss':float(np.mean(losses)),'classification_loss':float(np.mean(ces)),'triplet_loss':float(np.mean(triplets)),'validation_loss':validation_triplet(val,v),'validation_loss_reason':'Same-team batch-hard triplet on fixed validation PK samples; unseen identities make training classification loss inappropriate','validation_metrics':metrics,'train_seconds':train_seconds,'validation_seconds':eval_seconds,'checkpoint':str(epochfile),'checkpoint_sha256':digest(epochfile),'device':device};history.append(entry);(ROOT/'epoch_history.json').write_text(json.dumps(history,indent=2));print('EPOCH',augmentation,epoch+1,'loss',entry['train_loss'],'same_team_map',metrics['same_team_map'],'tracklet_map',metrics['tracklet_map'],'seconds',train_seconds,flush=True)
   if metrics['selection_score']>best+1e-8:best=metrics['selection_score'];stale=0
   else:stale+=1
   candidates.append(entry)
   if stale>=cfg['patience']:break
  (directory/'pair_sampling.json').write_text(json.dumps(pair_counts,indent=2));del model
  if device=='mps':torch.mps.empty_cache()
 chosen=max(candidates,key=lambda e:e['validation_metrics']['selection_score']);selection={'model_name':CONFIG['version'],'architecture':'OSNet x0.25,512D,late-stage fine-tuning','created_date':'2026-11-02','strategy':strategy,'selected_checkpoint':chosen['checkpoint'],'checkpoint_sha256':chosen['checkpoint_sha256'],'training_manifest_sha256':digest(manifest),'training_configuration_sha256':config_hash,'history_sha256':digest(ROOT/'epoch_history.json'),'augmentation':chosen['augmentation'],'epoch':chosen['epoch'],'validation_score':chosen['validation_metrics']['selection_score'],'test_seen':False};(ROOT/'selection_lock.json').write_text(json.dumps(selection,indent=2));print('SELECTION_LOCKED',selection,flush=True)
 return selection
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--benchmark',action='store_true');a=p.parse_args();benchmark() if a.benchmark else run()
