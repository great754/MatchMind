"""Match-conditioned retrieval, same-team and long-gap tracklet evaluation."""
import json,time
from pathlib import Path
import numpy as np,torch
from sklearn.metrics import roc_auc_score
from .protocol import ROOT,retrieval_sets
from .tracklet import pool
from .model import SoccerReID,image_tensor,choose_device,seed_all
from src.vision.world.player_appearance import digest

def load_model(checkpoint=None,strategy='m9_body',device=None):
 if checkpoint:
  payload=torch.load(checkpoint,map_location='cpu',weights_only=True);strategy=payload['strategy'];model=SoccerReID(payload['classes'],strategy);model.load_state_dict(payload['model'],strict=True)
 else:model=SoccerReID(strategy=strategy)
 model.eval();return model.to(device or choose_device()),strategy

def embed(rows,model,strategy,device,batch=48):
 result=[];start=time.perf_counter()
 with torch.inference_mode():
  for i in range(0,len(rows),batch):
   x=torch.from_numpy(np.stack([image_tensor(r,strategy) for r in rows[i:i+batch]])).to(device);v=model(x)[1].cpu().numpy();result.append(v)
 return np.vstack(result),time.perf_counter()-start

def embedding_cache(rows,manifest_hash,checkpoint=None,strategy='m9_body',split='validation'):
 if split=='test' and not (ROOT/'selection_lock.json').exists():raise ValueError('TEST is sealed until validation model selection is locked')
 seed_all(20261102);modelhash=digest(checkpoint) if checkpoint else digest('outputs/models/m9/osnet_x0_25_msmt17.pth');key=f'{split}-{strategy}-{modelhash[:16]}-{manifest_hash[:12]}';directory=ROOT/'embedding_cache';directory.mkdir(exist_ok=True);p=directory/(key+'.npz');metadata=directory/(key+'.json');ids=[r['sample_id'] for r in rows]
 if p.exists():
  m=json.loads(metadata.read_text())
  if m['samples']!=ids or digest(p)!=m['sha256']:raise ValueError('Embedding cache identity/hash mismatch')
  return np.load(p)['vectors'],{**m,'cache_hit':True}
 device=choose_device();model,strategy=load_model(checkpoint,strategy,device);v,elapsed=embed(rows,model,strategy,device);np.savez_compressed(p,vectors=v);meta={'samples':ids,'checkpoint_sha256':modelhash,'manifest_sha256':manifest_hash,'strategy':strategy,'device':device,'wall_seconds':elapsed,'sha256':digest(p),'cache_hit':False};metadata.write_text(json.dumps(meta));del model
 if device=='mps':torch.mps.empty_cache()
 return v,meta

def metrics_for(query,gallery,qv,gv,same_team=False,threshold=.82,subgroups=True):
 sims=qv@gv.T;rank1=[];rank5=[];aps=[];pairs=[];truth=[];same=[];sizes=[];validq=[];failure=[]
 for i,q in enumerate(query):
  eligible=np.array([g['match_id']==q['match_id'] and (not same_team or g['team_identity']==q['team_identity']) for g in gallery]);ids=np.where(eligible)[0];relevance=np.array([gallery[j]['player_identity']==q['player_identity'] for j in ids]);scores=sims[i,ids]
  if not relevance.any():continue
  order=np.argsort(-scores,kind='stable');rel=relevance[order];r1=float(rel[0]);r5=float(rel[:5].any());ap=float((np.cumsum(rel)/(np.arange(len(rel))+1)*rel).sum()/rel.sum());rank1.append(r1);rank5.append(r5);aps.append(ap);validq.append((q,r1,r5,ap));pairs.extend(scores.tolist());truth.extend(relevance.tolist());same.extend([gallery[j]['team_identity']==q['team_identity'] for j in ids]);
  wrong=[j for j in ids if gallery[j]['player_identity']!=q['player_identity'] and gallery[j]['team_identity']==q['team_identity']];positive=[j for j in ids if gallery[j]['player_identity']==q['player_identity']]
  if wrong:failure.append({'query':q['sample_id'],'wrong':gallery[max(wrong,key=lambda j:sims[i,j])]['sample_id'],'wrong_similarity':float(max(sims[i,j] for j in wrong)),'lowest_positive':gallery[min(positive,key=lambda j:sims[i,j])]['sample_id'],'positive_similarity':float(min(sims[i,j] for j in positive))})
 y=np.asarray(truth,bool);s=np.asarray(pairs);st=np.asarray(same,bool)
 def dist(a):return {'count':len(a),'median':float(np.median(a)) if len(a) else None,'p90':float(np.quantile(a,.9)) if len(a) else None}
 out={'queries':len(rank1),'rank1':float(np.mean(rank1)) if rank1 else None,'rank5':float(np.mean(rank5)) if rank5 else None,'map':float(np.mean(aps)) if aps else None,'roc_auc':float(roc_auc_score(y,s)) if len(np.unique(y))==2 else None,'false_match_rate_at_082':float(np.mean(s[~y]>=threshold)) if (~y).any() else None,'false_nonmatch_rate_at_082':float(np.mean(s[y]<threshold)) if y.any() else None,'similarities':{'positive':dist(s[y]),'same_team_negative':dist(s[~y&st]),'different_team_negative':dist(s[~y&~st])},'hard_cases':sorted(failure,key=lambda r:-r['wrong_similarity'])[:20]}
 if subgroups:
  out['size_buckets']={b:{'queries':len(a:=[r for r in validq if r[0]['quality']['size_bucket']==b]),'rank1':float(np.mean([x[1] for x in a])) if a else None,'map':float(np.mean([x[3] for x in a])) if a else None} for b in ['small','medium','large']}
  out['quality_buckets']={label:{'queries':len(a:=[r for r in validq if predicate(r[0]['quality'])]),'rank1':float(np.mean([x[1] for x in a])) if a else None,'map':float(np.mean([x[3] for x in a])) if a else None} for label,predicate in [('blur_proxy_low_sharpness',lambda q:q['sharpness']<50),('higher_sharpness',lambda q:q['sharpness']>=50),('overlap_proxy',lambda q:q['occlusion_proxy']>.1),('low_overlap',lambda q:q['occlusion_proxy']<=.1)]}
 return out

def evaluate_vectors(rows,vectors):
 by={r['sample_id']:v for r,v in zip(rows,vectors,strict=True)};crop_q,crop_g=retrieval_sets(rows);q=[x[0] for x in crop_q];g=[x[0] for x in crop_g];qv=np.array([by[r['sample_id']] for r in q]);gv=np.array([by[r['sample_id']] for r in g]);overall=metrics_for(q,g,qv,gv);same=metrics_for(q,g,qv,gv,True);tq,tg=retrieval_sets(rows,True);track={}
 for method in ['mean','quality_weighted']:
  def pooled(t):return pool([by[r['sample_id']] for r in t],[r['quality']['score'] for r in t] if method=='quality_weighted' else None)
  track[method]={'overall':metrics_for([x[0] for x in tq],[x[0] for x in tg],np.array([pooled(t) for t in tq]),np.array([pooled(t) for t in tg])),'same_team':metrics_for([x[0] for x in tq],[x[0] for x in tg],np.array([pooled(t) for t in tq]),np.array([pooled(t) for t in tg]),True),'query_lengths':[len(t) for t in tq],'gallery_lengths':[len(t) for t in tg],'time_separation_seconds':[min(g['timestamp'] for g in b)-max(q['timestamp'] for q in a) for a,b in zip(tq,tg)]}
 return {'crop':overall,'same_team_crop':same,'tracklet':track,'rank1':overall['rank1'],'map':overall['map'],'same_team_rank1':same['rank1'],'same_team_map':same['map'],'tracklet_map':track['quality_weighted']['same_team']['map'],'selection_score':.7*track['quality_weighted']['same_team']['map']+.3*same['map']}

def evaluate_split(split,checkpoint=None,strategy='m9_body',output=None):
 manifest=ROOT/'curated_manifest.json';rows=[r for r in json.loads(manifest.read_text())['samples'] if r['split']==split];v,meta=embedding_cache(rows,digest(manifest),checkpoint,strategy,split);report=evaluate_vectors(rows,v);report['embedding_metadata']=meta;report['split']=split;report['test_used_for_selection']=False
 if output:Path(output).write_text(json.dumps(report,indent=2))
 return report
