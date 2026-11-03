"""Sealed final comparisons; identical query/gallery crops and temporal samples."""
import json,pickle,time
import numpy as np,torch
from ml.reid.model import choose_device,seed_all
from ml.reid.evaluate import evaluate_vectors
from ml.reid.tracklet import pool
from .common import ROOT,M10,load,save,sha,SEED
from .features import cache,fine_vectors
from .model import Visual,Temporal
from .train import temporal_vectors,track_metrics
from .scorer import enrich,cues,probability_metrics,pair_data
from .data import tracks

def load_models():
 s=load('selection_lock.json');device=choose_device();v=Visual().to(device);p=torch.load(s['visual_checkpoint'],map_location=device,weights_only=True)
 if sha(s['visual_checkpoint'])!=s['visual_sha256'] or sha(s['temporal_checkpoint'])!=s['temporal_sha256']:raise ValueError('Selected model hash mismatch')
 v.net.blocks[-1].load_state_dict(p['last_block']);v.net.norm.load_state_dict(p['norm']);v.eval();t=Temporal().to(device);t.load_state_dict(torch.load(s['temporal_checkpoint'],map_location=device,weights_only=True)['model']);t.eval();return v,t,device

def osnet_vectors(split,kind):
 ck='584568ee5b04237f' if kind=='m10' else 'cf55163d78fc44c6';strategy='full_masked' if kind=='m10' else 'm9_body';p=next((M10/'embedding_cache').glob(f'{split}-{strategy}-{ck}-*.npz'));meta=json.loads(p.with_suffix('.json').read_text())
 if sha(p)!=meta['sha256']:raise ValueError('M10 cache changed')
 return np.load(p)['vectors'],meta['samples']

def pooled(tracklets,rows,vectors):
 by={r['sample_id']:i for i,r in enumerate(rows)}
 return np.array([pool(vectors[[by[k] for k in t['samples']]],[next(r['quality']['score'] for r in rows if r['sample_id']==k) for k in t['samples']]) for t in tracklets])

def paired(tracklets):
 idx={t['player_identity']:i for i,t in enumerate(tracklets) if t['window']==1};pairs=[(i,idx[t['player_identity']]) for i,t in enumerate(tracklets) if t['window']==0 and t['player_identity'] in idx];return [x for x,y in pairs],[y for x,y in pairs]

def matrix_metrics(tracklets,scores,threshold):
 qi,gi=paired(tracklets);out={}
 for same in [False,True]:
  rank1=[];rank5=[];ap=[];labels=[];values=[]
  for i,a in enumerate(qi):
   candidates=[j for j,b in enumerate(gi) if tracklets[a]['match_id']==tracklets[b]['match_id'] and (not same or tracklets[a]['team_identity']==tracklets[b]['team_identity'])];y=np.array([tracklets[a]['player_identity']==tracklets[gi[j]]['player_identity'] for j in candidates]);p=scores[i,candidates];order=np.argsort(-p,kind='stable');rel=y[order];rank1.append(float(rel[0]));rank5.append(float(rel[:5].any()));ap.append(float((np.cumsum(rel)/(np.arange(len(rel))+1)*rel).sum()/rel.sum()));labels.extend(y);values.extend(p)
  out['same_team' if same else 'overall']={'queries':len(qi),'rank1':float(np.mean(rank1)),'rank5':float(np.mean(rank5)),'map':float(np.mean(ap)),**probability_metrics(labels,values,threshold)}
 return out

def run():
 if (ROOT/'heldout_results.json').exists():raise ValueError('Final TEST report already exists; no iterative evaluation')
 lock=load('selection_lock.json')
 if not lock['selection_complete'] or lock['scorer_not_yet_locked']:raise ValueError('Unsealed selection')
 seed_all(SEED);start=time.perf_counter();tokens,rows=cache('test');visual,temporal,device=load_models();fine=fine_vectors(visual,tokens,device);np.save(ROOT/'test_finetuned.npy',fine);pre=np.load(ROOT/'test_pretrained.npy');ts=load('dataset.json')['tracklets']['test'];enriched=enrich(ts,rows);tv,attention=temporal_vectors(temporal,ts,rows,fine,device);np.save(ROOT/'test_temporal.npy',tv);save('test_attention.json',{'tracklets':ts,'weights':attention});report={'test_used_for_selection':False,'test_previously_published_in_m10':True,'protocol_sha256':sha(ROOT/'evaluation_protocol.json'),'models':{}}
 for name,vectors in [('m9_generic_osnet',None),('m10_soccer_osnet',None),('m11_pretrained_vit',pre),('m11_finetuned_vit',fine)]:
  if vectors is None:
   vectors,ids=osnet_vectors('test','m10' if name.startswith('m10') else 'm9')
   if ids!=[r['sample_id'] for r in rows]:raise ValueError('Baseline sample mismatch')
  trackv=pooled(ts,rows,vectors);report['models'][name]={'m10_crop_and_10_30frame_tracklet_protocol':evaluate_vectors(rows,vectors),'matched_8_12frame_tracklets':track_metrics(ts,trackv)}
 report['models']['m11_temporal']={'matched_8_12frame_tracklets':track_metrics(ts,tv)}
 qi,gi=paired(ts);cosine=tv[qi]@tv[gi].T;geometry=cosine.copy();x=[]
 for i,a in enumerate(qi):
  for j,b in enumerate(gi):
   c=cues(enriched[a],enriched[b],float(cosine[i,j]));x.append(c)
   # Fixed proxy reranking, not a camera certificate or a trained metric model.
   if np.isfinite(c[3]):geometry[i,j]=.8*cosine[i,j]+.2*np.exp(-c[3]/max(12*c[2]+3,1))
 with (ROOT/'reassociation.pkl').open('rb') as f:scorer=pickle.load(f)
 if sha(ROOT/'reassociation.pkl')!=lock['scorer_sha256']:raise ValueError('Scorer hash mismatch')
 probability=scorer.predict_proba(np.asarray(x))[:,1].reshape(cosine.shape);report['models']['m11_temporal_geometry']={'matched_8_12frame_tracklets':matrix_metrics(ts,geometry,.82),'geometry_is_supplied_calibration_proxy':True};report['models']['m11_temporal_learned_scorer']={'matched_8_12frame_tracklets':matrix_metrics(ts,probability,lock['probability_threshold'])}
 # Separate short/long window scorer benchmark, fixed before TEST.
 short=tracks(rows,True);sv,_=temporal_vectors(temporal,short,rows,fine,device);short=enrich(short,rows);sx,sy,sp=pair_data(short,sv);p=scorer.predict_proba(sx)[:,1];same=np.array([short[a]['team_identity']==short[b]['team_identity'] for a,b in sp]);m=same&(sx[:,2]<=8);report['scorer_gt_window_pairs']={'all':probability_metrics(sy,p,lock['probability_threshold']),'same_team':probability_metrics(sy[same],p[same],lock['probability_threshold']),'same_team_short_gap':probability_metrics(sy[m],p[m],lock['probability_threshold']) if m.any() else None};report['wall_seconds']=time.perf_counter()-start
 # Primary preregistered beat criterion; no selection after observing it.
 m10=report['models']['m10_soccer_osnet']['matched_8_12frame_tracklets']['same_team'];m11=report['models']['m11_temporal']['matched_8_12frame_tracklets']['same_team'];report['representation_beats_m10']=m11['rank1']>=m10['rank1']+.03 and m11['map']>=m10['map']+.03
 save('heldout_results.json',report);print('FINAL_HELDOUT',json.dumps({k:v['matched_8_12frame_tracklets']['same_team'] for k,v in report['models'].items()},indent=2),flush=True);return report
if __name__=='__main__':run()
