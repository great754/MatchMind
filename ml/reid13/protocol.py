"""New held-out broadcast protocol; legacy M10/M11 TEST remains historical only."""
import numpy as np
from ml.reid.evaluate import metrics_for
from ml.reid.tracklet import pool

def retrieval(rows):
 q=[];g=[];tq=[];tg=[]
 for identity in sorted({r['player_identity'] for r in rows}):
  items=[r for r in rows if r['player_identity']==identity];early=sorted([r for r in items if 2<=r['timestamp']<=9],key=lambda r:r['timestamp']);late=sorted([r for r in items if 20<=r['timestamp']<=29.99],key=lambda r:r['timestamp'])
  if len(early)<4 or len(late)<4:continue
  q.extend([early[i] for i in np.linspace(0,len(early)-1,3,dtype=int)]);g.extend([late[i] for i in np.linspace(0,len(late)-1,3,dtype=int)]);tq.append([early[i] for i in np.linspace(0,len(early)-1,min(8,len(early)),dtype=int)]);tg.append([late[i] for i in np.linspace(0,len(late)-1,min(8,len(late)),dtype=int)])
 return q,g,tq,tg

def evaluate(rows,vectors,pooler=None,device='cpu'):
 import torch
 from sklearn.metrics import precision_score,recall_score
 by={r['sample_id']:v for r,v in zip(rows,vectors,strict=True)};q,g,tq,tg=retrieval(rows)
 if not q:raise ValueError('No eligible heldout retrieval queries')
 qv=np.array([by[r['sample_id']] for r in q]);gv=np.array([by[r['sample_id']] for r in g]);crop=metrics_for(q,g,qv,gv,True);tracks={}
 for method in ['mean','quality_weighted']+(['learned'] if pooler is not None else []):
  def aggregate(t):
   x=np.array([by[r['sample_id']] for r in t]);quality=[r['quality']['score'] for r in t]
   if method=='learned':
    with torch.inference_mode():return pooler(torch.tensor(x[None],device=device),torch.tensor([quality],device=device))[0][0].cpu().numpy()
   return pool(x,quality if method=='quality_weighted' else None)
  tracks[method]=metrics_for([t[0] for t in tq],[t[0] for t in tg],np.array([aggregate(t) for t in tq]),np.array([aggregate(t) for t in tg]),True)
 y=[];p=[];sims=qv@gv.T
 for i,a in enumerate(q):
  for j,b in enumerate(g):
   if a['match_id']==b['match_id'] and a['team_identity']==b['team_identity']:y.append(a['player_identity']==b['player_identity']);p.append(sims[i,j]>=.82)
 crop['precision_at_082']=float(precision_score(y,p,zero_division=0));crop['recall_at_082']=float(recall_score(y,p,zero_division=0))
 gaps=[];labels=[];raw=[]
 for i,a in enumerate(q):
  for j,b in enumerate(g):
   if a['match_id']==b['match_id'] and a['team_identity']==b['team_identity']:gaps.append(b['timestamp']-a['timestamp']);labels.append(a['player_identity']==b['player_identity']);raw.append(sims[i,j])
 labels=np.array(labels,bool);raw=np.array(raw);gaps=np.array(gaps);breakdown={}
 for name,mask in [('10_15_seconds',(gaps>=10)&(gaps<15)),('15_30_seconds',(gaps>=15)&(gaps<=30))]:
  yy=labels[mask];ss=raw[mask];breakdown[name]={'pairs':int(mask.sum()),'positives':int(yy.sum()),'false_match_rate_at_082':float(np.mean(ss[~yy]>=.82)) if (~yy).any() else None,'false_nonmatch_rate_at_082':float(np.mean(ss[yy]<.82)) if yy.any() else None}
 return {'same_team_crop':crop,'tracklets':tracks,'same_team_pair_gap_buckets':breakdown,'selection_score':.5*crop['map']+.5*tracks['quality_weighted']['map'],'query_identity_count':len(tq),'time_gaps_seconds':[min(x['timestamp'] for x in b)-max(x['timestamp'] for x in a) for a,b in zip(tq,tg)],'protocol':'identical 3 crop queries/galleries and4–8 frame early/late GT tracklets; within match and team; new broadcast games only'}
