"""Interpretable learned evidence, explicit missingness and non-overridable physical gates."""
import json,pickle,math
from pathlib import Path
import numpy as np,torch
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,brier_score_loss
from ml.reid.model import choose_device
from src.vision.coordinate_transform import PitchCoordinateTransformer
from .common import ROOT,load,save,sha,SEED
from .data import tracks
from .train import temporal_vectors
from .model import Temporal
NAMES=['embedding_similarity','team_compatibility','elapsed_seconds','pitch_displacement_m','required_speed_mps','camera_uncertainty_m','trajectory_error_m','minimum_quality','minimum_observation_count','visibility_transition','geometry_available','camera_uncertainty_available','visibility_available']

def cues(a,b,similarity):
 dt=b['start']-a['end'];pa=a.get('end_pitch');pb=b.get('start_pitch');distance=float(np.linalg.norm(np.asarray(pb)-pa)) if pa is not None and pb is not None else None;speed=distance/dt if distance is not None and dt>0 else None;u1=a.get('uncertainty');u2=b.get('uncertainty');u=math.hypot(u1,u2) if u1 is not None and u2 is not None else None;velocity=a.get('end_velocity');trajectory=float(np.linalg.norm(np.asarray(pb)-(np.asarray(pa)+np.asarray(velocity)*dt))) if velocity is not None and distance is not None else None
 compatible=None if 'UNKNOWN' in [a['team_identity'],b['team_identity']] else float(a['team_identity']==b['team_identity']);visibility=None if a.get('visibility') is None or b.get('visibility') is None else float(a['visibility']==b['visibility'])
 values=[similarity,compatible,dt,distance,speed,u,trajectory,min(a['quality'],b['quality']),min(len(a['samples']),len(b['samples'])),visibility,float(distance is not None),float(u is not None),float(visibility is not None)]
 return np.array([np.nan if x is None else float(x) for x in values],np.float64)

def physical_gate(a,b,max_speed=12,slack=3,max_gap=8):
 """Unknown positions never produce a fabricated reachability certificate."""
 dt=b['start']-a['end']
 if dt<=0:return False,'OVERLAPPING_TRACKLETS'
 if a.get('epoch') is not None and b.get('epoch') is not None and a['epoch']!=b['epoch']:return False,'CAMERA_EPOCH_DISCONTINUITY'
 if dt>max_gap:return False,'STALE_IDENTITY_SEARCH'
 if a.get('team_strength',0)>=.8 and b.get('team_strength',0)>=.8 and a['team_identity']!='UNKNOWN' and b['team_identity']!='UNKNOWN' and a['team_identity']!=b['team_identity']:return False,'STRONG_TEAM_CONFLICT'
 if a.get('end_pitch') is not None and b.get('start_pitch') is not None:
  u1=a.get('uncertainty');u2=b.get('uncertainty')
  if u1 is not None and u2 is not None:
   u=math.hypot(u1,u2);distance=np.linalg.norm(np.asarray(b['start_pitch'])-a['end_pitch'])
   if distance>max_speed*dt+slack+2*u:return False,'PHYSICALLY_IMPLAUSIBLE_REASSOCIATION'
 return True,'NO_KNOWN_PHYSICAL_CONTRADICTION' # not a sufficient acceptance certificate

def enrich(tracklets,rows):
 by={r['sample_id']:r for r in rows};provenance=load('geometry_provenance.json');transformers={}
 for m,p in provenance.items():
  if sha(p['file'])!=p['sha256']:raise ValueError('Calibration hash mismatch')
  transformers[m]=PitchCoordinateTransformer(p['file'])
 result=[]
 for t in tracklets:
  selected=[by[s] for s in t['samples']];coords=[]
  for r in selected:
   x,y,w,h=r['source_bbox'];v=transformers[r['match_id']].transform_point(x+w/2,y+h);coords.append(list(v) if v is not None and np.isfinite(v).all() else None)
  velocity=None
  if len(coords)>=2 and coords[-1] is not None and coords[-2] is not None:
   dt=selected[-1]['timestamp']-selected[-2]['timestamp'];velocity=((np.asarray(coords[-1])-coords[-2])/dt).tolist() if dt>0 else None
  result.append({**t,'start_pitch':coords[0],'end_pitch':coords[-1],'end_velocity':velocity,'uncertainty':None,'visibility':None,'geometry_provenance':'supplied nonlinear-keypoint proxy; unknown error, never assign zero uncertainty'})
 return result

def pair_data(tracklets,vectors,max_gap=130):
 x=[];y=[];pairs=[]
 for i,a in enumerate(tracklets):
  for j,b in enumerate(tracklets):
   if a['match_id']!=b['match_id'] or not 0<b['start']-a['end']<=max_gap:continue
   x.append(cues(a,b,float(vectors[i]@vectors[j])));y.append(a['player_identity']==b['player_identity']);pairs.append((i,j))
 return np.asarray(x),np.asarray(y,bool),pairs

def probability_metrics(y,p,threshold):
 y=np.asarray(y,bool);p=np.asarray(p,float)
 bins=[]
 for lo in np.arange(0,1,.1):
  m=(p>=lo)&(p<(lo+.1) if lo<.9 else p<=1)
  if m.any():bins.append({'count':int(m.sum()),'predicted':float(p[m].mean()),'observed':float(y[m].mean())})
 return {'pairs':len(y),'positives':int(y.sum()),'roc_auc':float(roc_auc_score(y,p)) if len(set(y))==2 else None,'brier':float(brier_score_loss(y,np.clip(p,0,1))),'threshold':float(threshold),'false_match_rate':float(np.mean(p[~y]>=threshold)) if (~y).any() else None,'false_nonmatch_rate':float(np.mean(p[y]<threshold)) if y.any() else None,'calibration_bins':bins}

def choose_threshold(y,p,max_fmr=.01):
 candidates=np.unique(np.r_[0,p,np.nextafter(1.,2.)]);valid=[t for t in candidates if np.mean(p[~y]>=t)<=max_fmr]
 return float(max(valid,key=lambda t:(np.mean(p[y]>=t),-np.mean(p[~y]>=t),t)))

def train():
 if (ROOT/'selection_lock.json').exists():raise ValueError('Selection already locked')
 selection=load('visual_temporal_selection.json');device=choose_device();model=Temporal().to(device);payload=torch.load(selection['temporal_checkpoint'],map_location=device,weights_only=True);model.load_state_dict(payload['model']);model.eval();data=load('dataset.json');sets={}
 for split in ['train','validation']:
  rows=[r for r in data['samples'] if r['split']==split];ts=tracks(rows,True);vectors,_=temporal_vectors(model,ts,rows,np.load(ROOT/f'{split}_finetuned.npy'),device);ts=enrich(ts,rows);x,y,pairs=pair_data(ts,vectors);sets[split]=(x,y,ts,pairs)
 x,y,ts,pairs=sets['train'];scorer=make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),StandardScaler(),LogisticRegression(C=1,max_iter=2000,random_state=SEED));scorer.fit(x,y);vx,vy,vts,vpairs=sets['validation'];vp=scorer.predict_proba(vx)[:,1];threshold=choose_threshold(vy,vp);p=ROOT/'reassociation.pkl'
 with p.open('wb') as f:pickle.dump(scorer,f)
 coefficients=scorer[-1].coef_[0];names=scorer[0].get_feature_names_out(NAMES).tolist();save('scorer_validation.json',{'training':{'pairs':len(y),'positive':int(y.sum()),'same_team_negatives':int(sum(not label and ts[i]['team_identity']==ts[j]['team_identity'] for label,(i,j) in zip(y,pairs)))},'validation':probability_metrics(vy,vp,threshold),'same_team_validation':probability_metrics(vy[m:=np.array([vts[i]['team_identity']==vts[j]['team_identity'] for i,j in vpairs])],vp[m],threshold),'feature_coefficients':dict(zip(names,coefficients.tolist())),'all_missing_training_features':[NAMES[i] for i in range(len(NAMES)) if np.isnan(x[:,i]).all()],'unknown_geometry_uncertainty_never_fabricated':True,'pair_protocol':'all chronological 40-second GT windows in same match, positive elapsed<=130s; positives and negatives share schedule; no identity ID/team ID/absolute position features','known_limitation':'panorama calibration proxies, not broadcast uncertain camera supervision; learned probability not established calibrated on broadcast shadow domain'})
 # Calibrate explicitly on SAME-TEAM validation pairs, the critical regime.
 threshold=choose_threshold(vy[m],vp[m]);report=load('scorer_validation.json');report['validation']=probability_metrics(vy,vp,threshold);report['same_team_validation']=probability_metrics(vy[m],vp[m],threshold);save('scorer_validation.json',report)
 lock={**selection,'scorer_sha256':sha(p),'probability_threshold':threshold,'scorer_not_yet_locked':False,'test_seen':False,'selection_complete':True,'evaluation_protocol_sha256':sha(ROOT/'evaluation_protocol.json'),'training_source_sha256':{str(p):sha(p) for p in Path('ml/reid11').glob('*.py')},'test_policy':'No retraining/reselection after this lock; all TEST comparisons final. M10 TEST historically published.'};save('selection_lock.json',lock);print('LOCKED',lock,flush=True)
 return lock
if __name__=='__main__':train()
