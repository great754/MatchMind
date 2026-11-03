"""Learned candidate verification and confidence calibration on independent training/VAL clips."""
import json,math
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from ml.ball14.evaluate import iou
from src.vision.ball15.motion import extrapolate
from src.vision.ball15.association import evidence,score
from src.vision.ball15.tracker import GuidedTracker
from .common import ROOT,load,save,sha
from .camera import cameras

def examples(rows,predictions):
 x=[];y=[];previous=[];last=None
 for r,p in zip(rows,predictions,strict=True):
  if r['recording_id']!=last:previous=[];last=r['recording_id']
  pred=extrapolate([a[0] for a in previous],[a[1] for a in previous],r['timestamp']) if len(previous)>=3 else None;old=previous[-1][0] if previous else None
  for c in p['candidates']:
   x.append(evidence(c,pred,20,old,r['people'],r['width'],r['height'],.04,max([a['confidence'] for a in p['candidates'] if a is not c],default=0)));y.append(int(max([iou(c['bbox'],b['bbox']) for b in r['balls']],default=0)>=.25))
  if len(r['balls'])==1:
   b=r['balls'][0]['bbox'];previous=(previous+[([b[0]+b[2]/2,b[1]+b[3]/2],r['timestamp'])])[-8:]
  else:previous=[]
 return np.array(x),np.array(y)

def validation_run(rows,ps,cfg,verification=True):
 states=[];tracker=None;last=None;tp=fp=fn=predn=predbad=0
 cams=cameras(rows,'validation')
 for r,p,cam in zip(rows,ps,cams,strict=True):
  if r['recording_id']!=last:tracker=GuidedTracker(r['width'],r['height'],cfg,verification=verification);last=r['recording_id']
  s=tracker.step(r['frame'],r['timestamp'],p['candidates'],cam,r['people']);states.append(s)
  if s.directly_observed:
   good=max([iou(s.image_bbox,b['bbox']) for b in r['balls']],default=0)>=.25;tp+=good;fp+=not good;fn+=len(r['balls'])-int(good)
  else:fn+=len(r['balls'])
  if s.predicted_image_position is not None:
   predn+=1;predbad+=min([math.dist(s.predicted_image_position,(b['bbox'][0]+b['bbox'][2]/2,b['bbox'][1]+b['bbox'][3]/2)) for b in r['balls']],default=float('inf'))>12
 precision=tp/max(tp+fp,1);recall=tp/max(tp+fn,1);f1=2*precision*recall/max(precision+recall,1e-9)
 return {'tp':int(tp),'fp':int(fp),'fn':int(fn),'precision':precision,'recall':recall,'f1':f1,'predicted':predn,'prediction_inconsistent':predbad,'selection_score':f1-.1*predbad/len(rows)}

def run():
 if (ROOT/'pipeline_lock.json').exists():raise ValueError('Pipeline already sealed')
 rows=load('dataset.json')['samples'];train=[r for r in rows if r['split']=='train'];val=[r for r in rows if r['split']=='validation'];tx,ty=examples(train,load('cache/train_selected.json'));vx,vy=examples(val,load('cache/validation_selected.json'));scaler=StandardScaler().fit(tx);model=LogisticRegression(C=1,class_weight='balanced',max_iter=1000).fit(scaler.transform(tx),ty);serialized={'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),'coef':model.coef_[0].tolist(),'intercept':float(model.intercept_[0])};isotonic=IsotonicRegression(out_of_bounds='clip').fit(vx[:,0],vy);prob=isotonic.predict(vx[:,0]);calibration={'score_knots':isotonic.X_thresholds_.tolist(),'probability_knots':isotonic.y_thresholds_.tolist(),'validation_brier_raw':float(np.mean((vx[:,0]-vy)**2)),'validation_brier_calibrated':float(np.mean((prob-vy)**2)),'note':'in-sample VAL calibration diagnostic, prospective TEST Brier reported separately; low-threshold raw proposals, GT visibility unknown'};save('confidence_calibration.json',calibration);save('association_model.json',serialized)
 motion=load('motion_lock.json');mode=motion['selected'];radius=motion['validation'][mode]['p90_error_px'][0];det=load('detector_lock.json')['selected'];candidates=[];ps=load('cache/validation_selected.json')
 for kind in ['interpretable','logistic']:
  for threshold in [.35,.5,.65,.8]:
   for margin in [.08,.16]:
    cfg={'minimum_visual':det['threshold'],'prediction_seconds':.12,'margin':margin,'score_threshold':threshold,'association_kind':kind,'association_model':serialized,'motion_mode':mode if mode!='gru' else 'velocity','motion_kind':mode,'radius90_px':radius,'ground_validation_passed':False};metrics=validation_run(val,ps,cfg);candidates.append({'config':cfg,'metrics':metrics})
 chosen=max(candidates,key=lambda c:(c['metrics']['selection_score'],c['metrics']['precision']));baseline=validation_run(val,ps,chosen['config'],verification=False);enabled=chosen['metrics']['selection_score']>baseline['selection_score'];best_candidate=chosen;chosen={'config':chosen['config'],'metrics':chosen['metrics'] if enabled else baseline};chosen['config']['confidence_calibration']=calibration;save('pipeline_lock.json',{'selected':chosen,'verification_enabled':enabled,'best_candidate_verifier':best_candidate,'unverified_motion_baseline':baseline,'validation_candidates':candidates,'detector_sha256':det['checkpoint_sha256'],'motion_sha256':sha(ROOT/'motion_gru.pth'),'dataset_sha256':sha(ROOT/'dataset.json'),'demo_sha256':sha(ROOT/'demo_annotations.json'),'ground_validation':'No verified ground/airborne training+VAL labels or independent metric reference; mapping blocked pending validation','final_test_seen':False});print('PIPELINE_LOCK',chosen['config']['association_kind'],chosen['metrics'],flush=True)
if __name__=='__main__':run()
