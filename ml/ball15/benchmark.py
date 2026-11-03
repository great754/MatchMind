"""Single sealed prospective TEST run: seven ablations and independently annotated demo images."""
import json,time,math,copy
from pathlib import Path
import numpy as np,torch,cv2
from src.vision.ball.tracker import BallTracker,BallTrackerConfig
from src.vision.ball15.tracker import GuidedTracker
from ml.ball14.evaluate import detection_metrics,iou
from ml.ball14.model import make_model as oldmodel,infer as oldinfer
from .common import ROOT,load,save,sha,verify
from .select_detector import predictions
from .detector import make_model
from .camera import cameras
from .metrics import tracking
from .search import run_search
from .motion import Predictor,samples,errors,learned_callback,HORIZONS

def old_predictions(rows,name):
 path=ROOT/'cache'/f'{name}_m14.json'
 if path.exists():return json.loads(path.read_text())
 device='mps' if torch.backends.mps.is_available() else 'cpu';m=oldmodel(device=device).eval();out=[]
 for i,r in enumerate(rows):
  cs,t=oldinfer(m,cv2.imread(r['path']),False,True,device,256);out.append({'sample_id':r['sample_id'],'candidates':cs,**t})
  if i%40==0:print('OLD_INFERENCE',name,i,flush=True)
 path.write_text(json.dumps(out));return out

def ablations(rows,ps,oldps,cams,model,det,cfg):
 reports={};allstates={};learned=learned_callback()
 for name in ['m14_frozen','new_detector_only','new_kalman','learned_motion','candidate_verification']:
  states=[];last=None;tracker=None;start=time.perf_counter()
  for r,p,old,cam in zip(rows,ps,oldps,cams,strict=True):
   if r['recording_id']!=last:
    if name in ['m14_frozen','new_kalman']:tracker=BallTracker(r['width'],r['height'],BallTrackerConfig(minimum_confidence=.5 if name=='m14_frozen' else det['threshold']))
    else:tracker=GuidedTracker(r['width'],r['height'],cfg,learned_predictor=learned if name=='learned_motion' else None,verification=name=='candidate_verification',motion=name!='new_detector_only')
    last=r['recording_id']
   if name=='new_detector_only':
    # Reset state each frame; only exactly one above-threshold visual candidate is accepted.
    from src.vision.ball.schema import BallFrame
    cs=p['candidates'];viable=[c for c in cs if c['confidence']>=det['threshold']];c=viable[0] if len(viable)==1 else None;d={'version':'m15','frame':r['frame'],'timestamp':r['timestamp'],'camera_epoch':cam.get('epoch',0),'candidate_count':len(cs),'candidates':cs,'tracking_status':'DETECTED' if c else 'AMBIGUOUS' if viable else 'LOST'}
    if c:b=c['bbox'];d.update(directly_observed=True,image_bbox=b,image_position=[b[0]+b[2]/2,b[1]+b[3]/2],detection_confidence=c['confidence'])
    state=BallFrame.model_validate(d)
   else:state=tracker.step(r['frame'],r['timestamp'],old['candidates'] if name=='m14_frozen' else p['candidates'],cam,r.get('people',[]))
   states.append(state.model_dump())
  reports[name]=tracking(rows,states)|{'tracking_seconds':time.perf_counter()-start};allstates[name]=states
 localps,localstates=run_search(rows,cams,model,det,cfg,'test_guided',verification=False);reports['motion_guided_local_search']=tracking(rows,localstates);allstates['motion_guided_local_search']=localstates
 if load('search_lock.json')['local_enabled']:
  fullps,fullstates=run_search(rows,cams,model,det,cfg,'test_complete',verification=load('pipeline_lock.json')['verification_enabled'])
 else:
  fullps=ps
  if load('pipeline_lock.json')['verification_enabled']:fullstates=allstates['candidate_verification']
  else:
   fullstates=[];last=None
   for r,p,cam in zip(rows,ps,cams,strict=True):
    if r['recording_id']!=last:tracker=GuidedTracker(r['width'],r['height'],cfg,verification=False);last=r['recording_id']
    fullstates.append(tracker.step(r['frame'],r['timestamp'],p['candidates'],cam,r.get('people',[])).model_dump())
 reports['complete_selected']=tracking(rows,fullstates);allstates['complete_selected']=fullstates
 for name,states in allstates.items():(ROOT/f'test_{name}.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in states))
 return reports,fullps

def run():
 if (ROOT/'test_results.json').exists():raise ValueError('TEST already evaluated; no retuning')
 verify();lock=load('pipeline_lock.json');assert sha(ROOT/'dataset.json')==lock['dataset_sha256'] and sha(ROOT/'demo_annotations.json')==lock['demo_sha256'];load('search_lock.json');start=time.perf_counter();rows=[r for r in load('dataset.json')['samples'] if r['split']=='test'];det=load('detector_lock.json')['selected'];cfg=lock['selected']['config'];model=make_model(det['checkpoint']);ps=predictions(model,rows,'test_selected',det['tile'],det['kind']=='generic');oldps=old_predictions(rows,'test');cams=cameras(rows,'test');reports,fullps=ablations(rows,ps,oldps,cams,model,det,cfg)
 motion=Predictor();motion.load_state_dict(torch.load(ROOT/'motion_gru.pth',map_location='cpu',weights_only=True)['model']);motion.eval();ss=samples(rows);motionreports={};cal=load('motion_lock.json')['validation']
 for kind in ['velocity','acceleration','gru']:
  es=errors(motion,ss,kind);arr=np.array([e['errors'] for e in es]);sigma=np.array([e['sigma'] for e in es]) if kind=='gru' else np.ones_like(arr);radius=np.array(cal[kind]['radius90_scale'])*np.maximum(sigma,1);turn=[e['sudden_direction_change'] for e in es];motionreports[kind]={'samples':len(es),'horizons':HORIZONS,'mean_error_px':arr.mean(0).tolist(),'median_error_px':np.median(arr,0).tolist(),'p90_error_px':np.quantile(arr,.9,axis=0).tolist(),'inside_validation_radius90_fraction':(arr<=radius).mean(0).tolist(),'sudden_change_samples':sum(turn),'sudden_change_mean_error_px':arr[turn].mean(0).tolist() if any(turn) else None}
 demo=[r for r in load('demo_annotations.json')['rows'] if r['scorable']];demops=predictions(model,demo,'demo_selected',det['tile'],det['kind']=='generic');demoold=old_predictions(demo,'demo');demoreport={}
 for name in ['test_match','team_goal','southampton','all']:
  ix=[i for i,r in enumerate(demo) if name=='all' or r['recording_id']==name];rs=[demo[i] for i in ix];demoreport[name]={'m14':detection_metrics(rs,[demoold[i]['candidates'] for i in ix],.5,.25),'m15':detection_metrics(rs,[demops[i]['candidates'] for i in ix],det['threshold'],.25),'scope':'single-annotator source-only boxes; uncertain frames excluded; no independent height/metric GT'}
 calibration=load('confidence_calibration.json');xs=[];ys=[]
 for r,p in zip(rows,ps,strict=True):
  for c in p['candidates']:xs.append(c['confidence']);ys.append(max([iou(c['bbox'],b['bbox']) for b in r['balls']],default=0)>=.25)
 probs=np.interp(xs,calibration['score_knots'],calibration['probability_knots']);result={'detectors':{'m14':detection_metrics(rows,[p['candidates'] for p in oldps],.5,.25),'m15':detection_metrics(rows,[p['candidates'] for p in ps],det['threshold'],.25)},'tracking_ablations':reports,'motion':motionreports,'manual_demo':demoreport,'calibration_test':{'raw_brier':float(np.mean((np.array(xs)-ys)**2)),'calibrated_brier':float(np.mean((probs-ys)**2))},'seconds':time.perf_counter()-start,'test_samples':[r['sample_id'] for r in rows],'selected_on_test':False,'metric_gt_available':False,'metric_reconstruction_independently_validated':False};save('test_results.json',result);verify();print('TEST_DONE',result['detectors'],reports['complete_selected'],flush=True)
if __name__=='__main__':run()
