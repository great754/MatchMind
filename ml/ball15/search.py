"""Causal local/global search with bounded uncertainty and periodic global recovery."""
import json,math
import cv2
from src.vision.ball15.tracker import GuidedTracker
from src.vision.ball15.association import rank
from .detector import infer,search_region
from .common import ROOT,load,save,sha

def run_search(rows,cameras,model,detector,config,name,verification=True,image_loader=None):
 path=ROOT/'cache'/(name+'.json');tag={'dataset_samples':[r['sample_id'] for r in rows],'detector_sha256':detector['checkpoint_sha256'],'config':config,'verification':verification}
 if path.exists():
  data=json.loads(path.read_text());assert data['tag']==tag
  return data['predictions'],data['states']
 out=[];states=[];tracker=None;last=None
 for idx,(r,cam) in enumerate(zip(rows,cameras,strict=True)):
  if r['recording_id']!=last:tracker=GuidedTracker(r['width'],r['height'],config,verification=verification);last=r['recording_id']
  forecast=tracker.forecast(r['timestamp'],cam);point,radius=forecast;age=r['timestamp']-tracker.history.times[-1] if tracker.history.times else 0;region=search_region(point,radius,age,r['width'],r['height']);global_search=region is None or idx%8==0 or region[2]*region[3]>.6*r['width']*r['height'];image=image_loader(r) if image_loader else cv2.imread(r['path']);seconds=0;views=0
  if not global_search:
   cs,timing=infer(model,image,detector['tile'],region,detector['kind']=='generic');seconds+=timing['seconds'];views+=timing['views'];scores=rank(cs,point,radius,tracker.history.points[-1] if tracker.history.points else None,r.get('people',[]),r['width'],r['height'],age,config.get('association_model') if config['association_kind']=='logistic' else None);eligible=[s for s in scores if s['score']>=config['score_threshold'] and next(c['confidence'] for c in cs if c['candidate_id']==s['candidate_id'])>=config['minimum_visual']];global_search=not eligible or len(eligible)>1 and eligible[0]['score']-eligible[1]['score']<config['margin'];stage='LOCAL_GUIDED'
  if global_search:
   cs,timing=infer(model,image,detector['tile'],generic=detector['kind']=='generic');seconds+=timing['seconds'];views+=timing['views'];stage='GLOBAL_RECOVERY' if region is not None else 'GLOBAL_DISCOVERY'
  state=tracker.step(r['frame'],r['timestamp'],cs,cam,r.get('people',[]),forecast=forecast);state=state.model_copy(update={'search_stage':stage});out.append({'sample_id':r['sample_id'],'candidates':cs,'seconds':seconds,'views':views,'search_stage':stage});states.append(state.model_dump())
  if idx%40==0:print('SEARCH',name,idx,stage,flush=True)
 path.write_text(json.dumps({'tag':tag,'predictions':out,'states':states}));return out,states

def validate():
 from .camera import cameras
 from .detector import make_model
 from .metrics import tracking
 from .calibrate import validation_run
 lock=load('pipeline_lock.json');cfg=lock['selected']['config'];det=load('detector_lock.json')['selected'];rows=[r for r in load('dataset.json')['samples'] if r['split']=='validation'];cams=cameras(rows,'validation');model=make_model(det['checkpoint']);ps,states=run_search(rows,cams,model,det,cfg,'validation_guided',verification=lock['verification_enabled']);local=tracking(rows,states);global_metrics=lock['selected']['metrics'];tp=local['counts'].get('observed_iou_matched',0);fp=local['counts'].get('observed_iou_unmatched',0);fn=sum(len(r['balls']) for r in rows)-tp;f1=2*tp/max(1,2*tp+fp+fn);score=f1-.1*local['counts'].get('prediction_gt_inconsistent',0)/len(rows);selected=score>global_metrics['selection_score'];save('search_lock.json',{'local_enabled':selected,'global_validation':global_metrics,'local_validation':local,'local_selection_score':score,'local_observed_f1':f1,'criterion':'strictly improve observed IoU F1 minus .1*inconsistent-predictions/frame over global verifier; no TEST tuning','pipeline_sha256':sha(ROOT/'pipeline_lock.json'),'dataset_sha256':sha(ROOT/'dataset.json'),'test_seen':False});print('SEARCH_LOCK',selected,score,global_metrics['selection_score'],flush=True)
if __name__=='__main__':validate()
