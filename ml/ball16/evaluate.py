"""Sealed model selection/evaluation, identical candidate/box protocols across versions."""
import json,time
from pathlib import Path
import cv2,numpy as np,torch
from ml.ball14.evaluate import iou
from .metrics import detection_metrics
from ml.ball15.metrics import tracking
from src.vision.ball.tracker import BallTracker,BallTrackerConfig
from src.vision.ball16.tracker import MultiHypothesisTracker
from src.vision.ball.schema import BallFrame
from .common import ROOT,load,save,sha,verify
from .detector import make_model,infer,merge_additive,recovery_region

def predictions(rows,model,resolution,name,mode='m16'):
 p=ROOT/'cache'/(name+'.json');p.parent.mkdir(exist_ok=True)
 tag={'sample_ids':[r['sample_id'] for r in rows],'mode':mode,'resolution':resolution,'dataset_sha256':sha(ROOT/'dataset.json'),'weights_sha256':load('training_result.json')['checkpoint_sha256'] if mode=='m16' else sha(Path('outputs/ml/ball_m14/generic_fasterrcnn_mobilenet320.pth')) if mode=='m14' else sha(Path('outputs/ml/ball_m15/yolo11n.pt'))}
 source=p.with_suffix('.source.json')
 if source.exists():
  if json.loads(source.read_text())!=tag:raise ValueError('Candidate cache source mismatch')
 else:source.write_text(json.dumps(tag))
 if p.exists():
  data=json.loads(p.read_text());assert [x['sample_id'] for x in data]==[r['sample_id'] for r in rows];return data
 journal=p.with_suffix('.jsonl');out=[json.loads(l) for l in journal.open()] if journal.exists() else []
 if [x['sample_id'] for x in out]!=[r['sample_id'] for r in rows[:len(out)]]:raise ValueError('Candidate cache clock mismatch')
 for i,r in enumerate(rows):
  if i<len(out):continue
  image=cv2.imread(r['path'])
  if mode=='m14':
   from ml.ball14.model import infer as oldinfer
   cs,t=oldinfer(model,image,False,True,'mps' if torch.backends.mps.is_available() else 'cpu',256)
  elif mode=='m15':
   from ml.ball15.detector import infer as oldinfer
   cs,t=oldinfer(model,image,320,generic=True)
  else:cs,t=infer(model,image,resolution)
  record={'sample_id':r['sample_id'],'candidates':cs,**t};out.append(record)
  with journal.open('a') as checkpoint:checkpoint.write(json.dumps(record)+'\n')
  if i%50==0:print('INFER',name,i,flush=True)
 p.write_text(json.dumps(out));return out

def replay(rows,ps,threshold,kind='multi'):
 tracker=None;last=None;states=[]
 for r,p in zip(rows,ps,strict=True):
  if r['recording_id']!=last:
   if kind=='m15':
    from src.vision.ball15.tracker import GuidedTracker
    lock=json.loads(Path('outputs/ml/ball_m15/pipeline_lock.json').read_text());tracker=GuidedTracker(r['width'],r['height'],lock['selected']['config'],verification=lock['verification_enabled'])
   else:tracker=MultiHypothesisTracker(r['width'],r['height'],threshold) if kind=='multi' else BallTracker(r['width'],r['height'],BallTrackerConfig(minimum_confidence=threshold,prediction_seconds=.12 if kind!='m14' else .35))
   last=r['recording_id']
  s=tracker.step(r['frame'],r['timestamp'],p['candidates'],r.get('camera',{'epoch':0,'calibration_valid':False}),r.get('people',[]));states.append(s.model_dump())
 return states

def select():
 if (ROOT/'selection.json').exists():raise ValueError('Selection sealed')
 rows=[r for r in load('dataset.json')['samples'] if r['split']=='validation'];from .camera import cameras
 for r,c in zip(rows,cameras(rows,'validation'),strict=True):r['camera']=c
 subset=rows[::4];m=make_model();candidates=[]
 for resolution in [640,960,1280]:
  ps=predictions(subset,m,resolution,f'validation_{resolution}')
  for threshold in [.1,.2,.35,.5,.65]:
   metric=detection_metrics(subset,[p['candidates'] for p in ps],threshold,.25);candidates.append({'resolution':resolution,'threshold':threshold,'metrics':metric,'seconds':sum(p['seconds'] for p in ps)})
 chosen=max(candidates,key=lambda c:(c['metrics']['f1'],c['metrics']['recall'],-c['seconds']));ps=predictions(rows,m,chosen['resolution'],'validation_selected');from .local_audit import audit
 local_diagnostic=audit(rows,ps,m,chosen['resolution'],chosen['threshold']);multi=replay(rows,ps,chosen['threshold']);kalman=replay(rows,ps,chosen['threshold'],'kalman');ms=tracking(rows,multi);ks=tracking(rows,kalman);mcorrect=ms['counts'].get('observed_center_matched',0);kcorrect=ks['counts'].get('observed_center_matched',0);usemulti=mcorrect>kcorrect and ms['accepted_center_consistency']>=ks['accepted_center_consistency'];save('selection.json',{'detector':chosen,'candidates':candidates,'multi_enabled':usemulti,'validation_multi':ms,'validation_kalman':ks,'temporal_enabled':False,'temporal_blocker':load('temporal_result.json')['blocker'],'local_enabled':False,'local_policy':'Additive recovery implemented; remains diagnostic unless prospective validation shows higher candidate recall and accepted correct coverage. No global proposal replacement.','dataset_sha256':sha(ROOT/'dataset.json'),'checkpoint_sha256':load('training_result.json')['checkpoint_sha256'],'test_seen':False});print('SELECTION',chosen,'multi',usemulti,flush=True)

def test():
 if (ROOT/'test_results.json').exists():raise ValueError('TEST already evaluated')
 verify();sel=load('selection.json');assert sha(ROOT/'dataset.json')==sel['dataset_sha256'];rows=[r for r in load('dataset.json')['samples'] if r['split']=='test'];from .camera import cameras
 for r,c in zip(rows,cameras(rows,'test'),strict=True):r['camera']=c
 m=make_model();ps=predictions(rows,m,sel['detector']['resolution'],'test_m16');del m
 from ml.ball14.model import make_model as oldmake
 old=oldmake(device='mps' if torch.backends.mps.is_available() else 'cpu').eval();oldps=predictions(rows,old,384,'test_m14','m14');del old
 from ml.ball15.detector import make_model as oldmake
 old=oldmake('outputs/ml/ball_m15/yolo11n.pt');fifteen=predictions(rows,old,640,'test_m15','m15');del old
 results={};states={}
 for name,p,threshold,kind in [('m14',oldps,.5,'m14'),('m15',fifteen,.65,'m15'),('new_detector_motion',ps,sel['detector']['threshold'],'kalman'),('new_multi',ps,sel['detector']['threshold'],'multi')]:
  ss=replay(rows,p,threshold,kind);states[name]=ss;results[name]={'detection':detection_metrics(rows,[x['candidates'] for x in p],threshold,.25),'tracking':tracking(rows,ss),'per_match':{}}
  for match in sorted({r['match_id'] for r in rows}):
   ix=[i for i,r in enumerate(rows) if r['match_id']==match];rs=[rows[i] for i in ix];results[name]['per_match'][match]={'detection':detection_metrics(rs,[p[i]['candidates'] for i in ix],threshold,.25),'tracking':tracking(rs,[ss[i] for i in ix])}
  (ROOT/f'test_{name}_states.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in ss))
 from .temporal import visual_replay
 temporal_states=visual_replay(rows,ps,sel['detector']['threshold']);results['new_temporal_visual']={'detection':results['new_detector_motion']['detection'],'tracking':tracking(rows,temporal_states),'status':'unselected diagnostic, fixed0.5probability; no validated negative sequence calibration'}
 # Detector-only performance uses the same retained candidates, no temporal state; learned verification is explicitly unvalidated.
 results['new_detector_only']={'detection':results['new_detector_motion']['detection'],'tracking':None};selected='new_multi' if sel['multi_enabled'] else 'new_detector_motion';results['complete_m16']=results[selected];save('test_results.json',{'results':results,'test_frames':len(rows),'selected':selected,'temporal_ablation':'Trained visual CNN, not activated or claimed validated: no independent match-disjoint reviewed negative sequences','local_ablation':'Additive candidate recovery diagnosed separately; not selected without validated gain','limits':'Organizer boxes do not certify visibility/absence; correct-visible coverage, true occlusion recall and metric accuracy are unavailable. Per-match results reported; two test matches do not justify broad generalization CI.','metric_supported':0,'test_samples':[r['sample_id'] for r in rows],'selected_on_test':False});print('TEST_DONE',{k:v['detection']['f1'] for k,v in results.items()},flush=True)
if __name__=='__main__':
 import sys
 select() if len(sys.argv)<2 or sys.argv[1]=='select' else test()
