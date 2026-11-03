"""Locked held-out detector comparisons and identical-sample tracking ablations."""
import json,time
from pathlib import Path
from collections import Counter
import cv2,numpy as np
from ml.reid.model import choose_device,seed_all
from src.vision.ball.tracker import BallTracker,BallTrackerConfig
from src.vision.world.field_motion import field_image,measure_field_motion
from src.vision.world.pitch import TemporalCamera,PitchPerception
from .common import ROOT,SEED,load,save,sha
from .model import make_model,infer
from .evaluate import detection_metrics,iou

def gt_match(point,balls,tolerance=12):
 if point is None or not balls:return None,None
 ranked=[(float(np.linalg.norm(np.asarray(point)-np.array([b['bbox'][0]+b['bbox'][2]/2,b['bbox'][1]+b['bbox'][3]/2]))),b['track_id']) for b in balls];distance,identity=min(ranked)
 return (identity if distance<=tolerance else None),distance

def run():
 if (ROOT/'test_results.json').exists():raise ValueError('Held-out evaluation already completed; no retuning')
 campaign=load('campaign_selection.json');seed_all(SEED);device=choose_device();rows=[r for r in load('dataset.json')['samples'] if r['split']=='test'];generic=campaign['selected'];fine=campaign['best_finetuned'];models=[('generic_full',False,False,480,None),('generic_tile480',False,True,480,None),('generic_tile256',False,True,256,None),('best_finetuned',True,fine['tiled'],fine['tile_size'],fine['checkpoint'])];allpred={};reports={};wall=time.perf_counter()
 for name,isfine,tiled,tile_size,checkpoint in models:
  directory=ROOT/'test_cache';directory.mkdir(exist_ok=True);p=directory/(name+'.json')
  if p.exists():predictions=json.loads(p.read_text())
  else:
   model=make_model(isfine,checkpoint,device).eval();predictions=[]
   for i,row in enumerate(rows):
    candidates,timing=infer(model,cv2.imread(row['path']),isfine,tiled,device,tile_size);predictions.append({'sample_id':row['sample_id'],'candidates':candidates,**timing})
    if i%40==0:print('TEST_INFERENCE',name,i,flush=True)
   p.write_text(json.dumps(predictions));del model
  if [p['sample_id'] for p in predictions]!=[r['sample_id'] for r in rows]:raise ValueError('Test cache sample mismatch')
  threshold=fine['threshold'] if isfine else generic['threshold'];allpred[name]=predictions;reports[name]={'primary':detection_metrics(rows,[p['candidates'] for p in predictions],threshold,.25),'strict_iou05':detection_metrics(rows,[p['candidates'] for p in predictions],threshold,.5),'inference_seconds':sum(p['seconds'] for p in predictions),'source':'organizer ball GT boxes, visibility unannotated'};print('DETECTOR_RESULT',name,reports[name]['primary'],flush=True)
 selected_name='best_finetuned' if generic['kind']=='finetuned' else 'generic_tile'+str(generic['tile_size']) if generic['tiled'] else 'generic_full';predictions=allpred[selected_name];cameras=[];last_clip=None;oldfield=None;perception=PitchPerception();camera=None;motionseconds=0
 for row in rows:
  if row['recording_id']!=last_clip:oldfield=None;camera=TemporalCamera();last_clip=row['recording_id']
  start=time.perf_counter();image=cv2.imread(row['path']);c=camera.update(image,row['frame'],row['timestamp'],perception.detect(image)).model_dump();current=field_image(image);motion=measure_field_motion(oldfield,current) if oldfield is not None else {'accepted':False,'reason':'NO_PREVIOUS_FRAME'};oldfield=current;c['motion_evidence']['field_motion']=motion;cameras.append(c);motionseconds+=time.perf_counter()-start
 ablations={}
 for name,comp,physics in [('tracking_only',False,False),('tracking_camera',True,False),('tracking_camera_physics',True,True)]:
  tracker=None;last_clip=None;counts=Counter();errors=[];associations=correct=0;last_identity=None;last_observed=None;recoveries=0;pred_good=pred_total=0;states=[];overhead=0
  for row,p,camera in zip(rows,predictions,cameras,strict=True):
   if row['recording_id']!=last_clip:tracker=BallTracker(row['width'],row['height'],BallTrackerConfig(minimum_confidence=generic['threshold'],camera_compensation=comp,physical_constraints=physics));last_clip=row['recording_id'];last_identity=None;last_observed=None
   start=time.perf_counter();state=tracker.step(row['frame'],row['timestamp'],p['candidates'],camera);overhead+=time.perf_counter()-start;states.append({'sample_id':row['sample_id'],**state.model_dump()});counts[state.tracking_status]+=1;counts['observed']+=state.directly_observed;counts['predicted']+=state.predicted_image_position is not None;identity,error=gt_match(state.image_position,row['balls'])
   if state.directly_observed:
    counts['accepted_matched_gt']+=identity is not None;counts['accepted_unmatched_gt']+=identity is None
    if error is not None:errors.append(error)
    if last_identity is not None and identity is not None:associations+=1;correct+=last_identity==identity
    if last_observed is not None and row['frame']-last_observed>1 and identity is not None and identity==last_identity:recoveries+=1
    last_identity=identity;last_observed=row['frame']
   if state.predicted_image_position is not None and row['balls']:
    pred_total+=1;pred_good+=gt_match(state.predicted_image_position,row['balls'])[0] is not None
  (ROOT/f'test_{name}.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in states));ablations[name]={'counts':dict(counts),'frames':len(rows),'observed_coverage':counts['observed']/len(rows),'predicted_only_coverage':counts['predicted']/len(rows),'accepted_gt_consistency_precision':counts['accepted_matched_gt']/max(counts['observed'],1),'accepted_unmatched_per_frame':counts['accepted_unmatched_gt']/len(rows),'associations_with_both_gt_endpoints':associations,'same_gt_object_associations':correct,'same_gt_association_fraction':correct/associations if associations else None,'recovery_after_tracker_observation_gap':recoveries,'occlusion_recovery_ground_truth':None,'prediction_gt_consistent':pred_good,'prediction_gt_comparable':pred_total,'false_prediction_gt_inconsistency_rate':1-pred_good/pred_total if pred_total else None,'localization_median_px':float(np.median(errors)) if errors else None,'tracking_seconds':overhead,'metric_supported_frames':0,'metric_error_ground_truth':None,'limitation':'GT annotated coordinates are not independent visibility or in-play-ball identity truth; same-GT association fraction is conditioned on matched endpoints and not full tracking accuracy'}
 save('test_results.json',{'detectors':reports,'tracking_ablations':ablations,'detector_only':'all thresholded raw candidates; no forced highest candidate assignment','camera_motion_measurement_seconds':motionseconds,'camera_motion_accepted_frames':sum(c['motion_evidence']['field_motion'].get('accepted',False) for c in cameras),'world_mapping':'No independent metric camera/ball-height GT on these clips; image-only tracking, metric positions remain unavailable','wall_seconds':time.perf_counter()-wall,'model_selected_on_test':False,'evaluation_samples':[r['sample_id'] for r in rows]});print('ABLATIONS',ablations,flush=True)
if __name__=='__main__':run()
