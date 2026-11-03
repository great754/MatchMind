"""Offline additive ball sidecars over immutable M8/M9 frames, every source frame."""
import json,time,shutil
from pathlib import Path
from collections import Counter
import cv2,torch
from ml.reid.model import choose_device,seed_all
from src.vision.ball.tracker import BallTracker,BallTrackerConfig
from src.vision.ball.schema import canonical,BallCandidate
from .common import ROOT,SEED,save,load,sha,verify
from .model import make_model,infer

SOURCES=['test_match','team_goal','southampton']
def run():
 selection=load('campaign_selection.json');chosen=selection['selected'];cfg=BallTrackerConfig.model_validate(load('tracker_config.json'))
 if sha(ROOT/'tracker_config.json')!=selection['tracker_config_sha256']:raise ValueError('Sealed tracker configuration changed')
 seed_all(SEED);device=choose_device();model=make_model(chosen['kind']=='finetuned',chosen['checkpoint'] if chosen['kind']=='finetuned' else None,device).eval();summary={}
 for name in SOURCES:
  tick=time.perf_counter();base=Path('outputs/world_model')/(name+'_m9_ready');meta=json.loads((base/'manifest.json').read_text())
  if sha(base/'frames.jsonl')!=meta['frames_sha256'] or sha(meta['source_video'])!=meta['video_sha256']:raise ValueError('Frozen source changed')
  rows=[json.loads(x) for x in (base/'frames.jsonl').open()];people=json.loads((Path('outputs/world_model/m9_cache')/name/'features.json').read_text());cache=ROOT/'video_detections'/name;cache.mkdir(parents=True,exist_ok=True);cachepath=cache/'candidates.jsonl';tag={'video_sha256':meta['video_sha256'],'detector_sha256':chosen['checkpoint_sha256'],'tiled':chosen['tiled'],'tile_size':chosen['tile_size'],'frames':meta['frames']}
  if (cache/'source.json').exists():
   if json.loads((cache/'source.json').read_text())!=tag:raise ValueError('Ball inference cache source mismatch')
  else:(cache/'source.json').write_text(json.dumps(tag))
  cached=[json.loads(x) for x in cachepath.open()] if cachepath.exists() else []
  if any(r['frame']!=i for i,r in enumerate(cached)) or len(cached)>meta['frames']:raise ValueError('Incomplete or invalid cache index')
  out=Path('outputs/world_model/ball_m14')/name;out.mkdir(parents=True,exist_ok=True)
  if (out/'manifest.json').exists():raise ValueError('Ball sidecar exists; use a new artifact path/version')
  cap=cv2.VideoCapture(meta['source_video']);writer=cv2.VideoWriter(str(out/'ball_overlay.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),meta['fps'],(meta['width'],meta['height']))
  if not cap.isOpened() or not writer.isOpened():raise ValueError('Could not open source/overlay')
  tracker=BallTracker(meta['width'],meta['height'],cfg);offsets=[];counts=Counter();detector_seconds=tracker_seconds=0;examples=[]
  with cachepath.open('a') as raw,(out/'ball_frames.jsonl').open('wb') as output,(out/'video_ball_observations.jsonl').open('w') as canonical_file:
   for i,row in enumerate(rows):
    ok,image=cap.read()
    if not ok:raise ValueError('Incomplete source video')
    if i<len(cached):record=cached[i]
    else:
     candidates,timing=infer(model,image,chosen['kind']=='finetuned',chosen['tiled'],device,chosen['tile_size']);record={'frame':i,'timestamp':i/meta['fps'],'candidates':candidates,**timing};raw.write(json.dumps(record)+'\n');raw.flush()
    detector_seconds+=record['seconds'];start=time.perf_counter();state=tracker.step(i,i/meta['fps'],record['candidates'],row['camera'],people[i]);tracker_seconds+=time.perf_counter()-start;offsets.append(output.tell());output.write((state.model_dump_json()+'\n').encode());canonical_file.write(canonical(name,meta['fps'],state,chosen['checkpoint_sha256']).model_dump_json()+'\n');counts['frames']+=1;counts[state.tracking_status]+=1;counts['directly_observed']+=state.directly_observed;counts['predicted_only']+=state.predicted_image_position is not None;counts['metric_supported']+=state.pitch_position is not None;counts['raw_candidate_frames']+=bool(state.candidates);counts[state.coordinate_state]+=1
    for c in state.candidates:
     if c.confidence<cfg.minimum_confidence:continue
     x,y,w,h=map(int,c.bbox);cv2.rectangle(image,(x,y),(x+w,y+h),(0,190,230),1)
    if state.image_position is not None:cv2.circle(image,tuple(map(round,state.image_position)),max(5,round(max(state.image_bbox[2:])/2)),(90,255,100),2)
    elif state.predicted_image_position is not None:cv2.drawMarker(image,tuple(map(round,state.predicted_image_position)),(255,210,60),cv2.MARKER_TILTED_CROSS,14,2)
    cv2.putText(image,f'M14 {state.tracking_status} | {state.coordinate_state}',(12,25),cv2.FONT_HERSHEY_SIMPLEX,.55,(240,240,240),2);writer.write(image)
    if (state.directly_observed and len(examples)<4) or i in [0,round(meta['frames']*.25),round(meta['frames']*.5),meta['frames']-1]:
     p=out/f'frame_{i:04d}.jpg';cv2.imwrite(str(p),image);examples.append({'frame':i,'state':state.tracking_status,'coordinate':state.coordinate_state,'image':str(p)})
    if i%50==0:print('PROCESS',name,i,'observed',counts['directly_observed'],'elapsed',round(time.perf_counter()-tick,1),flush=True)
  cap.release();writer.release();stats={**dict(counts),'observed_coverage':counts['directly_observed']/meta['frames'],'predicted_only_coverage':counts['predicted_only']/meta['frames'],'supported_metric_coverage':counts['metric_supported']/meta['frames'],'detector_seconds_cached_or_measured':detector_seconds,'tracking_seconds':tracker_seconds,'end_to_end_seconds':time.perf_counter()-tick,'independent_ball_accuracy':None,'coordinate_error_m':None,'ground_height_ground_truth_available':False,'model_experimental':True};manifest={'version':'m14','recording_id':name,'frames':meta['frames'],'fps':meta['fps'],'width':meta['width'],'height':meta['height'],'video_sha256':meta['video_sha256'],'source_m9_frames_sha256':meta['frames_sha256'],'ball_frames_sha256':sha(out/'ball_frames.jsonl'),'canonical_sha256':sha(out/'video_ball_observations.jsonl'),'raw_candidates_sha256':sha(cachepath),'frame_offsets':offsets,'detector_sha256':chosen['checkpoint_sha256'],'detector_kind':chosen['kind'],'tile_size':chosen['tile_size'],'tracker_config':cfg.model_dump(),'statistics':stats,'examples':examples,'provenance':'VIDEO_CV; neural detections plus causal temporal state over offline M8 geometry; ground support is a heuristic, not3D truth; never fed to production analytics'};(out/'manifest.json').write_text(json.dumps(manifest,indent=2));summary[name]=stats;save('real_video_summary.json',summary);print('COMPLETED',name,stats,flush=True)
 verify();return summary
if __name__=='__main__':run()
