"""Separate M16 sidecars over frozen M8/M9, source and mini-pitch synchronized."""
import json,time,resource
from pathlib import Path
from collections import Counter
import cv2
from src.vision.ball.schema import BallFrame,canonical
from src.vision.ball.tracker import BallTracker,BallTrackerConfig
from src.vision.ball16.tracker import MultiHypothesisTracker
from ml.ball15.process import pitch_panel as previous_pitch_panel
from .common import ROOT,load,save,sha,verify
from .detector import make_model,infer

def pitch_panel(world, ball):
 panel = previous_pitch_panel(world, ball)
 cv2.rectangle(panel, (0, 0), (679, 28), (43, 90, 43), -1)
 cv2.putText(panel, "M8 camera / M9 players / M16 ball", (15, 20), 0, .55, (255, 255, 255), 1)
 return panel

def run():
 verify();sel=load('selection.json');model=make_model();summary={}
 for name in ['test_match','team_goal','southampton']:
  start=time.perf_counter();base=Path(f'outputs/world_model/{name}_m9_ready');meta=json.loads((base/'manifest.json').read_text());world=[json.loads(l) for l in (base/'frames.jsonl').open()];people=json.loads(Path(f'outputs/world_model/m9_cache/{name}/features.json').read_text());out=Path(f'outputs/world_model/ball_m16/{name}');out.mkdir(parents=True,exist_ok=True)
  if (out/'manifest.json').exists():raise ValueError('M16 release exists; do not overwrite')
  if sha(meta['source_video'])!=meta['video_sha256'] or sha(base/'frames.jsonl')!=meta['frames_sha256']:raise ValueError('Frozen source mismatch')
  tracker=MultiHypothesisTracker(meta['width'],meta['height'],sel['detector']['threshold']) if sel['multi_enabled'] else BallTracker(meta['width'],meta['height'],BallTrackerConfig(minimum_confidence=sel['detector']['threshold'],prediction_seconds=.12));cap=cv2.VideoCapture(meta['source_video']);writer=cv2.VideoWriter(str(out/'source_and_pitch.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),meta['fps'],(1320,480));offsets=[];counts=Counter();detections=[];seconds=0;examples=[];states=[]
  if not cap.isOpened() or not writer.isOpened():raise ValueError('Video source/renderer unavailable')
  journal=ROOT/'cache'/f'{name}_real.jsonl';tag_path=journal.with_suffix('.source.json');tag={'video_sha256':meta['video_sha256'],'checkpoint_sha256':sel['checkpoint_sha256'],'resolution':sel['detector']['resolution']}
  if tag_path.exists() and json.loads(tag_path.read_text())!=tag:raise ValueError('Real inference cache source mismatch')
  tag_path.write_text(json.dumps(tag));cached=[json.loads(l) for l in journal.open()] if journal.exists() else []
  if any(r['frame']!=i for i,r in enumerate(cached)):raise ValueError('Real cache clock mismatch')
  with (out/'ball_frames.jsonl').open('wb') as output,(out/'video_ball_observations.jsonl').open('w') as canonical_file:
   for i,w in enumerate(world):
    ok,image=cap.read()
    if not ok:raise ValueError('Source incomplete')
    if i<len(cached):record=cached[i];cs=record['candidates'];timing={k:record[k] for k in ['seconds','views']}
    else:
     cs,timing=infer(model,image,sel['detector']['resolution']);record={'frame':i,'candidates':cs,**timing}
     with journal.open('a') as checkpoint:checkpoint.write(json.dumps(record)+'\n')
    seconds+=timing['seconds'];detections.append(record);s=tracker.step(i,i/meta['fps'],cs,w['camera'],people[i]);d=s.model_dump();d.update(version='m16',candidate_verification=[{'candidate_id':c['candidate_id'],'score':c['confidence'],'features':[]} for c in cs],pitch_position=None,pitch_position_confidence=None,coordinate_uncertainty_m=None,coordinate_state='AIRBORNE_SUSPECTED' if s.coordinate_state=='AIRBORNE_SUSPECTED' else 'HEIGHT_UNKNOWN' if s.plane_hypothesis else 'METRIC_UNAVAILABLE',coordinate_provenance='M16_INDEPENDENT_GROUND_OR_METRIC_REFERENCE_UNAVAILABLE',trail=[{**p,'pitch':None} for p in s.trail],search_stage='GLOBAL_VALIDATION_SELECTED',motion_model='MULTI_HYPOTHESIS' if sel['multi_enabled'] else 'KALMAN');s=BallFrame.model_validate(d);states.append(d);offsets.append(output.tell());output.write((s.model_dump_json()+'\n').encode());canonical_file.write(canonical(name,meta['fps'],s,sel['checkpoint_sha256']).model_dump_json()+'\n');counts[s.tracking_status]+=1;counts['directly_observed']+=s.directly_observed;counts['predicted_only']+=s.predicted_image_position is not None;counts['metric_supported']+=s.pitch_position is not None
    for c in s.candidates:
     if c.confidence<sel['detector']['threshold']:continue
     x,y,bw,bh=map(int,c.bbox);cv2.rectangle(image,(x,y),(x+bw,y+bh),(0,180,240),1)
    if s.predicted_image_position is not None and s.prediction_uncertainty_px is not None:cv2.circle(image,tuple(map(round,s.predicted_image_position)),min(1000,round(s.prediction_uncertainty_px)),(255,200,60),1)
    point=s.image_position or s.predicted_image_position
    if point:cv2.circle(image,tuple(map(round,point)),7,(80,255,120) if s.directly_observed else (255,200,60),2)
    cv2.putText(image,f'M16 {s.tracking_status}',(10,22),0,.6,(255,255,255),1);combined=cv2.hconcat([cv2.resize(image,(640,480)),pitch_panel(w,s)]);writer.write(combined)
    if i in [0,meta['frames']//4,meta['frames']//2,meta['frames']-1] or s.directly_observed and len(examples)<4:
     p=out/f'frame_{i:04d}.jpg';cv2.imwrite(str(p),combined);examples.append({'frame':i,'path':str(p),'state':s.tracking_status})
    if i%50==0:print('REAL',name,i,flush=True)
  cap.release();writer.release();probe=cv2.VideoCapture(str(out/'source_and_pitch.mp4'))
  if not probe.isOpened() or int(probe.get(cv2.CAP_PROP_FRAME_COUNT))!=meta['frames']:raise ValueError('Rendered video incomplete')
  probe.release();elapsed=time.perf_counter()-start;stats={**dict(counts),'frames':meta['frames'],'observed_coverage':counts['directly_observed']/meta['frames'],'predicted_only_coverage':counts['predicted_only']/meta['frames'],'supported_metric_coverage':0,'detector_seconds':seconds,'cached_candidate_frames':len(cached),'end_to_end_seconds':elapsed,'fps_end_to_end':meta['frames']/elapsed,'process_max_rss_bytes_macos':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'independent_visible_accuracy':None,'metric_error_m':None};manifest={'version':'m16','recording_id':name,'frames':meta['frames'],'fps':meta['fps'],'width':meta['width'],'height':meta['height'],'video_sha256':meta['video_sha256'],'source_m9_frames_sha256':meta['frames_sha256'],'ball_frames_sha256':sha(out/'ball_frames.jsonl'),'canonical_sha256':sha(out/'video_ball_observations.jsonl'),'detector_sha256':sel['checkpoint_sha256'],'frame_offsets':offsets,'statistics':stats,'examples':examples,'pipeline_components':{'learned_verifier_enabled':False,'local_search_enabled':False,'multi_hypothesis_enabled':sel['multi_enabled']}};(out/'manifest.json').write_text(json.dumps(manifest,indent=2));(ROOT/f'{name}_detections.json').write_text(json.dumps(detections));summary[name]=stats;save('real_video_summary.json',summary);print('REAL_DONE',name,stats,flush=True)
  old14=[json.loads(l) for l in Path(f'outputs/world_model/ball_m14/{name}/ball_frames.jsonl').open()];old15=[json.loads(l) for l in Path(f'outputs/world_model/ball_m15/{name}/ball_frames.jsonl').open()];(ROOT/f'{name}_frame_comparison.jsonl').write_text(''.join(json.dumps({'frame':i,'timestamp':c['timestamp'],**{v:{k:s[k] for k in ['tracking_status','image_position','predicted_image_position','pitch_position','directly_observed']} for v,s in [('m14',a),('m15',b),('m16',c)]}})+'\n' for i,(a,b,c) in enumerate(zip(old14,old15,states,strict=True))))
 verify()
if __name__=='__main__':run()
