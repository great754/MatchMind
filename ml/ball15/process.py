"""Versioned M15 offline local review, never changing camera/player or M14 snapshots."""
import json,time,math
from pathlib import Path
from collections import Counter
import cv2,numpy as np
from src.vision.ball.schema import BallFrame,canonical
from src.vision.ball15.tracker import GuidedTracker
from .common import ROOT,load,save,sha,verify
from .detector import make_model,infer
from .search import run_search

def pitch_panel(world,ball):
 panel=np.full((480,680,3),(43,90,43),np.uint8);origin=np.array([25,35]);scale=np.array([6,6]);point=lambda p:tuple(np.round(origin+np.array(p)*scale).astype(int));cv2.rectangle(panel,point([0,0]),point([105,68]),(210,220,210),1);cv2.line(panel,point([52.5,0]),point([52.5,68]),(210,220,210),1);cv2.circle(panel,point([52.5,34]),55,(210,220,210),1)
 for p in world['player_world']['players']:
  if p['visibility_state']!='VISIBLE' or p['current_position'] is None:continue
  color=(70,100,255) if p['team']['label']=='TEAM_A' else (255,170,80) if p['team']['label']=='TEAM_B' else (180,180,180);cv2.circle(panel,point(p['current_position']),4,color,-1)
 if ball.pitch_position is not None:cv2.circle(panel,point(ball.pitch_position),5,(255,255,255),-1)
 cv2.putText(panel,'M8 camera / M9 players / M15 ball',(15,20),0,.55,(255,255,255),1);cv2.putText(panel,'Pitch position unavailable' if ball.pitch_position is None else 'Ground-supported ball',(15,470),0,.55,(255,255,255),1);return panel

def run():
 verify();det=load('detector_lock.json')['selected'];lock=load('pipeline_lock.json');cfg=lock['selected']['config'];search=load('search_lock.json');model=make_model(det['checkpoint']);summary={}
 for name in ['test_match','team_goal','southampton']:
  start=time.perf_counter();base=Path(f'outputs/world_model/{name}_m9_ready');meta=json.loads((base/'manifest.json').read_text());world=[json.loads(l) for l in (base/'frames.jsonl').open()];people=json.loads(Path(f'outputs/world_model/m9_cache/{name}/features.json').read_text());out=Path('outputs/world_model/ball_m15')/name;out.mkdir(parents=True,exist_ok=True)
  if (out/'manifest.json').exists():raise ValueError('M15 sidecar already exists; retain immutable release')
  if sha(base/'frames.jsonl')!=meta['frames_sha256'] or sha(meta['source_video'])!=meta['video_sha256']:raise ValueError('Source changed')
  rows=[{'recording_id':name,'sample_id':f'{name}:{i}','frame':i,'timestamp':i/meta['fps'],'fps':meta['fps'],'width':meta['width'],'height':meta['height'],'people':people[i]} for i in range(meta['frames'])];cams=[w['camera'] for w in world];cap=cv2.VideoCapture(meta['source_video'])
  def image_loader(row):
   ok,image=cap.read()
   if not ok:raise ValueError('Video incomplete')
   return image
  cached=(ROOT/'cache'/f'real_{name}.json').exists()
  if search['local_enabled']:ps,states=run_search(rows,cams,model,det,cfg,f'real_{name}',verification=lock['verification_enabled'],image_loader=image_loader)
  else:
   path=ROOT/'cache'/f'real_{name}.json';tag={'video_sha256':meta['video_sha256'],'checkpoint_sha256':det['checkpoint_sha256'],'pipeline_sha256':sha(ROOT/'pipeline_lock.json')};ps=[];states=[]
   if path.exists():
    data=json.loads(path.read_text());assert data['tag']==tag;ps=data['predictions'];states=data['states']
   else:
    tracker=GuidedTracker(meta['width'],meta['height'],cfg,verification=lock['verification_enabled'])
    for r,cam in zip(rows,cams,strict=True):
     cs,timing=infer(model,image_loader(r),det['tile'],generic=det['kind']=='generic');s=tracker.step(r['frame'],r['timestamp'],cs,cam,r['people']);s=s.model_copy(update={'search_stage':'GLOBAL_VALIDATION_SELECTED'});ps.append({'sample_id':r['sample_id'],'candidates':cs,**timing});states.append(s.model_dump())
     if r['frame']%50==0:print('REAL',name,r['frame'],flush=True)
    path.write_text(json.dumps({'tag':tag,'predictions':ps,'states':states}))
  cap.release();inference_wall=time.perf_counter()-start;cap=cv2.VideoCapture(meta['source_video']);writer=cv2.VideoWriter(str(out/'ball_overlay_and_pitch.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),meta['fps'],(1320,480));counts=Counter();offsets=[];examples=[];old=[json.loads(l) for l in Path(f'outputs/world_model/ball_m14/{name}/ball_frames.jsonl').open()]
  with (out/'ball_frames.jsonl').open('wb') as output,(out/'video_ball_observations.jsonl').open('w') as canonical_file:
   for i,(d,w,p) in enumerate(zip(states,world,ps,strict=True)):
    s=BallFrame.model_validate(d);offsets.append(output.tell());output.write((s.model_dump_json()+'\n').encode());canonical_file.write(canonical(name,meta['fps'],s,det['checkpoint_sha256']).model_dump_json()+'\n');counts[s.tracking_status]+=1;counts['directly_observed']+=s.directly_observed;counts['predicted_only']+=s.predicted_image_position is not None;counts['metric_supported']+=s.pitch_position is not None;counts[s.coordinate_state]+=1;ok,image=cap.read()
    if not ok:raise ValueError('Rendering source incomplete')
    for c in s.candidates:
     if c.confidence<cfg['minimum_visual']:continue
     x,y,bw,bh=map(int,c.bbox);cv2.rectangle(image,(x,y),(x+bw,y+bh),(0,150,230),1)
    if s.prediction_region:
     cx,cy=map(round,s.prediction_region.center);cv2.circle(image,(cx,cy),min(1000,round(s.prediction_region.radius90_px)),(255,200,60),1)
    point=s.image_position or s.predicted_image_position
    if point:cv2.circle(image,tuple(map(round,point)),7,(80,255,120) if s.directly_observed else (255,200,60),2)
    cv2.putText(image,f'M15 {s.tracking_status} / {s.search_stage}',(8,24),0,.5,(255,255,255),1);left=cv2.resize(image,(640,480));panel=pitch_panel(w,s);combined=np.hstack([left,panel]);writer.write(combined)
    if i in [0,meta['frames']//4,meta['frames']//2,meta['frames']-1] or s.directly_observed and len(examples)<4:
     imagepath=out/f'frame_{i:04d}.jpg';cv2.imwrite(str(imagepath),combined);examples.append({'frame':i,'status':s.tracking_status,'path':str(imagepath)})
  cap.release();writer.release();elapsed=time.perf_counter()-start;stats={**dict(counts),'frames':meta['frames'],'observed_coverage':counts['directly_observed']/meta['frames'],'predicted_only_coverage':counts['predicted_only']/meta['frames'],'supported_metric_coverage':counts['metric_supported']/meta['frames'],'detector_seconds':sum(p['seconds'] for p in ps),'search_and_tracking_wall_seconds':inference_wall,'end_to_end_seconds':elapsed,'fps_end_to_end':meta['frames']/elapsed,'cached_inference':cached,'independent_ball_accuracy':None,'metric_error_m':None,'independent_ground_validated':False,'m14_observed_frames':sum(s['directly_observed'] for s in old),'m14_predicted_frames':sum(s['predicted_image_position'] is not None for s in old)}
  manifest={'version':'m15','recording_id':name,'frames':meta['frames'],'fps':meta['fps'],'width':meta['width'],'height':meta['height'],'video_sha256':meta['video_sha256'],'source_m9_frames_sha256':meta['frames_sha256'],'ball_frames_sha256':sha(out/'ball_frames.jsonl'),'canonical_sha256':sha(out/'video_ball_observations.jsonl'),'frame_offsets':offsets,'detector_sha256':det['checkpoint_sha256'],'detector_kind':det['kind'],'pipeline_sha256':sha(ROOT/'pipeline_lock.json'),'statistics':stats,'examples':examples,'provenance':'M15 experimental VIDEO_CV; independently validated ground model unavailable; predictions never observed or analytics input'};(out/'manifest.json').write_text(json.dumps(manifest,indent=2));summary[name]=stats;save('real_video_summary.json',summary);print('REAL_DONE',name,stats,flush=True)
 verify();return summary
if __name__=='__main__':run()
