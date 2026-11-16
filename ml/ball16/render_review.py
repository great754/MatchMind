"""Regenerate only M16 presentation files from sealed observations, without inference."""
import json, time, os
from pathlib import Path
import cv2
from src.vision.ball.schema import BallFrame
from .common import save, verify, sha, load
from .process import pitch_panel

def run():
 verify(); report={}
 for name in ['test_match','team_goal','southampton']:
  started=time.perf_counter();out=Path('outputs/world_model/ball_m16')/name
  meta=json.loads((out/'manifest.json').read_text());base=Path(f'outputs/world_model/{name}_m9_ready');source=json.loads((base/'manifest.json').read_text())
  before={f:sha(out/f) for f in ['manifest.json','ball_frames.jsonl','video_ball_observations.jsonl']}
  cap=cv2.VideoCapture(source['source_video']);temporary=out/'source_and_pitch.rendering.mp4';writer=cv2.VideoWriter(str(temporary),cv2.VideoWriter_fourcc(*'mp4v'),meta['fps'],(1320,480))
  if not cap.isOpened() or not writer.isOpened():raise ValueError('Renderer unavailable')
  examples={e['frame']:e['path'] for e in meta['examples']};threshold=load('selection.json')['detector']['threshold']
  for i,(line,world) in enumerate(zip((out/'ball_frames.jsonl').open(),(base/'frames.jsonl').open(),strict=True)):
   s=BallFrame.model_validate_json(line);w=json.loads(world);ok,image=cap.read()
   if not ok:raise ValueError('Source incomplete')
   for c in s.candidates:
    if c.confidence<threshold:continue
    x,y,bw,bh=map(int,c.bbox);cv2.rectangle(image,(x,y),(x+bw,y+bh),(0,180,240),1)
   if s.predicted_image_position is not None and s.prediction_uncertainty_px is not None:cv2.circle(image,tuple(map(round,s.predicted_image_position)),min(1000,round(s.prediction_uncertainty_px)),(255,200,60),1)
   point=s.image_position or s.predicted_image_position
   if point:cv2.circle(image,tuple(map(round,point)),7,(80,255,120) if s.directly_observed else (255,200,60),2)
   cv2.putText(image,f'M16 {s.tracking_status}',(10,22),0,.6,(255,255,255),1);combined=cv2.hconcat([cv2.resize(image,(640,480)),pitch_panel(w,s)]);writer.write(combined)
   if i in examples:cv2.imwrite(examples[i],combined)
  cap.release();writer.release();probe=cv2.VideoCapture(str(temporary));frames=int(probe.get(cv2.CAP_PROP_FRAME_COUNT));probe.release()
  if frames!=meta['frames']:raise ValueError('Render clock mismatch')
  if any(sha(out/f)!=h for f,h in before.items()):raise ValueError('Presentation must not change sealed observations')
  os.replace(temporary,out/'source_and_pitch.mp4');report[name]={'frames':frames,'seconds':time.perf_counter()-started,'observation_hashes_unchanged':True,'reason':'Correct inherited M15 panel title to M16; no detector/tracker rerun or runtime replacement'}
 save('render_correction.json',report);verify();print(report)
if __name__=='__main__':run()
