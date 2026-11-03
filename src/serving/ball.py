"""Validated additive M14 ball sidecar. Old M8/M9 artifacts remain immutable."""
import json,math
from pathlib import Path
from src.vision.ball.schema import BallFrame,VideoBallObservation
from .cv import digest

class BallReview:
 def __init__(self,directory,base_manifest):
  self.directory=Path(directory).resolve();m=json.loads((self.directory/'manifest.json').read_text());self.manifest=m;self.path=self.directory/'ball_frames.jsonl';self.canonical=self.directory/'video_ball_observations.jsonl'
  for key in ['recording_id','frames','fps','width','height','video_sha256']:
   if m[key]!=base_manifest[key]:raise ValueError('Ball sidecar source/clock mismatch')
  if m['source_m9_frames_sha256']!=base_manifest['frames_sha256']:raise ValueError('Ball requires frozen matching M9 snapshot')
  if digest(self.path)!=m['ball_frames_sha256'] or digest(self.canonical)!=m['canonical_sha256']:raise ValueError('Ball artifact hash mismatch')
  self.offsets=m['frame_offsets']
  if len(self.offsets)!=m['frames']:raise ValueError('Invalid ball frame index')
  self.examples=[];seen=set()
  with self.path.open('rb') as f,self.canonical.open() as canonical:
   for i,offset in enumerate(self.offsets):
    if offset!=f.tell():raise ValueError('Ball index mismatch')
    ball=BallFrame.model_validate_json(f.readline());obs=VideoBallObservation.model_validate_json(canonical.readline())
    if ball.version!=m.get('version','m14') or obs.schema_version!=('ball-m15-v1' if ball.version=='m15' else 'ball-m14-v1'):raise ValueError('Ball model/schema version mismatch')
    signature=(ball.tracking_status,ball.directly_observed,ball.coordinate_state)
    if signature not in seen:self.examples.append({'frame':i,'timestamp':ball.timestamp,'status':ball.tracking_status,'directly_observed':ball.directly_observed,'coordinate_state':ball.coordinate_state});seen.add(signature)
    if ball.frame!=i or abs(ball.timestamp-i/m['fps'])>1e-6 or obs.frame!=i or obs.recording_id!=m['recording_id']:raise ValueError('Ball canonical/frame clock mismatch')
    if obs.fps!=m['fps'] or abs(obs.timestamp-ball.timestamp)>1e-6 or obs.detector_sha256!=m['detector_sha256']:raise ValueError('Ball canonical clock/model mismatch')
    if obs.image_bbox!=ball.image_bbox or obs.tracking_status!=ball.tracking_status or obs.coordinate_state!=ball.coordinate_state:raise ValueError('Ball canonical state mismatch')
    if obs.image!=ball.image_position or obs.pitch!=ball.pitch_position or obs.directly_observed!=ball.directly_observed or obs.predicted_image!=ball.predicted_image_position:raise ValueError('Ball canonical provenance mismatch')
    if any(c.bbox[0]+c.bbox[2]>m['width']+1e-5 or c.bbox[1]+c.bbox[3]>m['height']+1e-5 for c in ball.candidates):raise ValueError('Ball source bounds mismatch')
   if f.read(1) or canonical.read(1):raise ValueError('Unindexed ball data')
  self.stamps=(self.path.stat().st_mtime_ns,self.canonical.stat().st_mtime_ns)
 def frame(self,index):
  if (self.path.stat().st_mtime_ns,self.canonical.stat().st_mtime_ns)!=self.stamps:raise ValueError('Ball sidecar changed; restart validation')
  with self.path.open('rb') as f:f.seek(self.offsets[index]);return BallFrame.model_validate_json(f.readline())

def read_ball_registry(path):
 """An optional malformed ball registry never removes usable M9 recordings."""
 try:
  data=json.loads(Path(path).read_text())
  return {k:v for k,v in data.items() if isinstance(k,str) and isinstance(v,str)} if isinstance(data,dict) else {}
 except (ValueError,OSError):return {}
