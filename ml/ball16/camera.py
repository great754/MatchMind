"""Unsupervised image-only M8 motion on fresh organizer frames; no metric truth invented."""
import json,time
import cv2
from src.vision.world.pitch import TemporalCamera,PitchPerception
from src.vision.world.field_motion import field_image,measure_field_motion
from .common import ROOT,save

def cameras(rows,name):
 p=ROOT/(name+'_cameras.json')
 if p.exists():
  data=json.loads(p.read_text());assert data['sample_ids']==[r['sample_id'] for r in rows];return data['cameras']
 old=None;last=None;out=[];perception=PitchPerception();start=time.perf_counter()
 for r in rows:
  if r['recording_id']!=last:old=None;tracker=TemporalCamera();last=r['recording_id']
  image=cv2.imread(r['path']);camera=tracker.update(image,r['frame'],r['timestamp'],perception.detect(image)).model_dump();field=field_image(image);motion=measure_field_motion(old,field) if old is not None else {'accepted':False};old=field;camera['motion_evidence']['field_motion']=motion;out.append(camera)
 save(name+'_cameras.json',{'sample_ids':[r['sample_id'] for r in rows],'cameras':out,'seconds':time.perf_counter()-start});return out
