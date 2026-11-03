"""Audit M14 rejection mechanisms without relaxing geometry/height requirements."""
import json,math
from collections import Counter
from pathlib import Path
import numpy as np
from src.vision.ball.schema import BallCandidate
from src.vision.ball.geometry import plane_intersection,reconstruct,transform,warp_point
from .common import save

def ground_cues(candidate,camera,people,image=None):
 point=(candidate.bbox[0]+candidate.bbox[2]/2,candidate.bbox[1]+candidate.bbox[3]);plane,reason=plane_intersection(point,camera);answer={'m8_valid':bool(camera.get('calibration_valid')),'plane_supported':plane is not None,'reason':reason,'near_foot':False,'physical_radius_compatible':False,'independent_ground_model_validated':False}
 if plane is None:return answer
 for person in people:
  box=person.get('bbox') or person.get('image_bbox')
  if not box:continue
  x,y,w,h=box
  if abs(point[1]-(y+h))<=max(3,.3*candidate.bbox[3]) and abs(point[0]-(x+w/2))<=max(4,.12*w) and .02<=candidate.bbox[3]/max(h,1)<=.2:answer['near_foot']=True
 H=transform(camera)
 if H is not None:
  a=warp_point(H,point);b=warp_point(H,[point[0]+candidate.bbox[2],point[1]]);diameter=float(np.linalg.norm(b-a)) if a is not None and b is not None else None;answer['apparent_plane_diameter_m']=diameter;answer['physical_radius_compatible']=diameter is not None and .12<=diameter<=.35
 # Apparent size and proximity cannot independently resolve airborne height. Never grant metric authority here.
 return answer

def run():
 report={}
 for name in ['test_match','team_goal','southampton']:
  world=[json.loads(x) for x in Path(f'outputs/world_model/{name}_m9_ready/frames.jsonl').open()];balls=[json.loads(x) for x in Path(f'outputs/world_model/ball_m14/{name}/ball_frames.jsonl').open()];people=json.loads(Path(f'outputs/world_model/m9_cache/{name}/features.json').read_text());counts=Counter();streak=0;maxstreak=0
  for r,s,p in zip(world,balls,people,strict=True):
   if not s['directly_observed']:streak=0;continue
   counts['observed']+=1;c=BallCandidate(candidate_id='accepted',bbox=s['image_bbox'],confidence=s['detection_confidence']);cues=ground_cues(c,r['camera'],p);counts['accepted_m8_camera']+=cues['m8_valid'];counts['supported_plane']+=cues['plane_supported'];counts['near_foot']+=cues['near_foot'];counts['physical_radius_compatible']+=cues['physical_radius_compatible'];streak=streak+1 if cues['near_foot'] else 0;maxstreak=max(maxstreak,streak);u=r['camera']['motion_evidence'].get('metric_camera',{}).get('uncertainty_pitch_m_proxy');counts['camera_uncertainty_under2_5']+=u is not None and u<=2.5;counts['all_three_contact_and_camera']+=streak>=3 and u is not None and u<=2.5
  report[name]={**dict(counts),'longest_consecutive_near_foot_observations':maxstreak,'explanation':'M14 needs repeated tiny foot-contact tolerance plus supported camera and uncertainty <=2.5m; sparse/mistaken observations and broad player boxes do not establish contact. Image apparent diameter alone does not certify zero height.'}
 save('ground_audit.json',report);print(report)
if __name__=='__main__':run()
