"""Reuse M8 transforms only. Pitch-plane intersections are not airborne ground truth."""
from types import SimpleNamespace
import math
import cv2,numpy as np
from src.vision.world.player_geometry import project_player

def transform(camera):
 c=camera.get('calibration')
 if not camera.get('calibration_valid') or not c:return None
 h=np.asarray(c.get('image_to_pitch'),float)
 if h.size!=9 or not np.isfinite(h).all():return None
 h=h.reshape(3,3)
 return h if abs(np.linalg.det(h))>1e-10 else None

def warp_point(h,p):
 v=np.asarray(h,float)@np.array([*p,1.]);return v[:2]/v[2] if np.isfinite(v).all() and abs(v[2])>1e-10 else None

def camera_motion(previous,current):
 if previous is None or previous.get('epoch')!=current.get('epoch') or current.get('continuity')=='cut':return None,'NONE_EPOCH_OR_CUT'
 a=transform(previous);b=transform(current)
 if a is not None and b is not None:
  h=np.linalg.inv(b)@a
  if np.isfinite(h).all():return h,'M8_ACCEPTED_PLANE_WARP; HEIGHT_UNKNOWN_APPROXIMATION'
 field=current.get('motion_evidence',{}).get('field_motion',{})
 if field.get('accepted') and field.get('interval_frames')==1 and field.get('homography') is not None:
  h=np.asarray(field['homography'],float)
  if h.shape==(3,3) and np.isfinite(h).all():return h,'M8_MEASURED_ADJACENT_FIELD_MOTION'
 return None,'NONE_NO_SUPPORTED_M8_MOTION'

def plane_intersection(point,camera):
 c=camera.get('calibration') or {};proxy=SimpleNamespace(calibration_valid=camera.get('calibration_valid',False),calibration=SimpleNamespace(validity_region=c.get('validity_region',[]),image_to_pitch=c.get('image_to_pitch',[])))
 return project_player(point,proxy)

def reconstruct(candidate,camera,people,contact_streak):
 point=(candidate.bbox[0]+candidate.bbox[2]/2,candidate.bbox[1]+candidate.bbox[3]);plane,reason=plane_intersection(point,camera);metric=camera.get('motion_evidence',{}).get('metric_camera',{});u=metric.get('uncertainty_pitch_m_proxy');answer={'coordinate_state':'METRIC_UNAVAILABLE','coordinate_provenance':reason,'pitch_position':None,'pitch_position_confidence':None,'coordinate_uncertainty_m':None,'plane_hypothesis':plane,'ground_support_frames':0};near=[]
 if plane is None:
  region=(camera.get('calibration') or {}).get('validity_region',[])
  if region and candidate.center[1]<min(p[1] for p in region)-5:answer.update(coordinate_state='AIRBORNE_SUSPECTED',coordinate_provenance='ABOVE_SUPPORTED_PITCH_REGION; AIRBORNE_OR_FALSE_CANDIDATE')
  return answer,0
 answer.update(coordinate_state='HEIGHT_UNKNOWN',coordinate_provenance='M8_PLANE_INTERSECTION_ONLY; BALL_HEIGHT_UNVERIFIED')
 for p in people:
  box=p.get('bbox') or p.get('image_bbox')
  if box is None:continue
  x,y,w,h=box;foot=(x+w/2,y+h)
  if abs(point[1]-foot[1])<=max(3,.3*candidate.bbox[3]) and abs(point[0]-foot[0])<=max(4,.12*w) and .02<=candidate.bbox[3]/max(h,1)<=.2:near.append(p)
 streak=contact_streak+1 if near else 0;answer['ground_support_frames']=streak
 # Low-confidence, explicit ground-contact hypothesis; never a 3D height claim.
 if streak>=3 and u is not None and math.isfinite(u) and u<=2.5:
  h=transform(camera);a=warp_point(h,point);dx=warp_point(h,(point[0]+candidate.bbox[2]/2,point[1]));dy=warp_point(h,(point[0],point[1]+candidate.bbox[3]/2));extra=max(np.linalg.norm(dx-a),np.linalg.norm(dy-a)) if dx is not None and dy is not None else 3.
  answer.update(coordinate_state='GROUND_SUPPORTED',coordinate_provenance='VIDEO_CV_M14_REPEATED_FOOT_CONTACT_HEURISTIC_M8; EXPERIMENTAL_PLANE_HYPOTHESIS',pitch_position=plane,pitch_position_confidence=min(.35,candidate.confidence*.35/(1+u+extra)),coordinate_uncertainty_m=float(u+extra+1.))
 return answer,streak
