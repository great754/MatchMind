"""Single-ball Kalman evidence with competing births, ambiguity and expiring predictions."""
import math
import numpy as np
from pydantic import Field
from src.query.schema import Closed
from .schema import BallCandidate,BallFrame
from .geometry import camera_motion,warp_point,reconstruct

class BallTrackerConfig(Closed):
 minimum_confidence:float=Field(default=.5,ge=0,le=1)
 birth_hits:int=Field(default=2,ge=2)
 prediction_seconds:float=Field(default=.35,gt=0,le=1)
 ambiguity_margin:float=Field(default=.12,ge=0,le=1)
 max_image_diagonals_per_second:float=Field(default=2.,gt=0)
 movement_slack_px:float=Field(default=12,ge=0)
 maximum_metric_speed_mps:float=Field(default=60,gt=0)
 camera_compensation:bool=True
 physical_constraints:bool=True

class BallTracker:
 def __init__(self,width,height,config=None):
  self.width=width;self.height=height;self.diagonal=math.hypot(width,height);self.cfg=config or BallTrackerConfig();self.x=None;self.P=None;self.last_seen=None;self.last_confirmed=None;self.previous_camera=None;self.previous_time=None;self.births=[];self.track_id=0;self.confidence=0.;self.hits=0;self.contact=0;self.last_pitch=None;self.trail=[];self.prediction_sigma=0.
 def reset(self):self.x=None;self.P=None;self.births=[];self.hits=0;self.contact=0;self.last_pitch=None;self.trail=[];self.prediction_sigma=0.
 def step(self,frame,timestamp,candidates,camera,people=()):
  candidates=[c if isinstance(c,BallCandidate) else BallCandidate.model_validate(c) for c in candidates]
  if any(c.bbox[0]+c.bbox[2]>self.width+1e-5 or c.bbox[1]+c.bbox[3]>self.height+1e-5 for c in candidates):raise ValueError('Ball box outside source image')
  if self.previous_time is not None and timestamp<=self.previous_time:raise ValueError('Ball tracker clock must increase')
  dt=timestamp-self.previous_time if self.previous_time is not None else 0;cut=self.previous_camera is not None and (camera.get('epoch')!=self.previous_camera.get('epoch') or camera.get('continuity')=='cut');motion,source=camera_motion(self.previous_camera,camera) if self.cfg.camera_compensation else (None,'DISABLED_ABLATION');self.previous_camera=camera;self.previous_time=timestamp
  base={'frame':frame,'timestamp':timestamp,'camera_epoch':camera.get('epoch',0),'candidates':candidates,'candidate_count':len(candidates),'detection_status':'DETECTED' if candidates else 'MISSING','last_confirmed_image':self.last_confirmed,'last_confirmed_timestamp':self.last_seen,'track_id':self.track_id or None,'camera_compensation':source}
  if cut:
   self.reset();return BallFrame(**base,tracking_status='LOST',reason='CAMERA_CUT_OR_EPOCH_RESET')
  if self.x is not None and timestamp-self.last_seen>self.cfg.prediction_seconds:self.reset()
  if self.x is not None:
   if motion is not None:
    p=warp_point(motion,self.x[:2]);px=warp_point(motion,self.x[:2]+[1,0]);py=warp_point(motion,self.x[:2]+[0,1])
    if p is not None and px is not None and py is not None:
     jac=np.column_stack([px-p,py-p]);j=np.zeros((4,4));j[:2,:2]=jac;j[2:,2:]=jac;self.x[:2]=p;self.x[2:]=jac@self.x[2:];self.P=j@self.P@j.T+np.diag([4,4,25,25])
   F=np.eye(4);F[0,2]=F[1,3]=dt;acc=self.diagonal*2;Q=np.diag([(acc*dt*dt/2)**2]*2+[(acc*dt)**2]*2);self.x=F@self.x;self.P=F@self.P@F.T+Q
  viable=[c for c in candidates if c.confidence>=self.cfg.minimum_confidence];selected=None;reason='NO_SUPPORTED_CANDIDATE';ranked=[]
  if self.x is not None:
   sigma=math.sqrt(max(np.trace(self.P[:2,:2]),1));gate=self.diagonal*self.cfg.max_image_diagonals_per_second*max(dt,1e-3)+self.cfg.movement_slack_px+3*sigma
   for c in viable:
    residual=float(np.linalg.norm(np.asarray(c.center)-self.x[:2]))
    if self.cfg.physical_constraints and residual>gate:continue
    score=.55*c.confidence+.45*math.exp(-residual/max(12,2*sigma));ranked.append((score,c,residual))
   ranked.sort(key=lambda x:-x[0])
   if ranked and (len(ranked)==1 or ranked[0][0]-ranked[1][0]>=self.cfg.ambiguity_margin):selected=ranked[0][1]
   elif len(ranked)>1:
    self.contact=0;self.trail=[];return BallFrame(**{**base,'detection_status':'AMBIGUOUS'},tracking_status='AMBIGUOUS',tracking_confidence=self.confidence*.5,ambiguity_reason='COMPETING_TEMPORALLY_COMPATIBLE_CANDIDATES',reason='NO_FORCED_HIGHEST_CONFIDENCE_CHOICE')
   elif viable:reason='CANDIDATES_REJECTED_BY_DYNAMIC_DISPLACEMENT_GATE'
  else:
   new=[]
   for c in viable:
    matches=[]
    for b in self.births:
     previous=warp_point(motion,b['point']) if motion is not None else np.asarray(b['point']);distance=np.linalg.norm(np.asarray(c.center)-previous) if previous is not None else float('inf')
     if b['frame']==frame-1 and distance<=max(10,.035*self.diagonal):matches.append((distance,b))
    b=min(matches,key=lambda x:x[0])[1] if matches else None;new.append({'point':c.center,'frame':frame,'hits':b['hits']+1 if b else 1,'score':(.5*c.confidence+.5*b['score']) if b else c.confidence,'candidate':c})
   self.births=new;ready=sorted([b for b in new if b['hits']>=self.cfg.birth_hits],key=lambda b:-b['score'])
   if len(ready)==1:
    selected=ready[0]['candidate'];self.track_id+=1;self.x=np.array([*selected.center,0.,0.]);self.P=np.diag([16.,16.,400.,400.]);self.hits=0
   elif len(ready)>1:return BallFrame(**{**base,'detection_status':'AMBIGUOUS'},tracking_status='AMBIGUOUS',ambiguity_reason='MULTIPLE_REPEATED_BIRTH_HYPOTHESES',reason='COMPETING_BALL_IDENTITIES')
   else:return BallFrame(**base,tracking_status='DETECTED' if viable else 'LOST',reason='AWAIT_REPEATED_TEMPORAL_SUPPORT' if viable else 'NO_DETECTIONS_ABOVE_OPERATING_POINT')
  if selected is not None:
   geo,newcontact=reconstruct(selected,camera,people,self.contact)
   if self.cfg.physical_constraints and geo['pitch_position'] is not None and self.last_pitch is not None:
    elapsed=timestamp-self.last_pitch['time'];distance=np.linalg.norm(np.asarray(geo['pitch_position'])-self.last_pitch['point']);allow=self.cfg.maximum_metric_speed_mps*elapsed+3+2*(geo['coordinate_uncertainty_m']+self.last_pitch['uncertainty'])
    if elapsed>0 and distance>allow:selected=None;reason='IMPOSSIBLE_SUPPORTED_METRIC_ASSOCIATION'
  if selected is not None:
   z=np.asarray(selected.center);R=np.eye(2)*max(2,math.hypot(*selected.bbox[2:])*.15)**2/max(selected.confidence,.1);H=np.array([[1,0,0,0],[0,1,0,0]]);K=self.P@H.T@np.linalg.inv(H@self.P@H.T+R);self.x=self.x+K@(z-H@self.x);self.P=(np.eye(4)-K@H)@self.P;self.hits+=1;self.confidence=selected.confidence*min(1,.6+.1*self.hits);self.last_seen=timestamp;self.last_confirmed=selected.center;self.contact=newcontact;self.prediction_sigma=0.
   if geo['pitch_position'] is not None:self.last_pitch={'time':timestamp,'point':geo['pitch_position'],'uncertainty':geo['coordinate_uncertainty_m']}
   # The displayed current point is the actual detection, never the Kalman mean.
   if self.trail and self.trail[-1]['frame']!=frame-1:self.trail=[]
   self.trail.append({'frame':frame,'timestamp':timestamp,'image':selected.center,'pitch':geo['pitch_position'],'directly_observed':True});self.trail=self.trail[-12:]
   return BallFrame(**{**base,'track_id':self.track_id,'last_confirmed_image':self.last_confirmed,'last_confirmed_timestamp':self.last_seen},tracking_status='TRACKED' if self.hits>=2 else 'DETECTED',image_position=selected.center,image_bbox=selected.bbox,detection_confidence=selected.confidence,tracking_confidence=self.confidence,directly_observed=True,estimated_velocity=tuple(map(float,self.x[2:])),velocity_provenance='KALMAN_IMAGE_PX_PER_SECOND_AFTER_SUPPORTED_CAMERA_WARP; NOT_METRIC_SPEED',reason='REPEATED_NEURAL_AND_TEMPORAL_EVIDENCE',trail=self.trail.copy(),**geo)
  self.contact=0;self.trail=[]
  if self.x is not None and self.last_seen is not None and timestamp-self.last_seen<=self.cfg.prediction_seconds and 0<=self.x[0]<=self.width and 0<=self.x[1]<=self.height:
   self.prediction_sigma=max(math.sqrt(float(np.trace(self.P[:2,:2]))),self.prediction_sigma+self.diagonal*.03*dt)
   overlap=any((box:=p.get('bbox') or p.get('image_bbox')) is not None and box[0]<=self.x[0]<=box[0]+box[2] and box[1]<=self.x[1]<=box[1]+box[3] for p in people);age=timestamp-self.last_seen
   return BallFrame(**base,tracking_status='OCCLUDED' if overlap else 'PREDICTED',predicted_image_position=tuple(map(float,self.x[:2])),prediction_uncertainty_px=self.prediction_sigma,prediction_age_seconds=age,tracking_confidence=self.confidence*math.exp(-age/.15),reason='PREDICTION_OVERLAPS_PLAYER; OCCLUSION_HYPOTHESIS' if overlap else reason+'; BOUNDED_PREDICTION_ONLY')
  return BallFrame(**base,tracking_status='LOST',reason=reason+'; NO_CURRENT_OBSERVATION')
