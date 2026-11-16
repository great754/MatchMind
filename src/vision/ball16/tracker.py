"""Causal multi-hypothesis ball tracking; only supported visual candidates become current observations."""
import math
import numpy as np
from src.vision.ball.schema import BallFrame,BallCandidate
from src.vision.ball.geometry import camera_motion,warp_point
from src.vision.ball.tracker import BallTracker,BallTrackerConfig

class MultiHypothesisTracker:
 def __init__(self,width,height,threshold=.35,margin=.1,ttl=.12):
  self.width=width;self.height=height;self.threshold=threshold;self.margin=margin;self.ttl=ttl;self.hypotheses=[];self.previous_camera=None;self.last_time=None;self.counter=0;self.filter=BallTracker(width,height,BallTrackerConfig(minimum_confidence=threshold,prediction_seconds=ttl));self.last_selected=None
 def predict(self,timestamp):
  if not self.hypotheses:return None,None
  h=max(self.hypotheses,key=lambda h:h['weight']);age=timestamp-h['time']
  if age>self.ttl:return None,None
  return h['point']+h['velocity']*age,max(6,h['sigma']+age*math.hypot(self.width,self.height)*.1)
 def step(self,frame,timestamp,candidates,camera,people=(),verifier=None):
  if self.last_time is not None and timestamp<=self.last_time:raise ValueError('Clock must increase')
  cut=self.previous_camera is not None and (camera.get('epoch')!=self.previous_camera.get('epoch') or camera.get('continuity')=='cut');motion,_=camera_motion(self.previous_camera,camera)
  if cut:self.hypotheses=[]
  if motion is not None:
   for h in self.hypotheses:
    p=warp_point(motion,h['point']);q=warp_point(motion,h['point']+h['velocity']*.04)
    if p is not None and q is not None:h['point']=p;h['velocity']=(q-p)/.04
  self.hypotheses=[h for h in self.hypotheses if timestamp-h['time']<=self.ttl];self.previous_camera=camera;self.last_time=timestamp;new=[];ranked=[]
  for c in candidates:
   b=BallCandidate.model_validate(c)
   if b.confidence<self.threshold:continue
   z=np.asarray(b.center);matches=[]
   for h in self.hypotheses:
    age=timestamp-h['time'];pred=h['point']+h['velocity']*age;sigma=h['sigma']+age*math.hypot(self.width,self.height)*.1;distance=float(np.linalg.norm(z-pred));score=.6*b.confidence+.4*math.exp(-.5*(distance/max(sigma,6))**2)
    if distance<max(16,3*sigma):matches.append((score,h))
   if matches:
    score,h=max(matches,key=lambda x:x[0]);age=timestamp-h['time'];v=.5*h['velocity']+.5*(z-h['point'])/max(age,.001);new.append({'point':z,'velocity':v,'sigma':max(4,h['sigma']*.8),'time':timestamp,'hits':h['hits']+1,'weight':.5*h['weight']+.5*score,'candidate':c})
   else:new.append({'point':z,'velocity':np.zeros(2),'sigma':8.,'time':timestamp,'hits':1,'weight':b.confidence*.6,'candidate':c})
  # Retain unmatched hypotheses with growing uncertainty, bounded age and decaying weight.
  for h in self.hypotheses:
   if not any(np.linalg.norm(x['point']-h['point'])<4 for x in new):new.append({**h,'weight':h['weight']*.7,'sigma':h['sigma']+4,'candidate':None})
  self.hypotheses=sorted(new,key=lambda h:-h['weight'])[:4];eligible=[h for h in self.hypotheses if h['hits']>=2 and h['candidate'] is not None];selected=None;ambiguous=False
  if eligible:
   if len(eligible)==1 or eligible[0]['weight']-eligible[1]['weight']>=self.margin:selected=eligible[0]['candidate']
   else:ambiguous=True
  if selected is not None and verifier is not None and verifier(selected)<.5:selected=None
  state=self.filter.step(frame,timestamp,[selected] if selected else [],camera,people);data=state.model_dump();data.update(version='m16',detection_status='AMBIGUOUS' if ambiguous else 'DETECTED' if candidates else 'MISSING',candidates=candidates,candidate_count=len(candidates),motion_model='M16_MULTI_HYPOTHESIS_CV_DYNAMIC_GATE',search_stage='GLOBAL_PLUS_ADDITIVE_RECOVERY',candidate_verification=[])
  if ambiguous:data.update(tracking_status='AMBIGUOUS',predicted_image_position=None,ambiguity_reason='M16_COMPETING_TRAJECTORY_HYPOTHESES')
  # No metric authority without independent ground/metric validation; the old gate is not relaxed.
  data.update(pitch_position=None,pitch_position_confidence=None,coordinate_uncertainty_m=None,coordinate_state='AIRBORNE_SUSPECTED' if data['coordinate_state']=='AIRBORNE_SUSPECTED' else 'HEIGHT_UNKNOWN' if data['plane_hypothesis'] is not None else 'METRIC_UNAVAILABLE',coordinate_provenance='M16_NO_INDEPENDENT_HEIGHT_OR_METRIC_CERTIFICATE',trail=[{**p,'pitch':None} for p in data['trail']]);self.filter.last_pitch=None;self.last_selected=selected
  return BallFrame.model_validate(data)
