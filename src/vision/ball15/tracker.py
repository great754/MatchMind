"""M15 visual verification over a frozen M14 causal filter; predictions never become observations."""
import math
import numpy as np
from src.vision.ball.tracker import BallTracker,BallTrackerConfig
from src.vision.ball.schema import BallFrame
from src.vision.ball.geometry import camera_motion,warp_point
from .motion import MotionHistory
from .association import rank

class GuidedTracker:
 def __init__(self,width,height,config,learned_predictor=None,verification=True,motion=True):
  self.width=width;self.height=height;self.config=config;self.filter=BallTracker(width,height,BallTrackerConfig(minimum_confidence=config['minimum_visual'],prediction_seconds=config['prediction_seconds'],ambiguity_margin=config.get('margin',.1)));self.history=MotionHistory();self.predictor=learned_predictor;self.verification=verification;self.motion=motion;self.previous_camera=None;self.last_scores=[];self.prediction=None;self.radius=None
 def forecast(self,timestamp,camera):
  cut=self.previous_camera is not None and (camera.get('epoch')!=self.previous_camera.get('epoch') or camera.get('continuity')=='cut')
  if cut:self.history.reset();return None,None
  h,_=camera_motion(self.previous_camera,camera)
  if h is not None:
   warped=[warp_point(h,p) for p in self.history.points]
   if all(p is not None for p in warped):self.history.points=[tuple(p) for p in warped]
   else:self.history.reset()
  if not self.motion or len(self.history.points)<3:return None,None
  age=timestamp-self.history.times[-1]
  if age>self.config['prediction_seconds']:return None,None
  p=self.predictor(self.history,timestamp,self.width,self.height) if self.predictor and len(self.history.points)>=8 else self.history.predict(timestamp,self.config.get('motion_mode','velocity'));radius=max(5.,self.config['radius90_px']*math.hypot(self.width,self.height)/math.hypot(1920,1080)*max(1.,age/.04))
  return p,radius
 def step(self,frame,timestamp,candidates,camera,people=(),forecast=None):
  prediction,radius=forecast if forecast is not None else self.forecast(timestamp,camera);self.previous_camera=camera;self.prediction=prediction;self.radius=radius;age=timestamp-self.history.times[-1] if self.history.times else 0;previous=self.history.points[-1] if self.history.points else None
  ranked=rank(candidates,prediction,radius,previous,people,self.width,self.height,age,self.config.get('association_model') if self.config.get('association_kind')=='logistic' else None);self.last_scores=ranked
  calibration=self.config.get('confidence_calibration')
  if calibration:
   for entry in ranked:entry['calibrated_visual_probability']=float(np.interp(entry['features'][0],calibration['score_knots'],calibration['probability_knots']))
  selected=candidates;ambiguity=False
  if self.verification:
   eligible=[s for s in ranked if s['score']>=self.config['score_threshold'] and next(c['confidence'] for c in candidates if c['candidate_id']==s['candidate_id'])>=self.config['minimum_visual']]
   if eligible and (len(eligible)==1 or eligible[0]['score']-eligible[1]['score']>=self.config.get('margin',.1)):selected=[next(c for c in candidates if c['candidate_id']==eligible[0]['candidate_id'])]
   else:selected=[];ambiguity=len(eligible)>1
  state=self.filter.step(frame,timestamp,selected,camera,people);data=state.model_dump();data['version']='m15';data['candidates']=candidates;data['candidate_count']=len(candidates)
  if state.directly_observed:self.history.append(state.image_position,timestamp,state.detection_confidence)
  elif prediction is not None and state.predicted_image_position is not None and not ambiguity:
   if 0<=prediction[0]<=self.width and 0<=prediction[1]<=self.height:data['predicted_image_position']=tuple(map(float,prediction));data['prediction_uncertainty_px']=radius
   else:data.update(predicted_image_position=None,tracking_status='LOST',prediction_age_seconds=None,prediction_uncertainty_px=None)
  elif state.predicted_image_position is not None:
   # No calibrated reliable history, so no unsupported filter continuation is displayed.
   data.update(predicted_image_position=None,tracking_status='LOST',prediction_age_seconds=None,prediction_uncertainty_px=None)
  if ambiguity:data.update(tracking_status='AMBIGUOUS',predicted_image_position=None,ambiguity_reason='M15_VERIFIER_ALTERNATIVES_TOO_CLOSE')
  if self.history.times and timestamp-self.history.times[-1]>self.config['prediction_seconds']:self.history.reset()
  # M14's contact heuristic is preserved, but not promoted to metric authority without independent M15 validation.
  if data['pitch_position'] is not None and not self.config.get('ground_validation_passed',False):data.update(pitch_position=None,pitch_position_confidence=None,coordinate_uncertainty_m=None,coordinate_state='HEIGHT_UNKNOWN',coordinate_provenance='M14_CONTACT_HEURISTIC_ONLY; M15_INDEPENDENT_GROUND_VALIDATION_UNAVAILABLE')
  if not self.config.get('ground_validation_passed',False):
   data['trail']=[{**entry,'pitch':None} for entry in data['trail']];self.filter.last_pitch=None
  data.update(candidate_verification=ranked,prediction_region={'center':list(map(float,prediction)),'radius90_px':radius,'calibration':'held-out validation image trajectory errors; camera approximation'} if prediction is not None else None,motion_model=self.config.get('motion_kind','velocity'),search_stage='unknown')
  return BallFrame.model_validate(data)
