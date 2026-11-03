"""Closed, finite VIDEO_CV ball evidence. Current, predicted and historical are separate."""
from typing import Literal
from pydantic import Field,model_validator
from src.query.schema import Closed

class BallCandidate(Closed):
 candidate_id:str
 bbox:tuple[float,float,float,float]
 confidence:float=Field(ge=0,le=1)
 inference_view:str='full'
 source:Literal['VIDEO_CV_DETECTOR']='VIDEO_CV_DETECTOR'
 @model_validator(mode='after')
 def valid(self):
  if self.bbox[2]<=0 or self.bbox[3]<=0 or self.bbox[0]<0 or self.bbox[1]<0:raise ValueError('Positive in-image ball box required')
  return self
 @property
 def center(self):return (self.bbox[0]+self.bbox[2]/2,self.bbox[1]+self.bbox[3]/2)

class PredictionRegion(Closed):
 center:tuple[float,float]
 radius90_px:float=Field(ge=0)
 calibration:str

class CandidateVerification(Closed):
 candidate_id:str
 score:float=Field(ge=0,le=1)
 features:list[float]
 calibrated_visual_probability:float|None=Field(default=None,ge=0,le=1)

class BallFrame(Closed):
 version:Literal['m14','m15']='m14'
 frame:int=Field(ge=0)
 timestamp:float=Field(ge=0)
 camera_epoch:int=Field(ge=0)
 detection_status:Literal['DETECTED','MISSING','AMBIGUOUS']='MISSING'
 tracking_status:Literal['DETECTED','TRACKED','PREDICTED','OCCLUDED','LOST','AMBIGUOUS']='LOST'
 image_position:tuple[float,float]|None=None
 image_bbox:tuple[float,float,float,float]|None=None
 predicted_image_position:tuple[float,float]|None=None
 prediction_uncertainty_px:float|None=Field(default=None,ge=0)
 prediction_age_seconds:float|None=Field(default=None,ge=0)
 last_confirmed_image:tuple[float,float]|None=None
 last_confirmed_timestamp:float|None=None
 detection_confidence:float|None=Field(default=None,ge=0,le=1)
 tracking_confidence:float=Field(default=0,ge=0,le=1)
 pitch_position:tuple[float,float]|None=None
 pitch_position_confidence:float|None=Field(default=None,ge=0,le=1)
 coordinate_state:Literal['GROUND_SUPPORTED','AIRBORNE_SUSPECTED','HEIGHT_UNKNOWN','METRIC_UNAVAILABLE']='METRIC_UNAVAILABLE'
 coordinate_provenance:str='UNKNOWN'
 coordinate_uncertainty_m:float|None=Field(default=None,ge=0)
 plane_hypothesis:tuple[float,float]|None=None
 ground_support_frames:int=0
 estimated_velocity:tuple[float,float]|None=None
 velocity_provenance:str='UNKNOWN'
 metric_velocity:tuple[float,float]|None=None
 camera_compensation:str='NONE'
 track_id:int|None=None
 candidate_count:int=Field(default=0,ge=0)
 candidates:list[BallCandidate]=Field(default_factory=list)
 ambiguity_reason:str|None=None
 reason:str='NO_SUPPORTED_BALL'
 candidate_verification:list[CandidateVerification]=Field(default_factory=list)
 prediction_region:PredictionRegion|None=None
 motion_model:str|None=None
 search_stage:str|None=None
 observation_source:Literal['VIDEO_CV']='VIDEO_CV'
 directly_observed:bool=False
 trail:list[dict]=Field(default_factory=list)
 @model_validator(mode='after')
 def provenance(self):
  if any(v.candidate_id not in {c.candidate_id for c in self.candidates} for v in self.candidate_verification):raise ValueError('Verification must refer to retained visual candidates')
  if self.candidate_count!=len(self.candidates):raise ValueError('Candidate count mismatch')
  if self.directly_observed:
   if self.image_position is None or self.image_bbox is None or self.detection_confidence is None or self.tracking_status not in ['DETECTED','TRACKED']:raise ValueError('Observed ball requires supported current detection')
   if not any(c.bbox==self.image_bbox and all(abs(a-b)<1e-6 for a,b in zip(c.center,self.image_position)) and abs(c.confidence-self.detection_confidence)<1e-6 for c in self.candidates):raise ValueError('Observed ball must reference an actual retained candidate')
  elif self.image_position is not None or self.image_bbox is not None or self.pitch_position is not None:raise ValueError('Prediction/history cannot masquerade as current observation')
  if self.tracking_status=='TRACKED' and not self.directly_observed:raise ValueError('Tracked current point requires observation')
  if self.pitch_position is not None and (self.coordinate_state!='GROUND_SUPPORTED' or self.pitch_position_confidence is None or not self.directly_observed):raise ValueError('Metric ball requires explicit ground support and current observation')
  if self.coordinate_state in ['AIRBORNE_SUSPECTED','HEIGHT_UNKNOWN','METRIC_UNAVAILABLE'] and self.pitch_position is not None:raise ValueError('Unknown/airborne ball cannot have authoritative metric position')
  if self.tracking_status in ['LOST','AMBIGUOUS'] and self.predicted_image_position is not None:raise ValueError('Lost/ambiguous current prediction is not displayed')
  return self

class VideoBallObservation(Closed):
 source:Literal['VIDEO_CV']='VIDEO_CV'
 schema_version:Literal['ball-m14-v1','ball-m15-v1']='ball-m14-v1'
 recording_id:str
 frame:int=Field(ge=0)
 timestamp:float=Field(ge=0)
 fps:float=Field(gt=0)
 detection_status:str
 tracking_status:str
 image:tuple[float,float]|None=None
 image_bbox:tuple[float,float,float,float]|None=None
 predicted_image:tuple[float,float]|None=None
 pitch:tuple[float,float]|None=None
 coordinate_state:str
 coordinate_provenance:str
 detection_confidence:float|None=Field(default=None,ge=0,le=1)
 tracking_confidence:float=Field(ge=0,le=1)
 directly_observed:bool
 camera_epoch:int
 detector_sha256:str
 @model_validator(mode='after')
 def valid(self):
  if abs(self.timestamp-self.frame/self.fps)>1e-6:raise ValueError('Ball observation clock mismatch')
  if self.pitch is not None and self.coordinate_state!='GROUND_SUPPORTED':raise ValueError('Canonical metric ball requires ground support')
  if not self.directly_observed and (self.image is not None or self.pitch is not None):raise ValueError('Non-observed ball cannot supply observed coordinates')
  return self

def canonical(recording_id,fps,state,detector_hash):
 return VideoBallObservation(schema_version='ball-m15-v1' if state.version=='m15' else 'ball-m14-v1',recording_id=recording_id,frame=state.frame,timestamp=state.timestamp,fps=fps,detection_status=state.detection_status,tracking_status=state.tracking_status,image=state.image_position,image_bbox=state.image_bbox,predicted_image=state.predicted_image_position,pitch=state.pitch_position,coordinate_state=state.coordinate_state,coordinate_provenance=state.coordinate_provenance,detection_confidence=state.detection_confidence,tracking_confidence=state.tracking_confidence,directly_observed=state.directly_observed,camera_epoch=state.camera_epoch,detector_sha256=detector_hash)
