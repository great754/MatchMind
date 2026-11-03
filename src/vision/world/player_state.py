"""M9 authoritative anonymous player state, independent of M8 geometry/raw tracks."""
from typing import Literal
from pydantic import Field,model_validator
from src.query.schema import Closed

TeamLabel=Literal['TEAM_A','TEAM_B','OFFICIAL','OTHER','UNKNOWN']
class PlayerWorldConfig(Closed):
    min_track_observations:int=Field(default=3,ge=2)
    max_speed_mps:float=Field(default=12.,gt=0)
    movement_slack_m:float=Field(default=3.,ge=0)
    max_reconnect_seconds:float=Field(default=8.,gt=0)
    max_metric_uncertainty_m:float=Field(default=3.,gt=0)
    minimum_appearance:float=Field(default=.82,ge=0,le=1)
    minimum_score:float=Field(default=.78,ge=0,le=1)
    ambiguity_margin:float=Field(default=.06,gt=0)
    long_gap_decay_start_seconds:float=Field(default=3.,gt=0)
    long_gap_decay_seconds:float=Field(default=3.,gt=0)
    lost_after_seconds:float=Field(default=8.,gt=0)
    team_sample_seconds:float=Field(default=.12,gt=0)
    team_min_samples:int=Field(default=3,ge=2)
    team_switch_samples:int=Field(default=8,ge=3)
    team_switch_quality:float=Field(default=.45,ge=0,le=1)
    team_switch_strength:float=Field(default=.6,ge=0,le=1)
    team_switch_span_seconds:float=Field(default=.8,gt=0)

class TeamIntelligence(Closed):
    label:TeamLabel='UNKNOWN'
    confidence:float=Field(default=0,ge=0,le=1)
    probabilities:dict[str,float]=Field(default_factory=lambda:{k:.2 for k in ['TEAM_A','TEAM_B','OFFICIAL','OTHER','UNKNOWN']})
    source:str='UNKNOWN'
    current_visual_label:TeamLabel='UNKNOWN'
    current_visual_quality:float=Field(default=0,ge=0,le=1)
    independent_samples:int=0
    switch_candidate:TeamLabel|None=None
    events:list[dict]=Field(default_factory=list)
    calibrated:bool=False

class ReIDCandidate(Closed):
    entity_id:str
    segment_id:str
    appearance_similarity:float|None=None
    appearance_quality:float=0
    team_compatible:bool=True
    elapsed_seconds:float
    distance_m:float|None=None
    required_average_speed_mps:float|None=None
    allowed_distance_m:float|None=None
    position_uncertainty_m:float|None=None
    position_weight:float=0
    trajectory_error_m:float|None=None
    visibility_compatible:bool|None=None
    score:float=Field(ge=0,le=1)
    eligible:bool
    reason:str
    evidence_frames:list[int]=Field(default_factory=list)

class PersistentPerson(Closed):
    entity_id:str
    current_track_id:str|None=None
    current_segment_id:str|None=None
    visibility_state:Literal['VISIBLE','OFF_CAMERA','OCCLUDED','IDENTITY_AMBIGUOUS','LOST']
    identity_status:Literal['PROVISIONAL','ASSOCIATED','AMBIGUOUS']
    identity_confidence:float=Field(ge=0,le=1)
    team:TeamIntelligence
    current_position:tuple[float,float]|None=None
    position_confidence:float|None=Field(default=None,ge=0,le=1)
    position_uncertainty_m:float|None=None
    position_provenance:str='UNKNOWN'
    camera_epoch:int
    camera_state:str
    camera_source:str
    last_reliable_position:tuple[float,float]|None=None
    last_reliable_timestamp:float|None=None
    last_seen_time:float
    last_seen_velocity:tuple[float,float]|None=None
    last_disappearance_time:float|None=None
    last_reappearance_time:float|None=None
    visibility_reason:str
    appearance_confidence:float=Field(default=0,ge=0,le=1)
    current_appearance:dict|None=None
    appearance_evidence:list[dict]=Field(default_factory=list)
    track_ids:list[str]=Field(default_factory=list)
    segment_ids:list[str]=Field(default_factory=list)
    reid_candidates:list[ReIDCandidate]=Field(default_factory=list)
    reassociation_events:list[dict]=Field(default_factory=list)
    jersey_number:dict=Field(default_factory=dict)
    @model_validator(mode='after')
    def current_is_observed(self):
        if self.visibility_state not in ('VISIBLE','IDENTITY_AMBIGUOUS') and self.current_position is not None:raise ValueError('Historical position cannot become current')
        if self.current_position is None and (self.position_confidence is not None or self.position_provenance!='UNKNOWN'):raise ValueError('Unknown position has no current confidence/provenance')
        return self

class UnresolvedPlayerDetection(Closed):
    observation_id:str
    track_id:str
    bbox:tuple[float,float,float,float]
    team:TeamIntelligence
    current_position:tuple[float,float]|None=None
    position_confidence:float|None=None
    position_provenance:str='UNKNOWN'
    identity_confidence:float=0
    reason:str='INSUFFICIENT_REPEATED_OBSERVATIONS_FOR_PERSISTENT_ENTITY'

class PlayerWorldFrame(Closed):
    version:Literal['m9']='m9'
    source:Literal['VIDEO_CV']='VIDEO_CV'
    frame:int
    timestamp:float
    players:list[PersistentPerson]
    unresolved_detections:list[UnresolvedPlayerDetection]=Field(default_factory=list)
    offline_future_evidence:bool=True
    @model_validator(mode='after')
    def one_to_one(self):
        ids=[p.entity_id for p in self.players];tracks=[p.current_track_id for p in self.players if p.current_track_id is not None]+[p.track_id for p in self.unresolved_detections]
        if len(ids)!=len(set(ids)) or len(tracks)!=len(set(tracks)):raise ValueError('Duplicate entity or simultaneous track assignment')
        return self

class VideoPlayerObservation(Closed):
    source:Literal['VIDEO_CV']='VIDEO_CV'
    frame:int
    timestamp:float
    entity_id:str|None
    track_id:str|None
    team:TeamLabel
    team_confidence:float
    visible:bool
    visibility_state:str
    x:float|None
    y:float|None
    position_confidence:float|None
    position_provenance:str
    identity_confidence:float
