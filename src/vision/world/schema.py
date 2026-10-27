"""Inspectible prototype state: support scores are not calibrated probabilities."""
from typing import Literal
from pydantic import Field, model_validator
from src.query.schema import Closed
from src.observations.schema import PitchCalibrationObservation,CameraProjectionEvidence,TeamBeliefEvidence

class Config(Closed):
    track_grace_seconds: float = Field(default=.6,gt=0)
    off_camera_seconds: float = Field(default=1.,gt=0)
    min_track_hits: int = Field(default=2,ge=1)
    min_iou: float = Field(default=.05,ge=0,le=1)
    max_image_motion_per_second: float = Field(default=1.5,gt=0)
    appearance_gate: float = Field(default=.90,ge=0,le=1)
    association_threshold: float = Field(default=.90,ge=0,le=1)
    ambiguity_margin: float = Field(default=.08,ge=0,le=1)
    max_player_speed_mps: float = Field(default=12.,gt=0)
    movement_slack_m: float = Field(default=3.,ge=0)
    position_prior_seconds: float = Field(default=3.,gt=0)
    allow_appearance_only_reassociation: bool = False
    nms_threshold: float = Field(default=.5,gt=0,le=1)
    team_min_samples: int = Field(default=8,ge=2)
    team_center_separation: float = Field(default=25.,gt=0)
    team_max_color_distance: float = Field(default=65.,gt=0)
    team_margin: float = Field(default=.35,ge=0,le=1)

class KitObservation(Closed):
    torso_bbox: tuple[float,float,float,float] | None = None
    lab: tuple[float,float,float] | None = None
    dominant_bgr: tuple[int,int,int] | None = None
    background_lab: tuple[float,float,float] | None = None
    foreground_fraction: float = Field(default=0,ge=0,le=1)
    field_context: float = Field(default=0,ge=0,le=1)
    quality: float = Field(default=0,ge=0,le=1)
    cluster_distances: tuple[float,float] | None = None
    reason: str = 'No kit evidence'

class SizeEvidence(Closed):
    reference_height: float | None = None
    height_ratio: float | None = None
    source: str = 'unknown'
    support: float = Field(default=1,ge=0,le=1)
    flags: list[str] = Field(default_factory=list)

class TeamVotes(Closed):
    team_a: float = 0
    team_b: float = 0
    unknown: int = 0

class KitSample(Closed):
    timestamp: float
    lab: tuple[float,float,float]
    quality: float

class TeamMemory(Closed):
    votes: TeamVotes = Field(default_factory=TeamVotes)
    evidence_count: int = 0
    current_observation: Literal['team_a','team_b'] | None = None
    contradictory_streak: int = 0
    recent: list[KitSample] = Field(default_factory=list)
    historical_lab: tuple[float,float,float] | None = None
    stable: bool = False

class EntityEvent(Closed):
    frame: int
    timestamp: float
    kind: str
    reason: str
    previous: str | None = None
    current: str | None = None

class RosterState(Closed):
    team_a: list[str] = Field(default_factory=list)
    team_b: list[str] = Field(default_factory=list)
    unknown: list[str] = Field(default_factory=list)
    candidate_people: int = 0

class Detection(Closed):
    bbox: tuple[float,float,float,float]  # xywh, source image pixels
    detection_confidence: float = Field(ge=0,le=1)
    appearance: tuple[float,...] | None = None
    team: Literal['team_a','team_b'] | None = None
    team_confidence: float | None = Field(default=None,ge=0,le=1)
    kit: KitObservation | None = None
    size_evidence: SizeEvidence | None = None
    @model_validator(mode='after')
    def valid_box(self):
        if self.bbox[2]<=0 or self.bbox[3]<=0:raise ValueError('Positive box dimensions required')
        return self
    @property
    def foot(self):return (self.bbox[0]+self.bbox[2]/2,self.bbox[1]+self.bbox[3])

class TrackObservation(Detection):
    track_id: str
    hits: int = Field(ge=1)
    pitch: tuple[float,float] | None = None
    localization_confidence: float | None = Field(default=None,ge=0,le=1)
    localization_source: str = 'unknown'
    projection_reason: str = 'UNKNOWN'
    team_belief: TeamBeliefEvidence | None = None
    position_diagnostic: dict | None = None
    projection_camera: CameraProjectionEvidence | None = None
    image_velocity: tuple[float,float] | None = None

class CameraEstimate(Closed):
    frame: int
    timestamp: float
    epoch: int
    transform: tuple[float,...]
    source: str
    visible_region: list[tuple[float,float]]
    landmark_ids: list[str]
    support: float = Field(ge=0,le=1)
    motion_error_budget_px: float = Field(ge=0)

class CameraHypothesis(Closed):
    frame: int | None = None
    epoch: int | None = None
    transform: tuple[float,...] | None = None
    source: str
    accepted: bool
    heldout_mean_px: float | None = None
    heldout_p90_px: float | None = None
    correction_norm_px: float | None = None
    reason: str

class CameraFailure(Closed):
    frame: int
    timestamp: float
    reason: str

class CameraMemory(Closed):
    epoch: int = 0
    last_direct: CameraEstimate | None = None
    last_accepted: CameraEstimate | None = None
    last_constraint_timestamp: float | None = None
    accepted_history: list[CameraEstimate] = Field(default_factory=list)
    failure_history: list[CameraFailure] = Field(default_factory=list)
    candidates: list[CameraHypothesis] = Field(default_factory=list)
    pan_direction: str = 'unknown'
    scale_range: tuple[float,float] | None = None
    uncertainty_kind: str = 'accumulated relative image-motion budget; not absolute metric accuracy or probability'

class ImagePitchSegment(Closed):
    kind: str
    start: tuple[float,float]
    end: tuple[float,float]

class CameraState(Closed):
    frame: int = Field(ge=0)
    timestamp: float = Field(ge=0)
    epoch: int = Field(default=0,ge=0)
    mode: Literal['wide','medium','close_up','unknown'] = 'unknown'
    continuity: Literal['continuous','cut','unknown'] = 'unknown'
    calibration: PitchCalibrationObservation | None = None
    calibration_valid: bool = False
    calibration_source_kind: Literal['direct_calibrated','camera_propagated','partial_constrained','temporal_predicted','temporal_corrected','bidirectional_interpolated'] = 'direct_calibrated'
    calibration_confidence: float | None = Field(default=None,ge=0,le=1)
    visible_pitch_region: list[tuple[float,float]] | None = None
    memory: CameraMemory | None = None
    projected_pitch_geometry: list[ImagePitchSegment] = Field(default_factory=list)
    calibration_status: Literal['valid','partial','lost','unknown'] = 'unknown'
    pitch_evidence: dict = Field(default_factory=dict)
    motion_evidence: dict = Field(default_factory=dict)
    last_reliable_transform: tuple[float,...] | None = None
    last_reliable_timestamp: float | None = None
    source: str = 'unknown; automatic pitch perception not implemented'

class BallWorldState(Closed):
    visibility: Literal['visible','occluded','unknown'] = 'unknown'
    current_pitch: tuple[float,float] | None = None
    image_position: tuple[float,float] | None = None
    last_seen: float | None = None
    last_reliable_pitch: tuple[float,float] | None = None
    detection_confidence: float | None = None
    localization_confidence: float | None = None
    source: str = 'unknown; ball perception not implemented'

class AssociationEvidence(Closed):
    entity_id: str
    appearance: float | None = None
    team_compatible: bool = True
    elapsed_seconds: float
    displacement_m: float | None = None
    movement_feasible: bool | None = None
    position_prior: float | None = None
    trajectory_error_m: float | None = None
    image_trajectory_error: float | None = None
    image_motion_feasible: bool | None = None
    bbox_consistency: float | None = None
    camera_region_compatible: bool | None = None
    continuity_source: str = "unknown"
    score: float = Field(ge=0,le=1)
    eligible: bool
    reason: str

class IdentityHypothesis(Closed):
    entity_id: str | None
    support_score: float = Field(ge=0,le=1)
    evidence: AssociationEvidence | None = None

class PlayerEntity(Closed):
    entity_id: str
    team: Literal['team_a','team_b'] | None = None
    team_confidence: float | None = None
    visibility: Literal['visible','temporarily_missing','off_camera']
    identity_status: Literal['provisional','associated'] = 'provisional'
    identity_confidence: float | None = None
    track_id: str | None = None
    last_track_id: str | None = None
    track_history: list[str] = Field(default_factory=list)
    last_reliable_camera_epoch: int | None = None
    first_seen: float | None = None
    last_image_bbox: tuple[float,float,float,float] | None = None
    last_image_foot: tuple[float,float] | None = None
    last_image_velocity: tuple[float,float] | None = None
    memory_state: Literal['visible_tracked','temporarily_occluded','off_camera','lost'] = 'visible_tracked'
    memory_reason: str = 'current detection'
    team_memory: TeamMemory = Field(default_factory=TeamMemory)
    role: Literal['team_a_outfield','team_b_outfield','goalkeeper_a','goalkeeper_b','official','unknown_person'] = 'unknown_person'
    history: list[EntityEvent] = Field(default_factory=list)
    current_kit: KitObservation | None = None
    last_kit: KitObservation | None = None
    size_evidence: SizeEvidence | None = None
    projection_reason: str = 'UNKNOWN'
    team_belief: TeamBeliefEvidence | None = None
    position_diagnostic: dict | None = None
    projection_camera: CameraProjectionEvidence | None = None
    last_seen: float
    last_seen_frame: int
    image_bbox: tuple[float,float,float,float] | None = None
    image_foot: tuple[float,float] | None = None
    detection_confidence: float | None = None
    current_pitch: tuple[float,float] | None = None
    position_source: Literal['direct_calibrated','camera_propagated','partial_constrained','temporal_predicted','temporal_corrected','bidirectional_interpolated','unknown'] = 'unknown'
    localization_confidence: float | None = None
    last_reliable_pitch: tuple[float,float] | None = None
    last_localized_at: float | None = None
    last_velocity: tuple[float,float] | None = None
    appearance: tuple[float,...] | None = None
    appearance_evidence_count: int = 0
    camera_epoch: int = 0
    hypotheses: list[IdentityHypothesis] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)

class UnassignedTrack(Closed):
    observation: TrackObservation
    status: Literal['unknown','ambiguous']
    hypotheses: list[IdentityHypothesis]

class WorldState(Closed):
    recording_id: str
    frame: int
    timestamp: float
    camera: CameraState
    entities: list[PlayerEntity]
    unassigned: list[UnassignedTrack]
    decisions: list[dict] = Field(default_factory=list)
    reconciliations: list[dict] = Field(default_factory=list)
    roster: RosterState = Field(default_factory=RosterState)
    ball: BallWorldState = Field(default_factory=BallWorldState)
