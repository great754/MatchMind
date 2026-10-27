"""Metric pitch coordinates use SoccerTrack's 105 x 68 m convention.

Unknown values stay null. Actor IDs identify tracks, never jerseys or people.
"""
from typing import Literal
from pydantic import Field, model_validator
from src.query.schema import Closed

class Provenance(Closed):
    provider: str = Field(min_length=1)
    source: str = Field(min_length=1)
    quality: Literal['provided','reconstructed','observed','inferred','propagated','last_known','unknown'] = 'unknown'
    confidence: float | None = Field(default=None,ge=0,le=1)

class CameraProjectionEvidence(Closed):
    image_foot: tuple[float,float] | None = None
    epoch: int
    state: str
    source: str
    support: float | None = Field(default=None,ge=0,le=1)
    last_direct_age_seconds: float | None = None
    current_semantic_support: bool = False
    relative_motion_error_budget_px: float | None = None
    absolute_anchor_timestamp: float | None = None
    anchor_age_seconds: float | None = None
    temporal_support: bool = False
    temporal_direction: str | None = None
    forward_backward_agreement_p90_m: float | None = None

class TeamBeliefEvidence(Closed):
    current_team: Literal['team_a','team_b'] | None = None
    support: float | None = Field(default=None,ge=0,le=1)
    source: Literal['current_observation','temporal_memory','unknown']
    current_observation: Literal['team_a','team_b'] | None = None
    current_quality: float | None = None
    current_strong: bool = False
    independent_sample: bool = False
    unknown_reason: str | None = None
    current_observation_reason: str | None = None
    evidence_a: float = 0
    evidence_b: float = 0
    independent_samples: int = 0
    uncertain_observations: int = 0
    candidate_team: str | None = None
    candidate_streak: int = 0
    last_strong_observation: float | None = None
    identity_resolved: bool = True
    history: list[dict] = Field(default_factory=list)
    latest_switch: dict | None = None

class PlayerObservation(Closed):
    actor_id: str | None = None
    team: Literal['left','right','team_a','team_b'] | None = None
    track_id: str | None = None
    camera: CameraProjectionEvidence | None = None
    projection_reason: str | None = None
    localization_kind: Literal['direct_calibrated','provided','camera_propagated','partial_constrained','temporal_predicted','temporal_corrected','bidirectional_interpolated','predicted','last_known','unknown'] = 'unknown'
    visibility: Literal['visible','temporarily_missing','off_camera','unknown'] = 'unknown'
    detection_confidence: float | None = Field(default=None,ge=0,le=1)
    team_confidence: float | None = Field(default=None,ge=0,le=1)
    team_evidence: TeamBeliefEvidence | None = None
    position_diagnostic: dict | None = None
    role: str | None = None
    pitch: tuple[float,float] | None = None
    image_bbox: tuple[float,float,float,float] | None = None
    image_foot: tuple[float,float] | None = None
    position: Provenance
    identity: Provenance

class BallObservation(Closed):
    pitch: tuple[float,float] | None = None
    image: tuple[float,float] | None = None
    provenance: Provenance

class PitchCalibrationObservation(Closed):
    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)
    landmarks: list[tuple[float,float]] = Field(default_factory=list)
    image_to_pitch: tuple[float,float,float,float,float,float,float,float,float] | None = None
    validity_region: list[tuple[float,float]] = Field(default_factory=list)
    provenance: Provenance
    @model_validator(mode='after')
    def interval(self):
        if self.end_frame < self.start_frame: raise ValueError('Reversed calibration interval')
        return self

class RecordingObservation(Closed):
    recording_id: str = Field(min_length=1)
    frame: int = Field(ge=0)
    timestamp: float = Field(ge=0)
    fps: float = Field(gt=0)
    source_offset_frames: int = Field(default=0,ge=0)
    half: Literal[1,2] | None = None
    coordinate_system: Literal["pitch_corner_metres"] = "pitch_corner_metres"
    players: list[PlayerObservation]
    ball: BallObservation
    calibration: PitchCalibrationObservation | None = None
    visible_pitch_region: list[tuple[float,float]] | None = None
    provenance: Provenance
    @model_validator(mode='after')
    def aligned(self):
        if abs(self.timestamp-self.frame/self.fps)>1e-6: raise ValueError('Frame/time mismatch')
        ids=[p.actor_id for p in self.players if p.actor_id is not None]
        if len(ids)!=len(set(ids)): raise ValueError('Duplicate actor IDs in frame')
        if self.calibration and not self.calibration.start_frame<=self.frame<=self.calibration.end_frame:
            raise ValueError('Calibration outside validity interval')
        return self
