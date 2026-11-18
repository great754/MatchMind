"""Bounded, finite debug records, kept separate from observed ball evidence."""
from typing import Literal
from pydantic import Field, model_validator
from src.query.schema import Closed

Point = tuple[float, float]


class MotionHypothesis(Closed):
    track_id: int
    mean: Point
    velocity: Point
    acceleration: Point
    covariance: tuple[Point, Point]
    radius90_px: float = Field(ge=0)
    motion_probabilities: dict[str, float]
    age_seconds: float = Field(ge=0)
    camera_status: str
    confirmed: bool

    @model_validator(mode='after')
    def valid_covariance(self):
        a, b = self.covariance
        if a[0] <= 0 or b[1] <= 0 or abs(a[1]-b[0]) > 1e-5 or a[0]*b[1]-a[1]*b[0] < -1e-5:
            raise ValueError('Prediction covariance must be symmetric positive semidefinite')
        if len(self.motion_probabilities) != 4 or any(not 0 <= v <= 1 for v in self.motion_probabilities.values()) or abs(sum(self.motion_probabilities.values())-1) > 1e-6:
            raise ValueError('Four normalized motion probabilities required')
        return self


class SearchRegion(Closed):
    track_id: int
    bbox: tuple[int, int, int, int]
    radius90_px: float = Field(ge=0)

    @model_validator(mode='after')
    def valid_region(self):
        if self.bbox[0] < 0 or self.bbox[1] < 0 or min(self.bbox[2:]) <= 0:
            raise ValueError('Positive source crop required')
        return self


class CandidateDecision(Closed):
    candidate_id: str
    decision: Literal['OBSERVED', 'COMPETING', 'REJECTED']
    reason: str
    visual_confidence: float = Field(ge=0, le=1)
    appearance_similarity: float | None = Field(default=None, ge=0, le=1)
    mahalanobis_squared: float | None = Field(default=None, ge=0)
    association_score: float | None = Field(default=None, ge=0, le=1)


class TrajectoryFrame(Closed):
    schema_version: Literal['ball-m20-debug-v1'] = 'ball-m20-debug-v1'
    frame: int = Field(ge=0)
    timestamp: float = Field(ge=0)
    camera_epoch: int = Field(ge=0)
    state: Literal['OBSERVED', 'PREDICTED', 'AMBIGUOUS', 'UNKNOWN']
    selected_candidate_id: str | None
    hypotheses: list[MotionHypothesis] = Field(max_length=4)
    search_regions: list[SearchRegion] = Field(max_length=3)
    candidates: list[CandidateDecision] = Field(max_length=160)
    global_inference: bool
    inference_seconds: float = Field(ge=0)
    crop_inference_seconds: float = Field(ge=0)
    tracker_seconds: float = Field(ge=0)
    events: list[str] = Field(max_length=10)
    prediction_only: bool
    tracking_confidence: float = Field(ge=0, le=1)
    independent_accuracy: Literal['pending_human_review'] = 'pending_human_review'

    @model_validator(mode='after')
    def visual_provenance(self):
        accepted = [c.candidate_id for c in self.candidates if c.decision == 'OBSERVED']
        if self.state == 'OBSERVED':
            if self.selected_candidate_id is None or accepted != [self.selected_candidate_id] or self.prediction_only:
                raise ValueError('Observed state needs exactly one selected visual candidate')
        elif accepted or self.selected_candidate_id is not None:
            raise ValueError('Prediction/ambiguity cannot contain an observed selection')
        if self.prediction_only != (self.state == 'PREDICTED'):
            raise ValueError('Prediction state must remain explicit')
        return self
