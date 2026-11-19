"""Closed M21 diagnostic contract; inference is kept separate from direct observations."""
from typing import Literal
from pydantic import Field, model_validator
from src.query.schema import Closed


class CandidateDecision(Closed):
    candidate_id: str
    bbox: tuple[float,float,float,float]
    visual_confidence: float = Field(ge=0,le=1)
    inference_view: str
    selected: bool
    deterministic: dict
    temporal_score: float | None = Field(default=None,ge=0,le=1)
    temporal_features: dict
    cross_detector: dict | None = None
    score_calibrated: Literal[False]
    decision_reason: str = 'UNSPECIFIED_RESEARCH_DECISION'


class Reconstruction(Closed):
    image: tuple[float,float]
    directly_observed: Literal[False]
    support_frames: tuple[int,int]
    uses_future: Literal[True]
    reason: str


class AdvancedFrame(Closed):
    schema_version: Literal['cv-m21-debug-v1']
    variant: str
    frame: int = Field(ge=0)
    timestamp: float = Field(ge=0)
    camera_epoch: int = Field(ge=0)
    association_mode: Literal['offline','causal_ball_only']
    uses_future_for_ball_association: bool
    event_analysis_mode: Literal['offline_with_original_M9_participants']
    selected_candidate_id: str | None
    directly_observed: bool
    candidates: list[CandidateDecision]
    reconstruction: Reconstruction | None
    camera_refinement: dict
    player_tracks: list[dict]
    manual_identity_propagated: Literal[False]

    @model_validator(mode='after')
    def independent_states(self):
        selected = [c for c in self.candidates if c.selected]
        if len({c.candidate_id for c in self.candidates}) != len(self.candidates):
            raise ValueError('Duplicate M21 candidate decisions')
        if self.directly_observed:
            if len(selected) != 1 or selected[0].candidate_id != self.selected_candidate_id:
                raise ValueError('Observed M21 point needs its actual selected visual candidate')
        elif selected or self.selected_candidate_id is not None:
            raise ValueError('Inferred M21 position cannot select an observation')
        if self.association_mode == 'causal_ball_only' and (self.uses_future_for_ball_association or self.reconstruction is not None):
            raise ValueError('Causal ball mode cannot contain future reconstruction')
        if self.reconstruction is not None and self.directly_observed:
            raise ValueError('Reconstruction is never direct observation')
        if any(p.get('manual_identity_propagated') is not False for p in self.player_tracks):
            raise ValueError('Alternative tracking cannot inherit manual confirmations')
        return self
