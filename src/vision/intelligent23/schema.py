"""Closed M23 provenance: search hypotheses and inferred points are never observations."""
import math
from typing import Literal
import numpy as np
from pydantic import Field, model_validator
from src.query.schema import Closed


class SupportingDetection(Closed):
    candidate_id: str
    bbox: tuple[float, float, float, float]
    confidence: float = Field(ge=.35, le=1)
    current_only_confidence: float = Field(ge=.35, le=1)


class MemoryEvidence(Closed):
    track_id: int
    mahalanobis_squared: float = Field(ge=0)
    appearance_similarity: float | None = Field(default=None, ge=0, le=1)
    size_ratio: float = Field(gt=0)
    velocity: tuple[float, float]
    joint_support: float = Field(ge=0, le=1)
    stationary_frames: int = Field(ge=0)
    confirmed_history: bool = True


class Candidate(Closed):
    candidate_id: str
    bbox: tuple[float, float, float, float]
    confidence: float = Field(ge=0, le=1)
    inference_view: str
    source: Literal['VIDEO_CV_DETECTOR'] = 'VIDEO_CV_DETECTOR'
    origin: Literal['primary', 'guided', 'discovery', 'complement']
    visual_evidence: Literal['CURRENT_PIXELS']
    source_region: tuple[int, int, int, int] | None = None
    source_model_sha256: str | None = None
    current_only_confidence: float | None = Field(default=None, ge=0, le=1)
    temporal_features: dict[str, float | bool | None]
    deterministic: dict
    support_views: list[dict]
    observation_eligible: bool
    decision_reason: str
    joint_memory_evidence: MemoryEvidence | None
    blur_supported: bool
    blur_definition: str
    m22_support: list[SupportingDetection]
    association_adjustment: float = Field(ge=-3, le=.5)
    detector_confidence_unchanged: Literal[True]
    confidence_is_calibrated_probability: Literal[False]
    selected: bool

    @model_validator(mode='after')
    def current_pixels(self):
        x, y, w, h = self.bbox
        if min(x, y) < 0 or min(w, h) <= 0:
            raise ValueError('Positive source-pixel candidate required')
        if self.observation_eligible and self.confidence < .35:
            raise ValueError('Prediction cannot promote a below-threshold observation')
        if self.selected and not self.observation_eligible:
            raise ValueError('Rejected visual candidate cannot be selected')
        if self.origin == 'complement' and self.observation_eligible and (self.current_only_confidence is None or self.current_only_confidence < .35):
            raise ValueError('Complementary observation requires the separate current-pixel check')
        return self


class SearchRegion(Closed):
    bbox: tuple[int, int, int, int]
    track_id: int
    radius90_px: float = Field(ge=0)
    planned_before_current_candidates: Literal[True]
    kind: Literal['trajectory', 'tentative_visual', 'global_discovery', 'motion_mode'] = 'trajectory'
    motion_mode: str | None = None
    mode_probability: float | None = Field(default=None, ge=0, le=1)


class Hypothesis(Closed):
    mean: tuple[float, float]
    velocity: tuple[float, float]
    acceleration: tuple[float, float]
    covariance: tuple[tuple[float, float], tuple[float, float]]
    radius90_px: float = Field(ge=0)
    motion_probabilities: dict[str, float]
    age_seconds: float = Field(ge=0)
    camera_status: str
    track_id: int
    confirmed: bool
    archived: bool = False
    last_observed_frame: int | None = None
    appearance_template: list[float] | None = None

    @model_validator(mode='after')
    def uncertainty(self):
        matrix = np.asarray(self.covariance)
        if not np.allclose(matrix, matrix.T, atol=1e-6) or np.linalg.eigvalsh(matrix).min() <= 0:
            raise ValueError('Positive symmetric image uncertainty required')
        if any(not 0 <= v <= 1 for v in self.motion_probabilities.values()) or abs(sum(self.motion_probabilities.values())-1) > 1e-6:
            raise ValueError('Normalized motion mixture required')
        if self.appearance_template is not None and (len(self.appearance_template) != 32 or any(not 0 <= v <= 1 for v in self.appearance_template)):
            raise ValueError('Invalid appearance evidence')
        return self


class DirectMemory(Closed):
    position: tuple[float, float]
    frame: int = Field(ge=0)
    timestamp: float = Field(ge=0)
    bbox: tuple[float, float, float, float]
    trajectory_hypothesis: int


class Memory(Closed):
    state: Literal['OBSERVED', 'PREDICTED', 'AMBIGUOUS', 'UNKNOWN']
    last_direct_observation: DirectMemory | None
    missed_frames: int | None = Field(default=None, ge=0)
    missed_seconds: float | None = Field(default=None, ge=0)
    hypotheses: list[Hypothesis]
    search_regions: list[SearchRegion]
    active_trajectory_hypothesis: int | None
    camera_motion_source: str
    camera_homography: list[list[float]] | None
    camera_cut: bool
    reacquired_with_current_visual_evidence: bool
    predicted_image_position: tuple[float, float] | None
    reason: str
    appearance_is_calibrated: Literal[False]
    search_memory_is_ball_identity_truth: Literal[False]


class Reconstruction(Closed):
    image: tuple[float, float]
    directly_observed: Literal[False]
    support_frames: tuple[int, int]
    uses_future: Literal[True]
    reason: str


class HistoryPoint(Closed):
    frame: int = Field(ge=0)
    source_image: tuple[float, float]
    current_camera_image: tuple[float, float]
    directly_observed_at_source_frame: Literal[True]
    current_observation: Literal[False]


class IntelligentFrame(Closed):
    schema_version: Literal['cv-m23-frame-v1']
    recording_id: str
    variant: Literal['m21_temporal_offline', 'guided_search', 'guided_verified', 'full_m23']
    frame: int = Field(ge=0)
    timestamp: float = Field(ge=0)
    camera_epoch: int = Field(ge=0)
    state: Literal['OBSERVED', 'PREDICTED', 'AMBIGUOUS', 'UNKNOWN']
    directly_observed: bool
    selected_candidate_id: str | None
    candidates: list[Candidate]
    memory: Memory | None
    reconstruction: Reconstruction | None
    trajectory_history: list[HistoryPoint]
    interaction: dict
    source_pixels_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    model_set_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    config_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    uses_future_for_association: Literal[True]
    future_visual_frames_used: Literal[False]
    m22_context_frames: list[int]
    reasons: list[str]

    @model_validator(mode='after')
    def provenance(self):
        if len({c.candidate_id for c in self.candidates}) != len(self.candidates):
            raise ValueError('Duplicate source candidates')
        selected = [c for c in self.candidates if c.selected]
        if len(selected) != int(self.directly_observed) or (selected[0].candidate_id if selected else None) != self.selected_candidate_id:
            raise ValueError('One actual selected candidate required for an observation')
        if (self.state == 'OBSERVED') != self.directly_observed:
            raise ValueError('Observed status requires current pixels')
        if self.directly_observed and self.reconstruction is not None:
            raise ValueError('Reconstructed points are not observed')
        if self.reconstruction is not None and not 0 <= self.reconstruction.support_frames[0] < self.frame < self.reconstruction.support_frames[1]:
            raise ValueError('Two-sided reconstruction requires real bracketing support')
        if self.m22_context_frames and (self.m22_context_frames[-1] != self.frame or sorted(self.m22_context_frames) != self.m22_context_frames or min(self.m22_context_frames) < 0):
            raise ValueError('Temporal visual inference must be causal')
        if any(p.frame >= self.frame for p in self.trajectory_history):
            raise ValueError('History cannot masquerade as a current/future observation')
        if self.memory and self.memory.last_direct_observation and self.memory.last_direct_observation.frame > self.frame:
            raise ValueError('Search memory cannot know a future observation')
        return self
