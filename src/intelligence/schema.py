"""Versioned, closed schemas: arbitrary logs, names and statistics cannot enter events."""
from enum import Enum
import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = '1.0'


class EventType(str,Enum):
    POSSESSION_GAINED='POSSESSION_GAINED'
    POSSESSION_LOST='POSSESSION_LOST'
    POSSESSION_CHANGED='POSSESSION_CHANGED'
    PASS_COMPLETED='PASS_COMPLETED'
    TRANSFER_INTERCEPTED='TRANSFER_INTERCEPTED'
    TRANSFER_UNRESOLVED='TRANSFER_UNRESOLVED'
    TRANSFER_OUT_OF_PLAY='TRANSFER_OUT_OF_PLAY'
    LONG_PASS='LONG_PASS'
    DIFFICULT_PASS='DIFFICULT_PASS'
    PROGRESSIVE_PASS='PROGRESSIVE_PASS'
    BALL_PROGRESSION='BALL_PROGRESSION'
    SHOT_CANDIDATE='SHOT_CANDIDATE'
    SHOT_VALIDATED='SHOT_VALIDATED'
    GOAL='GOAL'
    ACTION_VALIDATED='ACTION_VALIDATED'
    PLAYER_SPEED_THRESHOLD='PLAYER_SPEED_THRESHOLD'
    BALL_SPEED_THRESHOLD='BALL_SPEED_THRESHOLD'
    PLAYER_DISTANCE_MILESTONE='PLAYER_DISTANCE_MILESTONE'
    TACTICAL_BLOCK_CHANGED='TACTICAL_BLOCK_CHANGED'
    TEAM_WIDTH_INCREASED='TEAM_WIDTH_INCREASED'
    TEAM_COMPACTNESS_CHANGED='TEAM_COMPACTNESS_CHANGED'
    HIGH_PRESSURE_SEQUENCE='HIGH_PRESSURE_SEQUENCE'
    TRANSFER_NETWORK_PATTERN='TRANSFER_NETWORK_PATTERN'
    FORMATION_ESTIMATE_CHANGED='FORMATION_ESTIMATE_CHANGED'


class ClosedModel(BaseModel):
    model_config=ConfigDict(extra='forbid',frozen=True,allow_inf_nan=False)


class EventMetrics(ClosedModel):
    start_frame: int | None = None
    end_frame: int | None = None
    start_x_m: float | None = None
    start_y_m: float | None = None
    end_x_m: float | None = None
    end_y_m: float | None = None
    distance_m: float | None = None
    duration_s: float | None = None
    ball_speed_kmh: float | None = None
    ball_speed_median_kmh: float | None = None
    speed_kmh: float | None = None
    threshold_kmh: float | None = None
    measured_distance_m: float | None = None
    milestone_m: float | None = None
    difficulty_score_0_100: float | None = Field(default=None,ge=0,le=100)
    forward_progression_m: float | None = None
    observed_step_fraction: float | None = Field(default=None,ge=0,le=1)
    window_coverage_fraction: float | None = Field(default=None,ge=0,le=1)
    outcome: Literal['completed','intercepted','unresolved','out_of_play'] | None = None
    possession_state: Literal['controlled','candidate','free','contested','unknown','out_of_play'] | None = None
    previous_player_id: int | None = None
    previous_team: Literal['left','right'] | None = None
    action_label: Literal['PASS','DRIVE','HEADER','HIGH PASS','OUT','CROSS','THROW IN','SHOT',
                          'BALL PLAYER BLOCK','PLAYER SUCCESSFUL TACKLE','FREE KICK','GOAL'] | None = None


class TacticalMetrics(ClosedModel):
    evidence_start_frame: int | None = None
    duration_s: float | None = Field(default=None,ge=0)
    previous_block: Literal['low','mid','high','unknown'] | None = None
    block: Literal['low','mid','high','unknown'] | None = None
    width_m: float | None = None
    previous_width_m: float | None = None
    compactness_m: float | None = None
    previous_compactness_m: float | None = None
    nearest_defender_distance_m: float | None = None
    defenders_within_local: int | None = None
    local_radius_m: float | None = None
    closing_defenders: int | None = None
    formation: Literal['4-3-3','4-4-2','4-2-3-1','3-5-2','other','unknown'] | None = None
    formation_fit_rmse_m: float | None = None
    formation_window_start_frame: int | None = None
    formation_window_end_frame: int | None = None
    formation_samples: int | None = None
    transfer_count: int | None = None
    network_window_seconds: float | None = None


class CanonicalEvent(ClosedModel):
    schema_version: Literal['1.0'] = SCHEMA_VERSION
    event_id: str
    group_id: str
    timestamp: float = Field(ge=0)
    frame: int = Field(ge=0)
    type: EventType
    team: Literal['left','right'] | None = None
    player_id: int | None = None
    actor_id: int | None = None
    receiver_id: int | None = None
    receiver_actor_id: int | None = None
    id_namespace: Literal['mot','gsr_actor'] = 'mot'
    source: Literal['possession','transfer','trajectory','bas','motion','tactical_state_engine']
    observation_kind: Literal['observable_transfer','bas_action','inferred_control','estimated_motion','trajectory_candidate','tactical_state'] | None = None
    validation_state: Literal['estimated','bas_reference','bas_time_actor_match','bas_time_match'] = 'estimated'
    validated_by_bas: bool = False
    bas_annotation_id: int | None = None
    significance: float = Field(ge=0,le=100)
    metrics: EventMetrics = Field(default_factory=EventMetrics)
    tactical: TacticalMetrics | None = None


def make_event(match_key: str, *, frame: int, fps: float, type: EventType,
               group_id: str, source: str, significance: float, **values) -> CanonicalEvent:
    """Content-derived IDs remain stable across identical analytics runs."""
    fields=dict(frame=int(frame),timestamp=int(frame)/fps,type=type,group_id=group_id,
                source=source,significance=significance,**values)
    normalized=CanonicalEvent(event_id='pending',**fields).model_dump(mode='json')
    normalized.pop('event_id')
    digest=hashlib.sha256(json.dumps([match_key,normalized],sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()[:20]
    return CanonicalEvent(event_id='evt_'+digest,**fields)

class PossessionFacts(ClosedModel):
    player_id: int | None
    team: Literal['left','right'] | None
    state: Literal['controlled','candidate','free','contested','unknown','out_of_play']
    streak_seconds: float = Field(ge=0)
    confidence: float | None = Field(default=None,ge=0,le=1)

class FastestPlayer(ClosedModel):
    player_id: int
    team: Literal['left','right']
    peak_speed_kmh: float = Field(ge=0)
    valid_frames: int = Field(ge=1)

from src.tactics.schema import TacticalState

class ContextFacts(ClosedModel):
    schema_version: Literal['1.0']
    timestamp: float = Field(ge=0)
    frame: int = Field(ge=0)
    window_seconds: Literal[15,30,60]
    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)
    sampled_seconds: float = Field(gt=0)
    frames: int = Field(gt=0)
    possession_percent: dict[Literal['left','right'],float]
    unknown_percent: float = Field(ge=0,le=100)
    current_possession: PossessionFacts
    completed_passes: dict[Literal['left','right'],int]
    interceptions: dict[Literal['left','right'],int]
    unsuccessful_transfers: dict[Literal['left','right'],int]
    recent_pass_sequence: list[int | None]
    sequence_team: Literal['left','right'] | None
    transfer_count: int = Field(ge=0)
    recent_transfers: list[CanonicalEvent]
    fastest_players: list[FastestPlayer]
    ball_displacement_m: float | None = Field(default=None,ge=0)
    completed_pass_progression_m: dict[Literal['left','right'],float]
    validated_actions: list[CanonicalEvent]
    recent_shot_candidates: list[CanonicalEvent]
    notable_events: list[CanonicalEvent]
    notable_event_ids: list[str]
    tactical_state: TacticalState | None = None
    recent_tactical_events: list[CanonicalEvent] = Field(default_factory=list)
    limitations: list[Literal['Offline centered smoothing uses later samples; windows exclude later event outcomes.',
                'Ball movement and speed are interpolated 2D estimates; height and strike speed are unavailable.',
                'Possession and passes are heuristic; BAS agreement is not independent validation.']]
