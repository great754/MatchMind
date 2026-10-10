"""Closed tactical facts accepted by the existing commentary boundary."""
from typing import Literal
from pydantic import BaseModel,ConfigDict,Field

class ClosedModel(BaseModel):
    model_config=ConfigDict(extra="forbid",frozen=True,allow_inf_nan=False)

class Shape(ClosedModel):
    available_players: int = Field(ge=0)
    goalkeeper_excluded: bool
    goalkeeper_source: Literal['provider_metadata','explicit','unavailable']
    reason: Literal['Goalkeeper identity unavailable; outfield shape withheld.','Insufficient or ambiguous active outfield roster.'] | None = None
    centroid_x_m: float | None = None
    centroid_y_m: float | None = None
    width_m: float | None = Field(default=None,ge=0)
    depth_m: float | None = Field(default=None,ge=0)
    compactness_m: float | None = Field(default=None,ge=0)
    spread_m: float | None = Field(default=None,ge=0)
    longitudinal_m: float | None = None
    lateral_m: float | None = None

class Formation(ClosedModel):
    estimate: Literal['4-3-3','4-4-2','4-2-3-1','3-5-2','other','unknown'] = 'unknown'
    reason: Literal['insufficient_history','goalkeeper_unknown','incomplete_roster','insufficient_coverage','ambiguous_fit','poor_fit','unstable_membership','direction_changed','pending_persistence'] | None = None
    window_start_frame: int | None = None
    window_end_frame: int | None = None
    samples: int = 0
    fit_rmse_m: float | None = None
    fit_margin_m: float | None = None
    membership_stability: float | None = Field(default=None,ge=0,le=1)
    fit_quality: Literal['clear','unavailable'] = 'unavailable'

class Pressure(ClosedModel):
    carrier_id: int | None = None
    carrier_actor_id: int | None = None
    attacking_team: Literal['left','right'] | None = None
    defending_team: Literal['left','right'] | None = None
    available_defenders: int = 0
    expected_defenders: int = 0
    near_radius_m: float = 3.0
    local_radius_m: float = 5.0
    outer_radius_m: float = 8.0
    nearest_defender_distance_m: float | None = None
    defenders_within_near: int | None = None
    defenders_within_local: int | None = None
    defenders_within_outer: int | None = None
    closing_defenders: int | None = None
    peak_closing_speed_mps: float | None = None
    teammates_within_local_including_carrier: int | None = None
    local_numerical_balance: int | None = None
    local_pressure: Literal['unavailable','low','nearby','multiple_nearby'] = 'unavailable'
    press_candidate: bool = False
    sustained_press: bool = False
    evidence_duration_s: float = 0
    reason: Literal['no_confirmed_carrier','missing_carrier_or_ball','incomplete_defenders','closing_unavailable'] | None = None

class BallRelative(ClosedModel):
    attacking_ahead: int | None = None
    attacking_behind: int | None = None
    attacking_level: int | None = None
    defenders_goal_side: int | None = None
    defenders_behind_ball_to_own_goal: int | None = None
    attacking_available: int = 0
    defending_available: int = 0
    goalkeeper_excluded: bool = True

class TeamState(ClosedModel):
    team: Literal['left','right']
    attack_direction: Literal[-1,1]
    phase: Literal['possessing','defending','unknown']
    shape: Shape
    block_candidate: Literal['low','mid','high','unknown']
    block: Literal['low','mid','high','unknown']
    block_evidence_s: float = 0
    formation: Formation

class TacticalState(ClosedModel):
    schema_version: Literal['1.0'] = '1.0'
    frame: int = Field(ge=0)
    timestamp: float = Field(ge=0)
    source: Literal['tactical_state_engine'] = 'tactical_state_engine'
    position_source: Literal['gsr','pitch_transform','unavailable']
    identity_source: Literal['mot_to_gsr_mapping','gsr_actor']
    possession_state: Literal['controlled','candidate','free','contested','unknown','out_of_play']
    possession_confidence: float | None = None
    left: TeamState
    right: TeamState
    centroid_distance_m: float | None = None
    pressure: Pressure
    ball_relative: BallRelative
