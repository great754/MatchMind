from dataclasses import dataclass
import math
from numbers import Integral


@dataclass(frozen=True)
class AnalyticsConfig:
    fps: float = 25.0
    player_smoothing_frames: int = 15
    ball_smoothing_frames: int = 11
    median_frames: int = 3
    player_speed_limit_mps: float = 14.0
    ball_speed_limit_mps: float = 50.0
    acquisition_radius_m: float = 2.5
    retention_radius_m: float = 3.5
    ambiguity_margin_m: float = 0.5
    control_ball_speed_mps: float = 8.0
    control_relative_speed_mps: float = 4.0
    possession_frames: int = 5
    pass_min_flight_frames: int = 3
    pass_min_distance_m: float = 3.0
    pass_min_speed_mps: float = 3.0
    pass_timeout_seconds: float = 8.0
    defender_radius_m: float = 5.0
    lane_radius_m: float = 2.0
    shot_min_speed_mps: float = 8.0
    shot_max_goal_distance_m: float = 35.0
    shot_goal_margin_m: float = 3.0
    shot_max_time_to_goal_seconds: float = 3.0
    shot_persistence_frames: int = 3
    shot_window_seconds: float = 1.2
    validation_tolerance_seconds: float = 1.0
    left_attack_direction: int = 1

    def __post_init__(self):
        for name in ('fps', 'player_speed_limit_mps', 'ball_speed_limit_mps',
                     'acquisition_radius_m', 'retention_radius_m', 'control_ball_speed_mps',
                     'control_relative_speed_mps', 'pass_min_distance_m', 'pass_min_speed_mps',
                     'pass_timeout_seconds', 'defender_radius_m', 'lane_radius_m',
                     'shot_min_speed_mps', 'shot_max_goal_distance_m',
                     'shot_max_time_to_goal_seconds', 'shot_window_seconds',
                     'validation_tolerance_seconds'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive')
        for name in ('player_smoothing_frames', 'ball_smoothing_frames', 'median_frames'):
            value = getattr(self, name)
            if not isinstance(value, Integral) or isinstance(value, bool) or value < 3 or value % 2 == 0:
                raise ValueError(f'{name} must be an odd integer >= 3')
        for name in ('possession_frames', 'pass_min_flight_frames', 'shot_persistence_frames'):
            if not isinstance(getattr(self, name), Integral) or isinstance(getattr(self, name), bool) or getattr(self, name) < 1:
                raise ValueError(f'{name} must be >= 1')
        if self.retention_radius_m < self.acquisition_radius_m:
            raise ValueError('retention radius must be >= acquisition radius')
        if not math.isfinite(self.ambiguity_margin_m) or not math.isfinite(self.shot_goal_margin_m) or self.ambiguity_margin_m < 0 or self.shot_goal_margin_m < 0:
            raise ValueError('distance margins must be nonnegative')
        if self.left_attack_direction not in (-1, 1):
            raise ValueError('left_attack_direction must be -1 or 1')

    def attack_direction(self, team):
        if team not in ('left', 'right'):
            raise ValueError(f'Unknown team side: {team}')
        return self.left_attack_direction if team == 'left' else -self.left_attack_direction
