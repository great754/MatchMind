"""Explicit heuristic thresholds and deterministic attack-direction changes."""
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class TacticsConfig:
    sample_hz: float = 5.0
    min_outfield_players: int = 8
    block_low_max_m: float = 35.0
    block_high_min_m: float = 70.0
    block_persistence_seconds: float = 3.0
    pressure_near_radius_m: float = 3.0
    pressure_local_radius_m: float = 5.0
    pressure_outer_radius_m: float = 8.0
    pressure_min_defenders: int = 2
    pressure_min_observed_defenders: int = 8
    closing_min_mps: float = 0.5
    closing_max_mps: float = 14.0
    closing_lookback_seconds: float = 1.0
    press_persistence_seconds: float = 2.0
    ball_line_tolerance_m: float = 0.5
    shape_window_seconds: float = 10.0
    shape_window_min_fraction: float = 0.8
    width_change_m: float = 5.0
    compactness_change_m: float = 4.0
    shape_persistence_seconds: float = 3.0
    event_cooldown_seconds: float = 10.0
    comparison_min_samples: int = 50
    comparison_width_margin_m: float = 5.0
    formation_window_seconds: float = 15.0
    formation_update_seconds: float = 1.0
    formation_min_fraction: float = 0.8
    formation_min_line_gap_m: float = 8.0
    formation_max_line_depth_m: float = 12.0
    formation_max_rmse_m: float = 4.5
    formation_min_fit_margin_m: float = 1.0
    formation_min_membership_stability: float = 0.8
    formation_persistence_seconds: float = 5.0
    network_window_seconds: float = 60.0
    network_long_transfer_m: float = 25.0
    network_pattern_min_transfers: int = 3
    network_pattern_persistence_seconds: float = 3.0
    attack_direction_changes: tuple = ()

    def __post_init__(self):
        integer_fields=('min_outfield_players','pressure_min_defenders','pressure_min_observed_defenders','comparison_min_samples','network_pattern_min_transfers')
        for name,value in vars(self).items():
            if name=='attack_direction_changes': continue
            if not isinstance(value,(float,int)) or isinstance(value,bool) or not math.isfinite(value) or value<=0:
                raise ValueError(f'{name} must be finite and positive')
            if name in integer_fields and (not isinstance(value,int) or isinstance(value,bool)): raise ValueError(f'{name} must be an integer')
        if not 1<=self.min_outfield_players<=10 or self.pressure_min_observed_defenders>11: raise ValueError('Invalid player availability threshold')
        if not self.block_low_max_m<self.block_high_min_m<=105: raise ValueError('Block thresholds must be ordered within the pitch')
        if not self.pressure_near_radius_m<self.pressure_local_radius_m<self.pressure_outer_radius_m: raise ValueError('Pressure radii must increase')
        if self.closing_min_mps>=self.closing_max_mps: raise ValueError('Closing speed limits must increase')
        for name in ('shape_window_min_fraction','formation_min_fraction','formation_min_membership_stability'):
            if getattr(self,name)>1: raise ValueError(f'{name} must be <= 1')
        previous=-1
        for frame,direction in self.attack_direction_changes:
            if not isinstance(frame,int) or isinstance(frame,bool) or frame<=previous or direction not in (-1,1): raise ValueError('Direction changes need increasing nonnegative integer frames and directions ±1')
            previous=frame

    def stride(self,fps):
        ratio=fps/self.sample_hz
        if self.sample_hz>fps or not math.isclose(ratio,round(ratio),abs_tol=1e-8): raise ValueError('Tactical sample_hz must divide recording FPS')
        return int(round(ratio))

    def direction(self,frame,team,analytics_config):
        left=analytics_config.attack_direction('left')
        for change,direction in self.attack_direction_changes:
            if change>frame: break
            left=direction
        return left if team=='left' else -left
