"""Independent thresholds for insight selection; analytics definitions stay unchanged."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class IntelligenceConfig:
    player_speed_kmh: float = 25.0
    ball_speed_kmh: float = 80.0
    threshold_persistence_frames: int = 3
    threshold_cooldown_seconds: float = 15.0
    distance_milestone_m: float = 100.0
    long_pass_m: float = 25.0
    difficult_pass_score: float = 50.0
    progressive_pass_m: float = 15.0
    significant_progression_m: float = 20.0
    context_snapshot_seconds: float = 5.0
    significance_threshold: float = 40.0
    grouping_seconds: float = 2.0
    commentary_cooldown_seconds: float = 8.0
    max_commentary_moments: int = 20
    max_provider_calls: int = 60
    start_seconds: float = 0.0
    end_seconds: float | None = None
    context_window_seconds: int = 30

    def __post_init__(self):
        positive = ('player_speed_kmh','ball_speed_kmh','distance_milestone_m','long_pass_m',
                    'progressive_pass_m','significant_progression_m','context_snapshot_seconds')
        nonnegative = ('threshold_cooldown_seconds','grouping_seconds','commentary_cooldown_seconds','start_seconds')
        for field in positive + nonnegative:
            value = getattr(self,field)
            if not math.isfinite(value) or value < 0 or (field in positive and value == 0):
                raise ValueError(f'{field} has an invalid threshold')
        if not 0 <= self.significance_threshold <= 100 or not 0 <= self.difficult_pass_score <= 100:
            raise ValueError('Significance and difficulty thresholds must be within 0–100')
        for field in ('threshold_persistence_frames','max_commentary_moments','max_provider_calls'):
            value = getattr(self,field)
            if not isinstance(value,int) or isinstance(value,bool) or value < (1 if field=='threshold_persistence_frames' else 0):
                raise ValueError(f'{field} must be a valid nonnegative integer')
        if self.end_seconds is not None and (not math.isfinite(self.end_seconds) or self.end_seconds < self.start_seconds):
            raise ValueError('end_seconds must be >= start_seconds')
        if self.context_window_seconds not in (15,30,60):
            raise ValueError('context window must be 15, 30 or 60 seconds')
