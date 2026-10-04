from enum import Enum
from typing import Optional

from pydantic import BaseModel


class EventType(str, Enum):
    PASS = "pass"
    SHOT = "shot"
    GOAL = "goal"
    FOUL = "foul"
    TACKLE = "tackle"
    INTERCEPTION = "interception"
    CORNER = "corner"
    OFFSIDE = "offside"
    UNKNOWN = "unknown"


class MatchEvent(BaseModel):
    match_id: str

    timestamp: float
    period: int

    event_type: EventType

    team_id: Optional[str] = None
    player_id: Optional[str] = None
    player_name: Optional[str] = None

    x: Optional[float] = None
    y: Optional[float] = None

    outcome: Optional[str] = None


class PlayerState(BaseModel):
    timestamp: float

    player_id: str
    team_id: str

    x: float
    y: float


class BallState(BaseModel):
    timestamp: float

    x: float
    y: float

    z: Optional[float] = None