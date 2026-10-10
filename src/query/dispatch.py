"""Provider-neutral JSON tool registry. Unknown tools/arguments are rejected."""
from typing import Literal
from pydantic import Field
from .schema import Closed

Team=Literal['left','right']
class Empty(Closed):pass
class Player(Closed):player_id:int=Field(gt=0,strict=True)
class TeamArgs(Closed):team:Team|None=None
class Limit(Closed):limit:int=Field(default=5,ge=1,le=100,strict=True)
class TeamLimit(Limit):team:Team|None=None
class Event(Closed):event_id:str=Field(min_length=1,max_length=100)
class EventWindow(Event):lookback_seconds:float=Field(default=15.,gt=0,le=60)
class Time(Closed):timestamp:float=Field(ge=0)
class Events(TeamArgs):
    event_type:str|None=None
    start_time:float|None=Field(default=None,ge=0)
    end_time:float|None=Field(default=None,ge=0)
class Network(TeamArgs):
    start_time:float|None=Field(default=None,ge=0)
    end_time:float|None=Field(default=None,ge=0)
class RequiredTeam(Closed):team:Team

REGISTRY={
 'get_match_summary':Empty,'get_player_summary':Player,'get_possession_summary':TeamArgs,
 'get_events':Events,'get_transfer':Event,'get_difficult_transfers':TeamLimit,
 'get_tactical_state':Time,'get_pre_event_evidence':Event,'get_pressure_sequences':TeamArgs,
 'get_formation_timeline':RequiredTeam,'get_transfer_network':Network,'get_fastest_players':Limit,
 'get_event_validation':Event,'get_event_sequence':EventWindow}


def tool_specifications():
    return [dict(name=name,input_schema=model.model_json_schema(),output_schema_title='QueryResult') for name,model in REGISTRY.items()]


def dispatch(tools,name,arguments):
    if name not in REGISTRY:raise ValueError('Unsupported MatchMind tool')
    parsed=REGISTRY[name].model_validate(arguments)
    return getattr(tools,name)(**parsed.model_dump())
