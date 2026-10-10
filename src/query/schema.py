"""Provider-independent closed scalar facts and answer plans; no executable queries."""
from typing import Literal
from pydantic import BaseModel,ConfigDict,Field

class Closed(BaseModel):
    model_config=ConfigDict(extra='forbid',frozen=True,allow_inf_nan=False)

class Fact(Closed):
    fact_id:str
    record_id:str
    field:str
    value:float|int|str|bool|None
    source:str
    timestamp:float|None=None
    event_id:str|None=None

class QueryResult(Closed):
    schema_version:Literal['1.0']='1.0'
    question_type:Literal['match_summary','player_summary','possession_summary','events','transfer','difficult_transfers','tactical_state','pre_event_evidence','pressure_sequences','formation_timeline','transfer_network','fastest_players','event_validation','unsupported']
    status:Literal['ok','unknown','unsupported']='ok'
    facts:list[Fact]=Field(default_factory=list)
    limitations:list[str]=Field(default_factory=list)
    provenance:list[str]=Field(default_factory=list)

class Statement(Closed):
    text:str
    fact_ids:list[str]
    timestamp:float|None=None
    event_id:str|None=None

class Answer(Closed):
    mode:Literal['beginner','fan','analyst']
    question_type:str
    status:Literal['ok','unknown','unsupported']
    statements:list[Statement]
    limitations:list[str]
    supporting_fact_ids:list[str]
    facts:list[Fact]=Field(default_factory=list)

class Selection(Closed):
    """Future providers may select supported statements, never author new facts."""
    statement_ids:list[int]
