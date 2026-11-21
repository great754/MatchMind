from typing import Literal
from uuid import UUID
from pydantic import Field,model_validator
from src.query.schema import Closed

class AssistantRequest(Closed):
    question:str=Field(min_length=1,max_length=1500)
    session_id:UUID
    previous_turn:UUID|None=None
    selected_player:str|None=Field(default=None,max_length=180)
    cv_pipeline:Literal['m14','m19','m20','m21','m22','m23']='m14'
    cv_experiment:str|None=Field(default=None,max_length=60,pattern=r'^[a-z0-9_]+$')
    provider:Literal['auto','local','gemini','microsoft']='auto'

    @model_validator(mode='after')
    def experiment_scope(self):
        if self.cv_experiment is not None and self.cv_pipeline!='m21':raise ValueError('Experiment selection requires M21')
        return self

class AssistantPlan(Closed):
    clarification:str|None=Field(default=None,max_length=240)
    operation:Literal['answer','actions','statistic','compare','pass_rank','sprint','explain','identity','followup','recap','player_profile','score','unsupported']='answer'
    actions:list[Literal['goal','shot','pass','reception','cross','tackle','interception','recovery','carry','header','sprint']]=Field(default_factory=list,max_length=10)
    players:list[str]=Field(default_factory=list,max_length=2)
    metric:Literal['distance','difficulty','speed','shots','passes','tackles','possession']|None=None
    video:bool=False
    overlay:bool=False
    review:bool=False
    explain:bool=False
    team:Literal['left','right']|None=None
    ordinal:int|None=Field(default=None,ge=1,le=100)
    start:float|None=Field(default=None,ge=0,le=14400)
    end:float|None=Field(default=None,ge=0,le=14400)
    duration:float|None=Field(default=None,ge=5,le=120)
    buildup:Literal['normal','more','short']='normal'
    reference:Literal['none','player','event','result']='none'
    period:int|None=Field(default=None,ge=1,le=4)

    @model_validator(mode='after')
    def valid_window(self):
        if self.start is not None and self.end is not None and self.end < self.start:
            raise ValueError('End must follow start')
        return self

class IdentityConfirmation(Closed):
    session_id:UUID
    player_ref:str=Field(min_length=1,max_length=180)
    jersey:int|None=Field(default=None,ge=0,le=999)
    name:str=Field(default='',max_length=80)
    team:Literal['left','right','unknown']='unknown'
    role:Literal['player','goalkeeper','official']='player'
    confirmed:Literal[True]
    image_id:str|None=Field(default=None,max_length=80)

class Metric(Closed):
    name:str
    value:float|None
    unit:str
    definition:str
    source:str
    support:Literal['recorded','estimated','unavailable']
    samples:int=Field(ge=0)
    window:tuple[float,float]
    player_ref:str|None=None
    limitations:list[str]=Field(default_factory=list)

class TeamLabel(Closed):
    name:str=Field(min_length=1,max_length=60)
    color:str=Field(pattern=r'^#[0-9a-fA-F]{6}$')
class TeamLabelsRequest(Closed):
    session_id:UUID
    left:TeamLabel
    right:TeamLabel
