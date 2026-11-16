from typing import Literal
from uuid import UUID
from pydantic import Field, model_validator
from src.query.schema import Closed

class HighlightQueryPlan(Closed):
    intent:Literal['create','identity','unsupported']='create'
    event_types:list[Literal['goal','shot','pass','cross','tackle','save','possession_change','sprint','reception','interception','carry','header']]=Field(default_factory=list,max_length=7)
    selection:Literal['all','maximum','ordinal','recap']='all'
    metric:Literal['distance_m','shot_speed','chance_quality']|None=None
    ordinal:int|None=Field(default=None,ge=1,le=100)
    team:Literal['left','right','home','away']|None=None
    player:int|None=Field(default=None,ge=1,le=999999999)
    player_namespace:Literal['tracker','actor','jersey']='jersey'
    start:float|None=Field(default=None,ge=0,le=14400)
    end:float|None=Field(default=None,ge=0,le=14400)
    duration:float|None=Field(default=None,ge=5,le=120)
    buildup:Literal['normal','more','short']='normal'
    review:bool=False
    @model_validator(mode='after')
    def consistent(self):
        if self.selection=='ordinal' and self.ordinal is None:raise ValueError('An ordinal is required')
        if self.selection=='maximum' and self.metric is None:raise ValueError('A metric is required')
        if self.start is not None and self.end is not None and self.start>=self.end:raise ValueError('Invalid time range')
        return self

class HighlightRequest(Closed):
    question:str=Field(min_length=1,max_length=1000)
    session_id:UUID
    previous_job:UUID|None=None
    provider:Literal['auto','local','gemini','microsoft']='auto'

class VideoTimebase(Closed):
    fps:float=Field(gt=0,le=240)
    frames:int=Field(ge=1)
    source_offset_frames:int=Field(default=0,ge=0)
    period:int|None=Field(default=1,ge=1,le=4)
    period_start_match_seconds:float|None=Field(default=None,ge=0)
    alignment_sources:list[dict]=Field(default_factory=list)
    def clip_frame(self,frame):return frame/self.fps
    def period_seconds(self,clip_seconds):return clip_seconds+self.source_offset_frames/self.fps
    def match_seconds(self,clip_seconds):
        return None if self.period_start_match_seconds is None else self.period_start_match_seconds+self.period_seconds(clip_seconds)
    def period_to_clip(self,seconds):return seconds-self.source_offset_frames/self.fps
    def match_to_clip(self,seconds):
        if self.period_start_match_seconds is None:raise ValueError('Period offset is not verified')
        return self.period_to_clip(seconds-self.period_start_match_seconds)

class HighlightEvent(Closed):
    event_id:str=Field(min_length=1,max_length=100)
    recording_id:str
    group_id:str
    event_type:Literal['goal','shot','pass','cross','tackle','save','possession_change','sprint','reception','interception','carry','header']
    start:float=Field(ge=0)
    end:float=Field(ge=0)
    timestamp:float=Field(ge=0)
    timestamp_confidence:str
    team:Literal['left','right']|None
    player:int|None
    actor_id:int|None
    player_ref:str|None=None
    receiver_ref:str|None=None
    related_players:list[int]
    related_actor_ids:list[int]=Field(default_factory=list)
    metrics:dict[str,float]
    evidence_source:str
    verification:Literal['annotated','inferred','candidate','unsupported']
    safe_for_automatic:bool
    validation_state:str
    annotation_id:int|None
    source_event_type:str
    source_video:str
    camera:str
    timebase:VideoTimebase
    period_relative_seconds:float
    match_relative_seconds:float|None
    related_event_ids:list[str]
    source_event_ids:list[str]=Field(default_factory=list)
    possession_sequence:str|None
    @model_validator(mode='after')
    def ordered(self):
        if self.start>self.end:raise ValueError('Event interval is reversed')
        return self

class ClipSegment(Closed):
    start:float=Field(ge=0)
    end:float=Field(gt=0)
    event_ids:list[str]=Field(min_length=1)
    event_start:float=Field(ge=0)
    event_end:float=Field(ge=0)
    period:int|None
    reason:str
    output_start:float=Field(ge=0)
    output_end:float=Field(gt=0)
    @model_validator(mode='after')
    def contains_event(self):
        if not self.start<=self.event_start<=self.event_end<=self.end or self.output_start>=self.output_end:raise ValueError('Clip omits its event')
        return self

class HighlightManifest(Closed):
    schema_version:Literal['highlight-m17-v1']='highlight-m17-v1'
    recording_id:str
    source_sha256:str|None=Field(pattern=r'^[a-f0-9]{64}$')
    source_identity_sha256:str|None=Field(default=None,pattern=r'^[a-f0-9]{64}$')
    source:dict
    index_version:str
    index_hash:str=Field(pattern=r'^[a-f0-9]{64}$')
    plan:HighlightQueryPlan
    render_settings:dict
    events:list[HighlightEvent]
    segments:list[ClipSegment]
    omitted_event_ids:list[str]=Field(default_factory=list)
    render:dict
    visual_annotation:dict|None=None
    ranking:str
    buildup:str
    timebase:VideoTimebase
    @model_validator(mode='after')
    def evidence_links(self):
        if self.source_sha256 is None and not (self.source.get('origin')=='registered_azure_stream' and self.source.get('etag') and self.source_identity_sha256 and self.source.get('identity_sha256')==self.source_identity_sha256):raise ValueError('Missing source integrity evidence')
        ids={e.event_id for e in self.events}
        if not ids or any(e.recording_id!=self.recording_id for e in self.events) or {i for s in self.segments for i in s.event_ids}!=ids:raise ValueError('Wrong recording or missing event provenance')
        if any(s.end>self.timebase.frames/self.timebase.fps+1e-6 for s in self.segments):raise ValueError('Clip outside source')
        return self
