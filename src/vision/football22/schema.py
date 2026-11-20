"""Finite source-indexed M22 visual diagnostics, separate from football facts."""
from typing import Literal
from pydantic import Field,model_validator
from src.query.schema import Closed

class FootballCandidate(Closed):
    candidate_id:str
    bbox:tuple[float,float,float,float]
    confidence:float=Field(ge=0,le=1)
    current_only_confidence:float=Field(ge=0,le=1)
    observation_eligible:bool
    decision_reason:str
    selected:bool

class FootballFrame(Closed):
    schema_version:Literal['cv-m22-frame-v1']
    recording_id:str
    frame:int=Field(ge=0)
    timestamp:float=Field(ge=0)
    camera_epoch:int=Field(ge=0)
    context_frames:tuple[int,int,int]
    checkpoint_sha256:str=Field(pattern=r'^[0-9a-f]{64}$')
    directly_observed:bool
    candidates:list[FootballCandidate]
    heatmap:list[list[float]]
    heatmap_is_calibrated:Literal[False]
    association:Literal['existing_m21_offline_dag']
    uses_future_for_association:Literal[True]
    reconstruction:dict|None
    interaction:dict

    @model_validator(mode='after')
    def current_evidence(self):
        if len(self.heatmap)!=24 or any(len(r)!=40 or any(not 0<=v<=1 for v in r) for r in self.heatmap):raise ValueError('Invalid source-mapped activation grid')
        if self.context_frames[-1]!=self.frame or any(i<0 or i>self.frame for i in self.context_frames):raise ValueError('Temporal visual input cannot contain future frames')
        if tuple(sorted(self.context_frames))!=self.context_frames:raise ValueError('Temporal context must be chronological')
        if len({c.candidate_id for c in self.candidates})!=len(self.candidates):raise ValueError('Duplicate candidates')
        selected=[c for c in self.candidates if c.selected]
        if len(selected)!=int(self.directly_observed):raise ValueError('Current observation requires one actual selected visual candidate')
        for c in self.candidates:
            if min(c.bbox[:2])<0 or min(c.bbox[2:])<=0:raise ValueError('Candidate outside positive source pixels')
            eligible=c.confidence>=.35 and c.current_only_confidence>=.35
            if c.observation_eligible!=eligible or c.selected and not eligible:raise ValueError('Neural confidence/current-image gate mismatch')
        if self.directly_observed and self.reconstruction is not None:raise ValueError('Reconstruction is not an observation')
        return self
