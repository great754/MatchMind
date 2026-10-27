"""Read-only frontend projection of already generated report/artifact evidence."""
import json
import re
from typing import Literal
from pathlib import Path
from pydantic import Field,model_validator
from src.query.schema import Closed,QueryResult
from src.query.intelligence import MatchIntelligence, build_intelligence
from src.intelligence.schema import CanonicalEvent
from src.tactics.schema import TacticalState,Pressure
from src.tactics.review import pressure_level

class Replay(Closed):
    fps:float=Field(gt=0)
    frames:int=Field(gt=0)
    stride:int=Field(gt=0)
    ids:list[int]
    teams:list[Literal['left','right']]
    side_colors:dict[str,int]
    positions:list[list[tuple[float|None,float|None]]]
    ball:list[tuple[float|None,float|None]]
    owner:list[int|None]
    states:list[str]
    @model_validator(mode='after')
    def aligned(self):
        count=(self.frames+self.stride-1)//self.stride
        if not len(self.ids)==len(self.teams) or len(set(self.ids))!=len(self.ids):raise ValueError('Invalid actor mapping')
        if any(len(x)!=count for x in (self.positions,self.ball,self.owner,self.states)):raise ValueError('Unaligned review samples')
        if any(len(row)!=len(self.ids) for row in self.positions):raise ValueError('Unaligned player positions')
        return self

class Ownership(Closed):
    frame:int=Field(ge=0)
    timestamp:float=Field(ge=0)
    player_id:int|None=None
    team:Literal['left','right']|None=None
    attribution:Literal['controlled','retained','unknown']

class FormationWindow(Closed):
    team:Literal['left','right']
    start_seconds:float=Field(ge=0)
    end_seconds:float=Field(gt=0)
    estimate:Literal['4-3-3','4-4-2','4-2-3-1','3-5-2','unknown']
    coverage:float=Field(ge=0,le=1)
    dominance:float=Field(ge=0,le=1)
    reason:Literal['insufficient_coverage','ambiguous_fit']|None=None
    samples:int=Field(ge=0)
    window_seconds:float=Field(gt=0)

class Review(Closed):
    recording_id:str
    intelligence:MatchIntelligence|None=None
    status:Literal['ok','unknown']
    summary:QueryResult
    possession:QueryResult
    pass_share:QueryResult
    ownership:list[Ownership]
    formation_windows:list[FormationWindow]
    pressure_levels:list[str]
    pressure_context:list[Pressure]
    unconfirmed_control_fraction:float|None=Field(default=None,ge=0,le=1)
    replay:Replay|None=None
    tactical_states:list[TacticalState]
    events:list[CanonicalEvent]
    media_url:str|None=None
    default_provider:Literal['local','gemini','microsoft','microsoft-framework']='local'
    sources:list[str]
    limitations:list[str]

def load_replay(directory):
    """Called only at startup. Older bundles already embed this exact projection.

    Parse the JSON literal, never execute HTML/JavaScript or read raw tracking.
    New report exports store it directly in review_frames.json.
    """
    directory=Path(directory);p=directory/'review_frames.json'
    if p.is_file():return Replay.model_validate_json(p.read_text()),'review_frames.json'
    p=directory/'report.html'
    if not p.is_file():return None,None
    page=p.read_text();match=re.search(r'\bconst data\s*=\s*',page)
    if not match:return None,None
    try:
        raw,_=json.JSONDecoder().raw_decode(page[match.end():])
        value={k:raw[k] for k in ('fps','frames','stride','ids','teams','positions','ball','owner','states')}
        value['side_colors']=raw['sideColors']
        return Replay.model_validate(value),'report.html#precomputed_review'
    except (ValueError,KeyError,TypeError):return None,None

def review(tools,recording_id,replay,source,default_provider,media_root):
    sources=['summary.json','events.json','tactical_states.json',tools.ownership_source,tools.formation_source,'transfers.csv']
    if source:sources.append(source)
    frames=tools.summary.get('frames');controlled=tools.summary.get('possession_state_frames',{}).get('controlled')
    unconfirmed=1-controlled/frames if isinstance(frames,int) and frames>0 and isinstance(controlled,int) and 0<=controlled<=frames else None
    media=None
    if media_root and tools.summary.get('scope')=='clip':
        match=str(tools.summary.get('match_id',''));root=Path(media_root).resolve()
        p=(root/(match+'.mp4')).resolve()
        if not p.is_file():p=(root/(match+'.txt.mp4')).resolve()
        if re.fullmatch(r'[A-Za-z0-9_-]{1,80}',match) and p.is_relative_to(root) and p.is_file():media=f'/matches/{recording_id}/video.mp4'
    return Review(recording_id=recording_id,intelligence=build_intelligence(tools,recording_id),status='ok' if replay else 'unknown',summary=tools.get_match_summary(),possession=tools.get_possession_summary(),pass_share=tools.get_confirmed_pass_share(),ownership=tools.ownership,formation_windows=tools.formations,pressure_levels=[pressure_level(p) for p in tools.review_pressure],pressure_context=tools.review_pressure,unconfirmed_control_fraction=unconfirmed,replay=replay,tactical_states=tools.states,events=tools.events,media_url=media,default_provider=default_provider,sources=sources,limitations=tools.limitations+([] if replay else ['Player/ball review samples unavailable.'])+['Pitch uses precomputed review samples; no browser interpolation or new detection.','Unconfirmed control includes every frame without confirmed ownership; it is not solely missing tracking.'])
