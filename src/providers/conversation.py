"""Small caller-owned references; no transcript, model memory or executable state."""
import re
from pydantic import Field
from src.query.schema import Closed,Answer,QueryResult
from src.query.answers import answer_result,explain_event

class ConversationState(Closed):
    recording_id:str|None=Field(default=None,max_length=80)
    last_event_id:str|None=Field(default=None,max_length=100)
    last_player_id:int|None=Field(default=None,gt=0,strict=True)
    last_timestamp:float|None=Field(default=None,ge=0)
    last_team:str|None=None
    last_query_type:str|None=None

def followup(tools,question,mode,state,recording_id):
    q=question.strip().lower().rstrip('?.!')
    kind='seek' if q in ('take me there','go there','show that moment') else 'player' if q in ('tell me more about that player','what about that player','tell me more about him') else 'event' if q in ('why','why was it difficult','why was that transfer difficult','explain it','what happened before it','what happened before this event') else None
    if kind is None:return None
    if state.recording_id!=recording_id:return answer_result(QueryResult(question_type='unsupported',status='unknown',limitations=['Follow-up context is missing or belongs to another recording.']),mode)
    if kind=='player' and state.last_player_id is not None:return answer_result(tools.get_player_summary(state.last_player_id),mode)
    event=next((e for e in tools.events if e.event_id==state.last_event_id),None)
    if event:
        if kind=='event':return explain_event(tools,event.event_id,mode,include_sequence='before' in q)
        return answer_result(tools._result('events',[(event.event_id,'events.json',dict(type=event.type.value),event.timestamp,event.event_id)]),mode)
    if kind=='seek' and state.last_timestamp is not None and any(p.get('player_id')==state.last_player_id and p.get('peak_speed_seconds')==state.last_timestamp for p in tools.players):
        # Timestamp must still exist in a deterministic player record.
        return answer_result(tools._result('player_summary',[(str(state.last_player_id),'player_summary.csv',dict(player_id=state.last_player_id),state.last_timestamp,None)]),mode)
    return answer_result(QueryResult(question_type='unsupported',status='unknown',limitations=['No verified event/player reference is available for this follow-up.']),mode)

def remember(answer,recording_id):
    event=next((s.event_id for s in answer.statements if s.event_id),None)
    timestamp=next((s.timestamp for s in answer.statements if s.timestamp is not None),None)
    actor=next((f.value for f in answer.facts if f.field in ('actor_id','sender_id','player_id') and f.source.startswith(('player_summary.csv','transfers.csv')) and isinstance(f.value,int) and not isinstance(f.value,bool) and f.value>0),None)
    return ConversationState(recording_id=recording_id,last_event_id=event,last_timestamp=timestamp,last_player_id=actor,last_query_type=answer.question_type)
