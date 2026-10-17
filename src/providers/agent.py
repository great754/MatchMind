"""Bounded tool routing and closed selection over provider-independent evidence."""
from time import perf_counter
import json
import re
import logging
from pydantic import Field
from typing import Literal
from src.query import query_question,dispatch,tool_specifications
from src.query.answers import answer_result
from src.query.schema import Closed,Answer,Selection,QueryResult
from .base import ToolPlan
from .diagnostics import MicrosoftProviderError
from src.query.dispatch import REGISTRY
from .conversation import ConversationState,followup,remember

class Evidence(Closed):
    fact_id:str
    event_id:str|None=None
    timestamp:float|None=None
    source:str
    field:str
    value:float|int|str|bool|None
    category:Literal['observed','inferred','validated','heuristic','unknown']

class AgentResponse(Closed):
    answer:Answer
    provider:str
    fallback:bool=False
    provider_status:str='local'
    evidence:list[Evidence]=Field(default_factory=list)
    conversation:ConversationState=Field(default_factory=ConversationState)
    tools:list[str]=Field(default_factory=list)
    provider_latency_ms:float=0

def evidence_for(answer):
    rows=[]
    for f in answer.facts:
        category='unknown' if f.value is None else 'validated' if (f.field=='validated_by_bas' and f.value is True) or (f.field=='validation_state' and f.value=='bas_time_actor_match') else 'heuristic' if any(x in f.field for x in ('difficulty','formation','pressure','compactness','block')) else 'observed' if f.field in ('match_id','scope','half','frames','actor_id','player_id','mot_id','team') else 'inferred'
        rows.append(Evidence(**{k:getattr(f,k) for k in ('fact_id','event_id','timestamp','source','field','value')},category=category))
    return rows

def ask(tools,question,mode='fan',timestamp=0.,provider=None,event_id=None,conversation=None,recording_id='recording',requested_provider='local'):
    state=ConversationState.model_validate(conversation or {})
    local=followup(tools,question,mode,state,recording_id)
    contextual=local is not None
    if local is None:local=query_question(tools,question,mode,timestamp,event_id)
    def response(answer,**kwargs):return AgentResponse(answer=answer,evidence=evidence_for(answer),conversation=remember(answer,recording_id),**kwargs)
    blocked=bool(re.search(r'\b(coach|intent|cause|caused|weather|predict|score|fatigue|tired)\b',question.lower()))
    freeform=local.status=='unsupported' and provider is not None and not blocked
    unsupported_local=answer_result(QueryResult(question_type='unsupported',status='unsupported'),mode)
    if contextual or (local.status!='ok' and not freeform):
        return response(local,provider='local',provider_status='local_followup' if contextual else 'local')
    if provider is None:
        return response(local,provider='local',fallback=requested_provider!='local',provider_status='unavailable_fallback' if requested_provider!='local' else 'local')
    start=perf_counter();selected=[];stage='tool_plan'
    try:
        plan=ToolPlan.model_validate(provider.plan(question+(f'\nCurrent recording time: {timestamp:g} seconds.' if freeform else ''),tool_specifications()))
        stage='tool_arguments';results=[]
        for call in plan.calls:
            if call.name not in REGISTRY:raise ValueError('Unsupported tool')
            selected.append(call.name)
            results.append(dispatch(tools,call.name,call.arguments))
        # Keep facts stable across presentation providers. Routing must retrieve
        # evidence for this question; unrelated tools cannot replace its answer.
        stage='tool_evidence'
        if freeform:
            from src.query.answers import combine
            canonical=combine('match_summary',*results)
            # Closed, evidence-derived English catalogue; provider selects only.
            local=answer_result(canonical,mode,question)
            if not any(f.value is not None for f in canonical.facts):raise ValueError('No evidence')
        supported={f.fact_id for f in local.facts}
        if not any(f.fact_id in supported for r in results for f in r.facts):raise ValueError('Unrelated tool plan')
        canonical=QueryResult(question_type=local.question_type,status=local.status,facts=local.facts,limitations=local.limitations,provenance=sorted({f.source.split('#')[0] for f in local.facts}))
        stage='selection'
        selection=Selection.model_validate(provider.select(question,mode,[canonical],local.statements))
        ids=selection.statement_ids
        if freeform and not ids:
            return response(unsupported_local,provider=provider.configuration.get('provider','external'),provider_status='insufficient_evidence',tools=selected,provider_latency_ms=(perf_counter()-start)*1000)
        if not ids or len(ids)!=len(set(ids)) or any(type(i)!=int or i<0 or i>=len(local.statements) for i in ids):raise ValueError('Unsupported selection')
        final=local.model_copy(update={'statements':[local.statements[i] for i in ids]})
        return response(final,provider=provider.configuration.get('provider','external'),provider_status='grounded_selection',tools=selected,provider_latency_ms=(perf_counter()-start)*1000)
    except Exception as error:
        if freeform:local=unsupported_local
        code=error.code.value if isinstance(error,MicrosoftProviderError) else {'tool_plan':'tool_plan_invalid','tool_arguments':'tool_arguments_invalid','tool_evidence':'unrelated_tool_plan','selection':'selection_invalid'}.get(stage,'agent_error')
        logging.getLogger('tactiqai.providers').warning(json.dumps(dict(event='agent_fallback',provider='external',stage=stage,code=code)))
        return response(local,provider='local',fallback=True,provider_status='Provider unavailable or unsupported response; deterministic answer retained.',tools=selected,provider_latency_ms=(perf_counter()-start)*1000)
