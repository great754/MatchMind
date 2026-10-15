"""Gemini query presentation using the same closed catalogue and tool boundary."""
import json
from src.intelligence.commentary.gemini_provider import GeminiCommentaryProvider
from src.query.schema import Selection
from .base import ToolPlan

class GeminiQueryProvider:
    def __init__(self,model='gemini-3.5-flash-lite',client=None):self.client=client or GeminiCommentaryProvider(model)
    @property
    def configuration(self):return dict(provider='gemini',model=self.client.configuration.get('model','configured'))
    def plan(self,question,tool_specs):
        # Gemini's existing selector accepts a catalogue; it chooses a validated
        # tool schema here, with conservative arguments and no computed numbers.
        candidates={s['name']:s for s in tool_specs if not s['input_schema'].get('required')}
        raw=self.client.generate(dict(instruction='Choose one tool name from sentences. Return JSON sentence_ids only; do not invent statistics.',question=question,sentences={k:json.dumps(v) for k,v in candidates.items()}))
        plan=json.loads(raw);names=plan['sentence_ids']
        if len(names)!=1 or names[0] not in candidates:raise ValueError('Invalid Gemini tool selection')
        return ToolPlan(calls=[dict(name=names[0],arguments={'limit':1} if names[0]=='get_fastest_players' else {})])
    def select(self,question,mode,results,statements):
        request=dict(instruction='Select supported statement IDs only. Return JSON sentence_ids. Never calculate or write new claims.',question=question,mode=mode,facts=[r.model_dump(mode='json') for r in results],sentences={str(i):s.text for i,s in enumerate(statements)})
        raw=json.loads(self.client.generate(request))
        if set(raw)!= {'sentence_ids'} or any(not isinstance(i,str) or i not in request['sentences'] for i in raw['sentence_ids']):raise ValueError('Invalid Gemini selection')
        return Selection(statement_ids=[int(i) for i in raw['sentence_ids']])
