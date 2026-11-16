"""Language produces plans only. Providers cannot select evidence or invent timestamps."""
import os,re,json
from .schema import HighlightQueryPlan as Plan

WORDS={'first':1,'second':2,'third':3,'fourth':4,'fifth':5,'one':1,'two':2,'three':3,'thirty':30,'sixty':60}
class HighlightAgent:
    def parse(self,question,previous=None):
        q=question.lower().strip().replace('’',"'");base={};follow=False
        if previous and (re.search(r'\b(one|ones|that pass|those|them|buildup|shorter|longer)\b',q) or re.search(r'who (made|played|passed|scored)',q)):
            base=previous.model_dump();follow=True
        p={**base,'intent':'create'}
        kinds=[]
        for kind,pattern in [('goal',r'\b(goals?|scor(?:ed|ing)|netted)\b'),('shot',r'\b(shots?|attempts?|strikes?|shooting)\b'),('pass',r'\b(pass(?:es)?|passing|through balls?)\b'),('cross',r'\bcross(?:es)?\b'),('tackle',r'\btackles?\b'),('save',r'\bsaves?\b'),('possession_change',r'\b(turnovers?|possession changes?)\b')]:
            if re.search(pattern,q):kinds.append(kind)
        if kinds:p['event_types']=kinds
        if re.search(r'\b(recap|highlights?|exciting|attacking plays|attacking moments|biggest chances)\b',q) and not kinds:
            p.update(event_types=[],selection='recap')
        elif not kinds and not follow:p['intent']='unsupported'
        if re.search(r'who (made|played|passed|scored)',q):p['intent']='identity' if previous else 'unsupported'
        if re.search(r'\b(longest|farthest|furthest|longest-distance)\b',q):p.update(selection='maximum',metric='distance_m')
        if re.search(r'\b(fastest|hardest|most powerful)\b',q):p.update(selection='maximum',metric='shot_speed')
        if 'biggest chances' in q:p.update(selection='maximum',metric='chance_quality')
        ordinal_text=re.sub(r'\b(?:\d+|one|thirty|sixty)[ -]seconds?\b','',q)
        ordinal=re.search(r'\b(first|second|third|fourth|fifth|\d+(?:st|nd|rd|th))\b',ordinal_text)
        number_word=re.search(r'\bnumber (one|two|three)\b',q)
        if number_word:p.update(selection='ordinal',ordinal=WORDS[number_word[1]])
        if ordinal:p.update(selection='ordinal',ordinal=WORDS.get(ordinal[1],int(re.sub(r'\D','',ordinal[1]) or '0')))
        if re.search(r'\b(all|every|each)\b',q):p.update(selection='all',ordinal=None)
        if re.search(r'\b(buildup|build-up|before.*scor(?:ed|ing)|lead.up)\b',q):p['buildup']='more'
        if re.search(r'\b(shorter|less buildup|short clips?)\b',q):p['buildup']='short'
        if re.search(r'\b(more buildup|longer clips?)\b',q):p['buildup']='more'
        for name,pattern in [('home',r'\bhome\b'),('away',r'\baway\b'),('left',r'\b(left (?:side|team)|team 1)\b'),('right',r'\b(right (?:side|team)|team 2)\b')]:
            if re.search(pattern,q):p['team']=name
        player=re.search(r'\b(?:(tracked|tracker|actor|jersey)\s+)?(?:player\s+)?(?:number\s+|#)?(\d+)\s*(?:\x27s)?\s*(?:highlights|moments|passes|goals)',q) or re.search(r'\b(?:(tracked|tracker|actor|jersey)\s+)?player\s+#?(\d+)\b',q)
        if player and p['intent']=='unsupported':p['intent']='create'
        if player:p.update(player=int(player[2]),player_namespace='actor' if player[1]=='actor' else 'tracker' if player[1] in ['tracked','tracker'] else 'jersey')
        duration=re.search(r'\b(\d+|one|thirty|sixty)[ -]?(seconds?|minutes?)\b',q)
        if duration and not re.search(r'\b(after|before|between|from)\b',q):
            n=WORDS.get(duration[1],int(duration[1]) if duration[1].isdigit() else 0);p['duration']=n*(60 if duration[2].startswith('minute') else 1)
        timerange=re.search(r'\b(?:between|from)\s+(\d+):(\d{2})\s+(?:and|to|until|-)\s+(\d+):(\d{2})',q)
        if timerange:p.update(start=60*int(timerange[1])+int(timerange[2]),end=60*int(timerange[3])+int(timerange[4]))
        single=re.search(r'\b(after|before)\s+(\d+):(\d{2})',q)
        if single:p['start' if single[1]=='after' else 'end']=60*int(single[2])+int(single[3])
        if re.search(r'\b(review|candidate|unconfirmed)\b',q):p['review']=True
        # Unrecognized names/teams must not silently disappear as filters.
        if re.search(r'\b(?:by|for)\s+(?!the\s+(?:home|away|left|right)|(?:home|away|left|right|team|player|tracked|actor)\b)[a-z]+',q):p['intent']='unsupported'
        return Plan.model_validate(p)
    def interpret(self,question,previous=None,provider='auto'):
        local=self.parse(question,previous);used='local';diagnostic=None
        name=provider
        if name=='auto':name='gemini' if os.environ.get('GEMINI_API_KEY') else 'local'
        if name!='local' and local.intent!='unsupported':
            try:
                from src.providers.factory import build_provider
                client=build_provider(name)
                # Gemini uses its existing closed catalogue selector; Microsoft uses
                # the existing structured function-call abstraction.
                if name=='gemini':
                    options={'parsed':local.model_dump()}
                    for kind in ['goal','shot','pass','cross','tackle','save']:
                        options[kind]=local.model_copy(update={'event_types':[kind]}).model_dump()
                    response=json.loads(client.client.generate({'instruction':'Select the query plan that matches the request. Return JSON sentence_ids with one ID only. Do not invent facts, IDs or times. Keep filters and unsupported metrics unchanged.','question':question,'sentences':{k:json.dumps(v) for k,v in options.items()}}))
                    if set(response)!={'sentence_ids'} or len(response['sentence_ids'])!=1:raise ValueError()
                    selected=Plan.model_validate(options[response['sentence_ids'][0]])
                else:
                    result=client.plan(question,[{'name':'highlight_query','input_schema':Plan.model_json_schema()}]);call=result.calls[0]
                    if len(result.calls)!=1 or call.name!='highlight_query':raise ValueError()
                    selected=Plan.model_validate(call.arguments)
                    # Numeric filters/ranking cannot be broadened by a hosted model.
                    for key in ['team','player','player_namespace','start','end','duration','metric','review','selection','ordinal']:
                        if getattr(selected,key)!=getattr(local,key):raise ValueError()
                # Do not let a provider silently replace an explicitly parsed event.
                if selected.event_types!=local.event_types:raise ValueError()
                local=selected;used=name
            except Exception:used='local_fallback';diagnostic='Hosted interpretation unavailable or inconsistent; used the local query interpreter.'
        return local,used,diagnostic
