"""Closed multi-operation intent plans; only deterministic services resolve evidence."""
import re,json,os
from .schema import AssistantPlan

class IntentAndQueryPlanner:
    def plan(self,q,players=None,previous=None,provider='auto',roster=None,match_context=None):
        text=q.lower();p={'players':players or []};reason=None
        actions=[]
        for k,pattern in [('goal',r'\b(goal|goals|scored|scoring)\b'),('shot',r'\b(shots?|attempts? on goal)\b'),('pass',r'\b(pass|passes|passing)\b'),('reception',r'\b(receptions?|received passes)\b'),('tackle',r'\btackles?\b'),('interception',r'\binterceptions?\b'),('cross',r'\bcross(?:es)?\b'),('header',r'\bheaders?\b'),('carry',r'\b(carries|carry|carrying)\b'),('sprint',r'\b(sprints?|fastest run)\b')]:
            if re.search(pattern,text):actions.append(k)
        if 'reception' in actions and not re.search(r'passes and receptions|passes,.*receptions',text):actions=[a for a in actions if a!='pass']
        p['review']=bool(re.search(r'\b(possible|candidates?|review|uncertain)\b',text));p['actions']=actions;p['video']=bool(re.search(r'\b(show|watch|video|clip|highlight|play|again|see it)\b',text));p['overlay']='highlight the player' in text or 'highlight that player' in text;p['explain']=bool(re.search(r'\b(why|explain|defenders|difficult because)\b',text))
        duration_text=re.sub(r'\b(?:\d+|one|a)[ -]seconds?\b','',text)
        ordinal=re.search(r'\b(first|second|third|fourth|\d+(?:st|nd|rd|th))\b',duration_text)
        if ordinal:p['ordinal']={'first':1,'second':2,'third':3,'fourth':4}.get(ordinal[1],int(re.sub(r'\D','',ordinal[1]) or 0))
        duration=re.search(r'\b(\d+|one|a)[ -]?(seconds?|minutes?)\b',text)
        if duration and not re.search(r'\b(before|after|between|from)\b',text):p['duration']=(int(duration[1]) if duration[1].isdigit() else 1)*(60 if duration[2].startswith('minute') else 1)
        if re.search(r'\b(buildup|build-up|longer clips)\b',text):p['buildup']='more'
        if 'shorter' in text:p['buildup']='short'
        if 'first half' in text:p['period']=1;p.pop('ordinal',None)
        if 'second half' in text:p['period']=2;p.pop('ordinal',None)
        timerange=re.search(r'\b(?:between|from) (\d+):(\d\d) (?:and|to|until) (\d+):(\d\d)',text)
        if timerange:p.update(start=60*int(timerange[1])+int(timerange[2]),end=60*int(timerange[3])+int(timerange[4]))
        for name,pattern in [('left',r'\b(team a|team 1|left team|left side)\b'),('right',r'\b(team b|team 2|right team|right side)\b')]:
            if re.search(pattern,text):p['team']=name
        if re.search(r'\b(dribbles?|recoveries|ball recoveries|fastest shot|fastest pass|biggest chances)\b',text):p['operation']='unsupported';reason='I don’t have reliable records for that action or measurement in this footage.'
        elif re.search(r'\b(stats|statistics|player profile|performance overview|statistical breakdown)\b',text):p['operation']='player_profile';p['video']=bool(re.search(r'\b(video|clip|compilation|footage)\b',text))
        elif re.search(r'\b(score|scoreline|who is winning|who won|who is ahead)\b',text):p['operation']='score'
        elif 'compare' in text:p['operation']='compare';p['metric']='shots' if 'shot' in text else 'passes' if 'pass' in text else 'tackles' if 'tackle' in text else None
        elif re.search(r'\b(who was that|who was the player|who made|who played|who is that)\b',text) and not re.search(r'\b(longest|farthest|furthest|most difficult|hardest|fastest|sprint|most passes|most shots|most tackles|most possession)\b',text):p.update(operation='identity',reference='event')
        elif ('sprint' in text or re.search(r'\bfastest run\b',text)) and re.search(r'\b(how many|count)\b',text):p.update(operation='statistic',metric='speed')
        elif 'sprint' in text or re.search(r'\b(fastest run|fastest player)\b',text):p.update(operation='sprint',metric='speed',overlay=p['video'] or p['overlay'])
        elif 'pass' in text and re.search(r'\b(longest|farthest|furthest|most difficult|hardest|most challenging)\b',text):p.update(operation='pass_rank',metric='difficulty' if re.search(r'\b(most difficult|hardest|most challenging)\b',text) and not re.search(r'\b(longest|farthest|furthest)\b',text) else 'distance',video=True)
        elif previous and re.search(r'\b(that pass|that clip|that one|it difficult|defenders around|the second one|the first one|second one|again|take me there|go there|more buildup|shorter clips|longer clips|highlight that player|highlight the player)\b',text):p.update(operation='explain' if p['explain'] else 'followup',reference='event',video=p['video'] or 'take me there' in text)
        elif re.search(r'\b(how many|count|percentage|completion rate)\b',text):p.update(operation='statistic',metric='shots' if 'shot' in text else 'passes' if 'pass' in text else 'tackles' if 'tackle' in text else None)
        elif re.search(r'\b(which team|which side)\b',text):p.update(operation='compare' if any(a in actions for a in ['pass','shot','tackle']) else 'answer',metric='passes' if 'pass' in actions else 'shots' if 'shot' in actions else 'tackles' if 'tackle' in actions else None)
        elif re.search(r'\b(most passes|most shots|most tackles|most possession|highest recorded speed|highest speed)\b',text):p.update(operation='statistic',metric='passes' if 'passes' in text else 'shots' if 'shots' in text else 'tackles' if 'tackles' in text else 'possession' if 'possession' in text else 'speed')
        elif re.search(r'\b(ran the farthest|ran furthest|most distance|longest distance|ran the most)\b',text):p.update(operation='statistic',metric='distance')
        elif p['explain'] and 'goal' in actions:p.update(operation='explain',video=True)
        elif re.search(r'\b(recap|highlights|highlight video|attacking plays|exciting moments)\b',text) and not p['players'] and not actions:p.update(operation='recap',video=True)
        elif re.search(r'\b(actions|everything|all his|all her|all their|this player|attacking actions|what he did|what she did|what they did)\b',text) or p['video'] and (actions or p['players']):p['operation']='actions'
        else:p['operation']='answer'
        if previous and re.search(r'\b(his|her|their|he|she|this player|that player)\b',text) and not p['players']:p.update(players=previous.get('players',[]),reference='player')
        if previous and p.get('reference')=='event':
            if 'duration' not in p:p['duration']=previous.get('plan',{}).get('duration')
            if p['video'] and previous.get('plan',{}).get('overlay'):p['overlay']=True
        local=AssistantPlan.model_validate(p);used='local';diagnostic=None
        chosen=provider if provider!='auto' else 'gemini' if os.environ.get('GEMINI_API_KEY') else 'local'
        if chosen!='local':
            try:
                from src.providers.factory import build_provider
                client=build_provider(chosen)
                instruction=('You are the question interpreter for TactiqAI. Interpret natural language directly, not by copying the fallback plan. '
                 'Return only a closed supported plan. Choose player_profile for broad player stats, form/performance summaries or requests for their numbers; '
                 'score for scorelines; actions with video=true for show/watch/generate goals, shots or other recorded plays; pass_rank for longest/difficult passes; '
                 'statistic for a specific count; sprint for sustained runs; compare for comparisons; recap for highlight summaries; answer for other factual tools. '
                 'A request can combine statistics, explanation and video. Use previous_plan/selected player for followups. '
                 'Only use provided player references. Jersey numbers mean confirmed jerseys, never actor/track IDs or temporary label indexes. '
                 'Unknown names/numbers, ambiguous players or unsupported actions require clarification, never silently discard their filter. '
                 'Never invent metrics, action outcomes or event timestamps. Start/end are explicit user time filters bounded by recording duration. '
                 'Distinguish longest distance from estimated difficulty. Normal goal videos include buildup; more means longer lead-in. '
                 'Do not claim tackle outcomes, dribbles, recoveries, assists or speed truth that is absent from the capability list. '
                 'Set review=true only when the user explicitly asks for possible/uncertain/candidate plays or review; candidates are never confirmed events. '
                 'Clarification may ask a short question but must contain no football facts or measurements.')
                payload={'question':q,'resolved_players':local.players,'roster':roster or [],'match_context':match_context or {},'previous_plan':previous.get('plan') if previous else None,'fallback_hint':local.model_dump(),'capabilities':['recorded goals/shots/crosses/tackles/headers/carries','qualified pass/reception/interception estimates','player statistics/heatmap/pass map','sustained metric sprint estimates','recorded score','source video clips with buildup']}
                if chosen=='gemini':
                    raw=client.client.generate_structured(instruction,payload,AssistantPlan.model_json_schema());selected=AssistantPlan.model_validate_json(raw)
                else:
                    tool=client.plan(instruction+'\n'+json.dumps(payload),[{'name':'assistant_query','input_schema':AssistantPlan.model_json_schema()}]);assert len(tool.calls)==1 and tool.calls[0].name=='assistant_query';selected=AssistantPlan.model_validate(tool.calls[0].arguments)
                if selected.review and not local.review:raise ValueError('Unrequested candidate review')
                if local.review and not selected.review and selected.operation!='unsupported':raise ValueError('Explicit candidate review request lost')
                allowed={p['player_ref'] for p in roster} if roster is not None else set(local.players)
                if any(ref not in allowed for ref in selected.players):raise ValueError('Unknown player reference')
                if local.players and not set(local.players)<=set(selected.players) and selected.operation not in ['unsupported','score']:raise ValueError('Resolved player filter lost')
                if local.operation=='pass_rank' and local.metric=='distance' and selected.metric=='difficulty':raise ValueError('Distance changed to difficulty')
                if (selected.start is not None or selected.end is not None) and local.start is None and local.end is None and not re.search(r'\b(before|after|between|until|minutes?|seconds?|last|opening|closing)\b|\d+:\d\d',text):raise ValueError('Unrequested time filter')
                duration=(match_context or {}).get('duration_seconds',14400)
                if any(value is not None and value>duration for value in [selected.start,selected.end]):raise ValueError('Time filter outside recording')
                if local.start is not None and selected.start!=local.start or local.end is not None and selected.end!=local.end:raise ValueError('Explicit time range changed')
                if local.team is not None and selected.team!=local.team:raise ValueError('Explicit team filter lost')
                if local.reference=='event' and selected.reference not in ['event','result']:raise ValueError('Previous event reference lost')
                if selected.operation=='unsupported':reason=selected.clarification or 'I don’t have reliable evidence for that request. Try a recorded action, statistic or player selection.'
                local=selected;used=chosen
            except Exception:diagnostic='The language model could not interpret this request, so the built-in fallback was used.';used='local_fallback'
        return local,reason,used,diagnostic
