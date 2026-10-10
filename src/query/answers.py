"""Grounded local Q&A. Providers may select a closed catalogue, not generate claims."""
import re
from typing import Protocol
from .schema import Answer,Statement,QueryResult,Selection

UNKNOWN='MatchMind cannot determine that from the available tracking data.'

class StatementSelector(Protocol):
    def select(self,question:str,mode:str,catalogue:list[Statement])->Selection: ...


def answer_result(result,mode='fan',question='',selector=None):
    if mode not in ('beginner','fan','analyst'):raise ValueError('Unknown explanation mode')
    grouped={}
    for f in result.facts:grouped.setdefault((f.record_id,f.timestamp,f.event_id),[]).append(f)
    statements=[]
    for (_,timestamp,event_id),facts in grouped.items():
        parts=[]
        for f in facts:
            if f.value is None:continue
            value=f'{f.value:.3f}'.rstrip('0').rstrip('.') if isinstance(f.value,float) else str(f.value)
            label=f.field.replace('_',' ')
            if mode=='beginner':
                label={'compactness_m':'how tightly players grouped (m)','forward_progression_m':'distance gained toward goal (m)',
                    'passing_lane_defenders':'opponents along the ball route','sender_nearby_defenders':'opponents near the sender',
                    'fraction_of_all_frames':'share of the full recording','fraction_of_controlled_frames':'share of confirmed ball control'}.get(f.field,label)
            if f.field=='difficulty_score_0_100':label='retrospective heuristic difficulty (0–100)'
            if f.field=='max_speed_kmh':label='peak estimated speed (km/h)'
            if f.field=='ball_speed_peak_kmh':label='peak estimated 2D ball speed (km/h)'
            if f.field=='ball_speed_kmh':label='estimated 2D ball speed (km/h)'
            if f.field=='ball_speed_median_kmh':label='median estimated 2D ball speed (km/h)'
            if f.field=='outcome' and f.value=='intercepted':value='opposite-team receipt (interception candidate)'
            if f.field=='type':value=value.replace('PASS','TRANSFER')
            if f.field.startswith('fraction_of_'):label+=' (fraction)'
            if f.field=='mot_id':label='MOT tracker ID (not verified jersey number)'
            parts.append(f'{label}: {value}')
        if parts:statements.append(Statement(text='; '.join(parts),fact_ids=[f.fact_id for f in facts],timestamp=timestamp,event_id=event_id))
    if not statements:statements=[Statement(text=UNKNOWN,fact_ids=[])]
    if selector:
        selection=Selection.model_validate(selector.select(question,mode,statements))
        if not selection.statement_ids or len(set(selection.statement_ids))!=len(selection.statement_ids) or any(i<0 or i>=len(statements) for i in selection.statement_ids):
            raise ValueError('Provider selected unsupported statements')
        statements=[statements[i] for i in selection.statement_ids]
    explanation={
        'beginner':'Team possession means inferred control of the ball. A transfer is observed movement between players; it may not be an intentional pass.',
        'fan':'Transfers are observable ball movements. Difficulty and tactical labels are heuristic estimates.',
        'analyst':'Facts retain source-field provenance. BAS support checks release time/sender; centered ownership smoothing and BAS-anchored ball interpolation prevent independent causal validation.'}[mode]
    return Answer(mode=mode,question_type=result.question_type,status=result.status,statements=statements,
        limitations=list(dict.fromkeys(result.limitations+[explanation])),supporting_fact_ids=[f.fact_id for f in result.facts],facts=result.facts)


def combine(kind,*results):
    return QueryResult(question_type=kind,status='ok' if any(r.status=='ok' for r in results) else 'unknown',
        facts=[f for r in results for f in r.facts],limitations=list(dict.fromkeys(x for r in results for x in r.limitations)),
        provenance=sorted({x for r in results for x in r.provenance}))


def explain_event(tools,event_id,mode='fan',include_sequence=False):
    event=next((e for e in tools.events if e.event_id==event_id),None)
    event_facts=tools._result('events',[(event_id,'events.json',dict(type=event.type.value,player_id=event.player_id,actor_id=event.actor_id,receiver_id=event.receiver_id,team=event.team,**event.metrics.model_dump(exclude_none=True)),event.timestamp,event_id)] if event else [])
    result=combine('transfer' if event and event.source=='transfer' else 'pre_event_evidence',event_facts,tools.get_transfer(event_id),tools.get_pre_event_evidence(event_id),tools.get_event_validation(event_id))
    if include_sequence:
        result=combine(result.question_type,tools.get_event_sequence(event_id),result)
    if not result.facts:
        result=tools.get_events()
        result=QueryResult(question_type='events',status='unknown',limitations=result.limitations)
    return answer_result(result,mode)


def intent(question):
    """Conservative finite intent router, shared with offline report catalogue."""
    q=question.lower().strip()
    if re.search(r'\b(tired|fatigue|coach|intent|score|jersey|name|win|predict|weather)\b',q):return 'unsupported'
    if re.search(r'\b(fastest|quickest)\b',q):return 'fastest'
    if re.search(r'\b(hardest|difficult|difficulty)\b',q):return 'difficult'
    if 'possession' in q:return 'possession'
    if 'formation' in q:return 'formation'
    if re.search(r'\b(pressure|pressured)\b',q):return 'pressure'
    if re.search(r'\b(connect|connected|teammates|network)\b',q):return 'network'
    if re.search(r'\b(shape|width|compactness)\b',q):return 'shape'
    if 'high block' in q:return 'block'
    if re.search(r'\b(before|explain|turnover|shot|event)\b',q):return 'event'
    if re.search(r'\b(summary|overview)\b',q):return 'summary'
    return 'unsupported'


def query_question(tools,question,mode='fan',timestamp=0,event_id=None):
    q=question.lower();kind=intent(q)
    team='left' if re.search(r'\b(team\s*1|left|red)\b',q) else 'right' if re.search(r'\b(team\s*2|right|blue)\b',q) else None
    # Color aliases are intentionally excluded: color/team assignments vary by recording.
    if re.search(r'\b(red|blue)\b',q):return answer_result(QueryResult(question_type='unsupported',status='unsupported',limitations=['Use left/right or Team 1/Team 2; color labels depend on the selected match.']),mode)
    if kind=='unsupported':return answer_result(QueryResult(question_type='unsupported',status='unsupported'),mode)
    if kind=='fastest':result=tools.get_fastest_players(1)
    elif kind=='difficult':
        result=tools.get_difficult_transfers(team,1)
        if result.facts:
            eid=result.facts[0].event_id
            return explain_event(tools,eid,mode)
    elif kind=='possession':result=tools.get_possession_summary(team)
    elif kind=='formation':result=combine('formation_timeline',*[tools.get_formation_timeline(t) for t in ([team] if team else ['left','right'])])
    elif kind=='pressure':result=tools.get_pressure_peaks(team)
    elif kind=='network':
        network=tools.get_transfer_network(team)
        groups={}
        for fact in network.facts:
            if fact.record_id.startswith('node:'):groups.setdefault(fact.record_id,[]).append(fact)
        ordered=sorted(groups.values(),key=lambda facts:(-next(f.value for f in facts if f.field=='distinct_teammates'),facts[0].record_id))
        best=ordered[:1]
        result=QueryResult(question_type='transfer_network',status='ok' if best else 'unknown',facts=[f for group in best for f in group],
            limitations=network.limitations+['Connected teammates counts distinct incoming/outgoing partners in completed observable transfers; ties use stable player order.'],provenance=network.provenance)
    elif kind=='shape':
        records=[]
        for side,phases in tools.tactical_summary.get('phase_comparison',{}).items():
            if team is not None and side!=team:continue
            for phase in ('possessing','defending'):
                records.append((side+':'+phase,'tactical_summary.json',dict(team=side,phase=phase,**phases[phase]),None,None))
        result=tools._result('tactical_state',records,('Whole-recording observed phase means, not causal effects of possession.',))
    elif kind=='block':
        result=tools.get_events('TACTICAL_BLOCK_CHANGED',team)
        # Actual confirmed high-state intervals are selected from snapshots, rather than claiming every block event is high.
        rows=[];previous={}
        for s in tools.states:
            for side in ([team] if team else ['left','right']):
                block=getattr(s,side).block
                if block=='high' and previous.get(side)!='high':rows.append((side+':'+str(s.frame),'tactical_states.json',dict(team=side,block=block),s.timestamp,None))
                previous[side]=block
        result=tools._result('events',rows)
    elif kind=='event':
        stamp=re.search(r'\b(\d+):(\d+(?:\.\d+)?)\b',q)
        if stamp:timestamp=int(stamp[1])*60+float(stamp[2])
        if event_id is None:
            types=['TRANSFER_INTERCEPTED'] if 'turnover' in q else ['SHOT_CANDIDATE'] if 'shot' in q else None
            candidates=[e for e in tools.events if e.source in ('transfer','trajectory','possession') and (types is None or e.type.value in types)]
            event=min(candidates,key=lambda e:abs(e.timestamp-timestamp)) if candidates else None
            event_id=event.event_id if event and (not stamp or abs(event.timestamp-timestamp)<=2) else None
        if event_id:return explain_event(tools,event_id,mode,include_sequence='before' in q or 'sequence' in q)
        result=QueryResult(question_type='pre_event_evidence',status='unknown')
    else:result=tools.get_match_summary()
    return answer_result(result,mode,question)
