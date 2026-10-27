"""Bounded semantic retrieval over approved facts, independent of wording/providers."""
import re
from .dispatch import dispatch
from .schema import QueryResult

UNSUPPORTED=re.compile(r'\b(tired|fatigue|coach|intent|score|jersey|name|win|predict|weather|cause|caused|causality|referee|salary|injury)\b',re.I)


def semantic_plan(question,timestamp=0):
    q=question.lower()
    if UNSUPPORTED.search(q):return []
    if re.search(r'\b(match summary|post.match summary|key match insights|summarize (?:this|the) (?:match|clip))\b',q):return [('get_match_intelligence',{})]
    left=bool(re.search(r'\b(team\s*1|left|home(?: side)?)\b',q))
    right=bool(re.search(r'\b(team\s*2|right|away(?: side)?)\b',q))
    both=bool(re.search(r'\b(both|each|altogether|total|together|compare|comparison)\b',q))
    team='left' if left and not right and not both else 'right' if right and not left and not both else None
    calls=[]
    passes=bool(re.search(r'\b(passes|passed|passing|ball transfers|completed transfers|successful transfers|pass count|pass total)\b',q))
    if passes and not re.search(r'\b(difficult|difficulty|hardest|explain|before|network|connected)\b',q):
        calls.append(('get_confirmed_pass_share',{'team':team}))
    if re.search(r'\b(possession|pass share|pass ratio)\b',q):
        call=('get_current_possession',{'timestamp':timestamp}) if re.search(r'\b(now|current|currently|at this moment)\b',q) else ('get_possession_summary',{'team':team})
        # Overall possession and pass-count retrieval share one canonical source.
        if not (passes and call[0]=='get_possession_summary'):calls.append(call)
    if re.search(r'\b(fastest|quickest|highest speed|top speed|speed ranking)\b',q):
        match=re.search(r'\b(?:top|fastest|rank)\s+(\d+)\b',q)
        limit=max(1,min(100,int(match[1]))) if match else 5 if re.search(r'\b(players|rankings|rank|ranking)\b',q) else 1
        calls.append(('get_team_speed_peaks',{'team':team}) if re.search(r'\b(teams?|sides?|home|away|left|right)\b',q) else ('get_fastest_players',{'limit':limit}))
    if re.search(r'\b(distance|ground covered|how far)\b',q) and not re.search(r'\b(pass|passes|transfer|transfers|ball)\b',q):
        calls.append(('get_distance_leaders',{'limit':5 if re.search(r'\b(players|rank)\b',q) else 1}))
    if re.search(r'\b(pressure|pressured|pressed|pressing)\b',q):
        if re.search(r'\b(now|current|currently|level)\b',q):calls.append(('get_pressure_level',{'timestamp':timestamp}))
        elif both and team is None:calls.extend(('get_pressure_peaks',{'team':side}) for side in ('left','right'))
        else:calls.append(('get_pressure_peaks',{'team':team}))
    if re.search(r'\b(formation|shape)\b',q) and not re.search(r'\b(width|depth|compactness|change|changed|in possession)\b',q):
        calls.extend(('get_formation_timeline',{'team':side}) for side in ([team] if team else ['left','right']))
    return list(dict((name+repr(args),(name,args)) for name,args in calls).values())


def retrieve_question(tools,question,timestamp=0):
    calls=semantic_plan(question,timestamp)
    if not calls:return None
    results=[dispatch(tools,name,args) for name,args in calls]
    facts=[f for r in results for f in r.facts]
    return QueryResult(question_type=results[0].question_type if len(results)==1 else 'match_summary',
        status='ok' if any(r.status=='ok' for r in results) else 'unknown',facts=facts,
        limitations=list(dict.fromkeys(x for r in results for x in r.limitations)),
        provenance=sorted({p for r in results for p in r.provenance}))
