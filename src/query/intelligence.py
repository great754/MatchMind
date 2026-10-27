"""Deterministic review catalogue; only existing grounded query tools supply facts."""
import hashlib
from .schema import Closed, Answer
from .answers import answer_result

class Insight(Closed):
    insight_id: str
    title: str
    classification: str
    timestamp: float | None = None
    actor_id: int | None = None
    event_id: str | None = None
    presentations: dict[str, Answer]

class MatchIntelligence(Closed):
    recording_id: str
    insights: list[Insight]
    limitations: list[str]

def build_intelligence(tools, recording_id):
    entries = [
        ('Completed observable transfers / pass share', 'INFERRED', tools.get_confirmed_pass_share()),
        ('Attributed control diagnostics', 'INFERRED', tools.get_attributed_control_summary()),
        ('Fastest tracked player', 'INFERRED', tools.get_fastest_players(limit=1)),
        ('Measured distance leader', 'INFERRED', tools.get_distance_leaders(limit=1)),
        ('Difficult completed transfer', 'HEURISTIC', tools.get_difficult_transfers(limit=1)),
        ('Passing connections', 'INFERRED', tools.get_transfer_network()),
        ('Geometric pressure observation', 'HEURISTIC', tools.get_pressure_peaks()),
    ]
    # Select the strongest repeated directed connection; no synthetic timestamp.
    network=entries[5][2]
    edges={}
    for fact in network.facts:
        if fact.record_id.startswith('edge:'):edges.setdefault(fact.record_id,[]).append(fact)
    repeated=[(next((f.value for f in facts if f.field=='transfer_count'),0),key,facts) for key,facts in edges.items()]
    repeated=[edge for edge in repeated if edge[0]>=2]
    chosen=sorted(repeated,key=lambda edge:(-edge[0],edge[1]))[:1]
    entries[5]=('Repeated passing connection','INFERRED',network.model_copy(update={'facts':chosen[0][2] if chosen else [],'status':'ok' if chosen else 'unknown'}))
    shots=tools.get_match_summary()
    shotfacts=[f for f in shots.facts if f.field=='trajectory_shot_candidates' and isinstance(f.value,(int,float)) and f.value>0]
    entries.append(('Trajectory shot candidates — not confirmed shots','HEURISTIC',shots.model_copy(update={'facts':shotfacts,'status':'ok' if shotfacts else 'unknown'})))
    insights=[]
    for title, classification, result in entries:
        if result.status != 'ok' or not result.facts: continue
        # Namespace presentation facts by recording, preserving canonical source locators.
        facts=[f.model_copy(update={'fact_id':hashlib.sha256((recording_id+':'+f.fact_id).encode()).hexdigest()[:24]}) for f in result.facts]
        result=result.model_copy(update={'facts':facts})
        times=[f.timestamp for f in facts if f.timestamp is not None]
        timestamp=min(times) if times else None
        actors=[f.value for f in facts if f.field in ('actor_id','sender_id','player_id') and isinstance(f.value,int)]
        events=[f.event_id for f in facts if f.event_id]
        insights.append(Insight(insight_id=hashlib.sha256((recording_id+title).encode()).hexdigest()[:20], title=title,
            classification=classification,timestamp=timestamp,actor_id=actors[0] if actors else None,
            event_id=events[0] if events else None,
            presentations={mode:answer_result(result,mode) for mode in ('beginner','fan','analyst')}))
    return MatchIntelligence(recording_id=recording_id,insights=insights,limitations=[
        'Post-match catalogue of existing canonical evidence; no provider computes statistics.',
        'Completed transfers are not official passes; pass share is not official possession.',
        'Pressure and difficulty are geometric heuristics, not intent or causal explanations.',
        'Unsupported formations and shot claims are omitted.'])
