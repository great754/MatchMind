"""Directed observable-transfer network; never an official passing accuracy model."""
from collections import defaultdict
import math
from src.intelligence.schema import EventType as T


def transfer_network(events,start_seconds=None,end_seconds=None,team=None,long_transfer_m=25.) -> dict:
    """Aggregate confirmed receipts in (start, end], with no future outcome inclusion."""
    if team not in (None,'left','right') or not math.isfinite(long_transfer_m) or long_transfer_m<=0: raise ValueError('Invalid network team or distance threshold')
    if any(value is not None and not math.isfinite(value) for value in (start_seconds,end_seconds)): raise ValueError('Network window must be finite')
    if start_seconds is not None and end_seconds is not None and start_seconds>end_seconds: raise ValueError('Network window must be ordered')
    selected=[e for e in events if e.type==T.PASS_COMPLETED and e.receiver_id is not None and
        (start_seconds is None or e.timestamp>start_seconds) and (end_seconds is None or e.timestamp<=end_seconds) and (team is None or e.team==team)]
    edges=defaultdict(list);nodes=defaultdict(list)
    for e in selected:
        edges[(e.team,e.player_id,e.receiver_id)].append(e)
        if e.metrics.start_x_m is not None and e.metrics.start_y_m is not None:nodes[(e.team,e.player_id)].append([e.metrics.start_x_m,e.metrics.start_y_m])
        else:nodes[(e.team,e.player_id)]
        if e.metrics.end_x_m is not None and e.metrics.end_y_m is not None:nodes[(e.team,e.receiver_id)].append([e.metrics.end_x_m,e.metrics.end_y_m])
        else:nodes[(e.team,e.receiver_id)]
    result=[]
    for (side,sender,receiver),items in sorted(edges.items()):
        distances=[e.metrics.distance_m for e in items if e.metrics.distance_m is not None]
        progression=[e.metrics.forward_progression_m for e in items if e.metrics.forward_progression_m is not None]
        difficulty=[e.metrics.difficulty_score_0_100 for e in items if e.metrics.difficulty_score_0_100 is not None]
        mean=lambda values:sum(values)/len(values) if values else None
        result.append(dict(team=side,sender_id=sender,receiver_id=receiver,transfer_count=len(items),mean_distance_m=mean(distances),
            total_progression_m=sum(progression) if progression else None,mean_progression_m=mean(progression),
            mean_heuristic_difficulty=mean(difficulty),distance_samples=len(distances),progression_samples=len(progression),difficulty_samples=len(difficulty),
            long_transfer_count=sum(d>=long_transfer_m for d in distances),
            bas_supported_pass_count=sum(e.validation_state=='bas_time_actor_match' for e in items)))
    node_list=[]
    for (side,pid),points in sorted(nodes.items()):
        incoming=sum(e['transfer_count'] for e in result if e['team']==side and e['receiver_id']==pid)
        outgoing=sum(e['transfer_count'] for e in result if e['team']==side and e['sender_id']==pid)
        node_list.append(dict(team=side,player_id=pid,label=f'Player {pid}',incoming_transfers=incoming,outgoing_transfers=outgoing,
            endpoint_samples=len(points),mean_x_m=sum(p[0] for p in points)/len(points) if points else None,
            mean_y_m=sum(p[1] for p in points)/len(points) if points else None))
    return dict(schema_version='1.0',name='Transfer Network',start_seconds=start_seconds,end_seconds=end_seconds,
        definition='Directed completed observable transfers, published at confirmed receipt. BAS support checks release/sender only.',
        team=team,transfer_count=len(selected),nodes=node_list,edges=result)
