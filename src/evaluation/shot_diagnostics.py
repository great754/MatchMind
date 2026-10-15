"""Post-hoc shot error evidence. Context labels never feed the prediction path."""
from collections import Counter
from pathlib import Path
import html
import json
import numpy as np
import pandas as pd
from src.analytics.shot_evidence import trace_shot_frames
from src.analytics.motion import segments


def nearest_action(events,frame,fps):
    if not events:return None
    e=min(events,key=lambda e:(abs(e['frame']-frame),e['annotation_id']))
    return dict(annotation_id=e['annotation_id'],label=e['label'],seconds=e['frame']/fps,offset_seconds=(e['frame']-frame)/fps)


def analyze_shot_errors(recording,result,config,recording_id,trace=None):
    trace=trace if trace is not None else trace_shot_frames(result['possession'],result['ball'],config)
    flags=np.array([r['eligible'] for r in trace]);runs=list(segments(flags))
    rows=[];predictions=result['shots'].to_dict('records');refs={e['annotation_id']:e for e in recording.events if e['label']=='SHOT'}
    for match in result['shot_matches'].to_dict('records'):
        status=match['match_status'];prediction=predictions[int(match['prediction_index'])] if pd.notna(match['prediction_index']) else None
        ref=refs.get(int(match['annotation_id'])) if pd.notna(match['annotation_id']) else None
        target=int(prediction['start_frame']) if prediction else int(ref['frame'])
        lo=max(0,target-round(config.validation_tolerance_seconds*config.fps));hi=min(len(trace),target+round(config.validation_tolerance_seconds*config.fps)+1)
        window=trace[lo:hi]
        # The most advanced gate evidence within the matching window, then nearest reference.
        chosen=trace[target] if prediction else min(window,key=lambda r:(len(r['failures']),abs(r['frame']-target),r['frame']))
        failure=chosen['primary_gate'];hypothesis=None
        if prediction:
            failure='MATCHED' if status=='matched' else 'UNMATCHED_CANDIDATE' if status=='false_positive' else 'NO_REFERENCE_VALIDATION'
        elif chosen['eligible']:
            active=[(a,b) for a,b in runs if a<=chosen['frame']<b]
            if active and active[0][1]-active[0][0]<config.shot_persistence_frames:failure='INSUFFICIENT_PERSISTENCE'
            elif any(p['start_frame']<=chosen['frame']<=p['end_frame'] for p in predictions):failure='EARLIER_CANDIDATE_WINDOW_OR_TIMING'
            else:failure='ELIGIBLE_BUT_SUPPRESSED_WINDOW'
        action=nearest_action(recording.events,target,config.fps)
        if status=='false_positive':
            label=action['label'] if action and abs(action['offset_seconds'])<=2 else None
            hypothesis={'CROSS':'CROSS_CONTEXT','HIGH PASS':'LONG_BALL_CONTEXT','PASS':'FORWARD_PASS_CONTEXT','OUT':'OUT_CONTEXT','FREE KICK':'RESTART_CONTEXT'}.get(label,'UNKNOWN')
            if chosen['direction_source']=='velocity_fallback':hypothesis='DIRECTION_UNCONFIRMED'
        ball=result['ball'];sf=chosen['frame'];end=int(prediction['end_frame']) if prediction else min(len(trace)-1,hi-1)
        recent_id=chosen['recent_possessing_player'];slot=np.flatnonzero(recording.player_ids==recent_id) if recent_id is not None else []
        nearby=[]
        if chosen['start_x'] is not None and chosen['start_y'] is not None:
            origin=np.array([chosen['start_x'],chosen['start_y']])
            for pid,team,pos in zip(recording.player_ids,recording.player_teams,recording.positions[sf]):
                if np.isfinite(pos).all() and np.linalg.norm(pos-origin)<=15:
                    nearby.append(dict(actor_id=int(pid),team=team,x_m=float(pos[0]),y_m=float(pos[1])))
        row=dict(recording_id=recording_id,half=recording.half,match_status=status,
            candidate_id=int(prediction['shot_id']) if prediction else None,BAS_event_id=ref['annotation_id'] if ref else None,
            candidate_time=prediction['start_seconds'] if prediction else None,reference_time=ref['seconds'] if ref else None,
            timing_error=float(match['timing_error_seconds']) if pd.notna(match['timing_error_seconds']) else None,
            inspect_time=target/config.fps,evidence_time=sf/config.fps,evidence_frame=sf,
            **{k:v for k,v in chosen.items() if k not in ('frame','primary_gate')},
            candidate_duration=(end-target)/config.fps if prediction else None,candidate_frame_count=end-target+1 if prediction else None,
            failure_reason=failure,diagnostic_hypothesis=hypothesis,nearest_bas_action=action,
            reference_team=ref.get('team') if ref else None,reference_player=ref.get('player_id') if ref else None,
            reference_frame_gates=trace[target]['failures'] if ref else None,
            gate_failure_counts=dict(Counter(g for r in window for g in r['failures'])),
            ball_data_quality=dict(position_source='BAS_event_anchored_interpolation',height_available=False,
                observed_position=bool(ball.observed[sf]),usable_position=bool(ball.valid[sf]),raw_speed_flag=bool(ball.raw_speed_flag[sf]),
                rejected_speed_flag=bool(ball.rejected_speed_flag[sf]),source_status=int(recording.ball_status[sf])),
            nearby_players=nearby,recent_possessor_position=recording.positions[sf,int(slot[0])].tolist() if len(slot) and np.isfinite(recording.positions[sf,int(slot[0])]).all() else None,
            interpretation='Post-hoc consistency evidence; FP hypotheses describe nearby BAS context, not verified intent. FN gate is the best-frame failure within the timing window, not a causal explanation.')
        rows.append(row)
    return rows


def diagnostic_svg(row):
    """Static pitch debug drawing, never another detector."""
    x,y=row['start_x'],row['start_y'];parts=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="-5 -5 115 78" role="img" aria-label="Shot geometry debug pitch"><rect x="0" y="0" width="105" height="68" fill="#245236" stroke="white" stroke-width=".3"/><path d="M52.5 0V68 M0 30.34V37.66 M105 30.34V37.66" stroke="white" stroke-width=".8"/>']
    if x is not None and y is not None:
        for p in row['nearby_players']:
            color='#4695ff' if p['team']=='right' else '#ff6363'
            parts.append(f'<circle cx="{p["x_m"]}" cy="{p["y_m"]}" r=".7" fill="{color}"/>')
        if row['goal_x'] is not None and row['projected_goal_crossing_y'] is not None:
            # Clamp only drawing bounds; exact projection stays in the diagnostic JSON.
            gy=float(np.clip(row['projected_goal_crossing_y'],-5,73))
            parts.append(f'<path d="M{x} {y}L{row["goal_x"]} {gy}" stroke="#ffdf35" stroke-width=".5" stroke-dasharray="1 1"/>')
        if row['vx'] is not None and row['vy'] is not None:
            parts.append(f'<path d="M{x} {y}l{row["vx"]*.3} {row["vy"]*.3}" stroke="#00ffff" stroke-width=".8"/>')
        parts.append(f'<circle cx="{x}" cy="{y}" r=".8" fill="white"/>')
        if row['recent_possessor_position']:
            px,py=row['recent_possessor_position'];parts.append(f'<circle cx="{px}" cy="{py}" r="1.1" fill="none" stroke="white" stroke-width=".4"/>')
    return ''.join(parts)+'</svg>'


def write_shot_diagnostics(rows,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    payload=dict(schema_version='1.0',items=rows,
        counts=dict(Counter(r['match_status'] for r in rows)),
        false_negative_gate_counts=dict(Counter(r['failure_reason'] for r in rows if r['match_status']=='false_negative')),
        false_positive_hypotheses=dict(Counter(r['diagnostic_hypothesis'] for r in rows if r['match_status']=='false_positive')),
        limitations=['BAS checks are not independent; the supplied ball path uses BAS anchors.','2D straight-line extrapolation cannot establish shot intent, ball height or goalkeeper saves.','FP context labels are post-hoc hypotheses, not detector inputs or verified causes.'])
    (output/'shot_error_analysis.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    flat=[]
    for row in rows:flat.append({k:json.dumps(v,allow_nan=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})
    pd.DataFrame(flat).to_csv(output/'shot_error_analysis.csv',index=False)
    return payload


def report_section(payload):
    parts=['<h2>Shot error analysis</h2><p>White: ball; cyan: 0.3-second velocity vector; yellow: projected goal crossing; white ring: recent possessor. Red/blue are team sides. Full-half video may be unavailable; inspection links still seek the pitch. Geometry and BAS context do not establish intent.</p>']
    parts.append('<pre>'+html.escape(json.dumps({k:payload[k] for k in ('counts','false_negative_gate_counts','false_positive_hypotheses')},indent=2))+'</pre>')
    for r in payload['items']:
        key=html.escape(r['recording_id']);timestamp=r['inspect_time'];label=r['match_status']
        parts.append(f'<details><summary>{html.escape(label)} · {key} · {timestamp:.2f}s · {html.escape(r["failure_reason"])}</summary>')
        parts.append(f'<a href="per_match/{key}/report.html?t={timestamp:.4f}#pitch">Inspect event at {timestamp:.2f}s</a>')
        parts.append('<div style="max-width:600px">'+diagnostic_svg(r)+'</div><pre>'+html.escape(json.dumps(r,indent=2))+'</pre></details>')
    return ''.join(parts)
