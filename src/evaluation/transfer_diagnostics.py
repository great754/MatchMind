"""Transfer discrepancies: observable movements versus annotated pass-like actions."""
from collections import Counter
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.analytics.passes import detect_passes
from .event_evaluation import PASS_CLASSES,evaluate_transfers
from .shot_diagnostics import nearest_action


def analyze_transfer_errors(recording,result,config,recording_id):
    audit=[];predicted=detect_passes(result['possession'],result['ball'],config,audit)
    if predicted.start_frame.tolist()!=result['passes'].start_frame.tolist():raise AssertionError('Transfer baseline did not reproduce')
    refs={e['annotation_id']:e for e in recording.events if e['label'] in PASS_CLASSES}
    predictions=result['passes'].to_dict('records');rows=[]
    for m in result['pass_matches'].to_dict('records'):
        if m['match_status']=='matched':continue
        p=predictions[int(m['prediction_index'])] if pd.notna(m['prediction_index']) else None
        ref=refs[int(m['annotation_id'])] if pd.notna(m['annotation_id']) else None
        frame=int(p['start_frame']) if p else int(ref['frame']);tol=config.validation_tolerance_seconds*config.fps
        lo=max(0,frame-int(tol));hi=min(len(result['possession']),frame+int(tol)+1)
        window=result['possession'].iloc[lo:hi]
        nearby=[a for a in audit if abs(a['start_frame']-frame)<=tol]
        same_sender=[a for a in nearby if ref and a['sender_id']==ref['player_id']]
        candidates=[r for r in predictions if abs(r['start_frame']-frame)<=tol]
        if p:
            nearrefs=[r for r in refs.values() if abs(r['frame']-frame)<=tol]
            reason='SENDER_MISMATCH' if any(r['player_id']!=p['sender_id'] for r in nearrefs) else 'ANNOTATION_TIMING_OR_SEMANTICS_UNSUPPORTED'
        elif any(c['sender_id']==ref['player_id'] for c in candidates):reason='ONE_TO_ONE_MATCH_COMPETITION'
        elif candidates:reason='SENDER_MISMATCH'
        elif same_sender:reason=min(same_sender,key=lambda a:abs(a['start_frame']-frame))['reason']
        elif window.state.eq('unknown').all():reason='BALL_RECONSTRUCTION_GAP'
        elif not window.possessing_player_id.eq(ref['player_id']).any():reason='POSSESSION_NEVER_ACQUIRED'
        else:reason='NO_RELEASED_INDEPENDENT_FLIGHT_IN_WINDOW'
        if reason=='ACCEPTED':reason='ACCEPTED_OUTSIDE_ASSIGNMENT_OR_TIMING'
        rows.append(dict(recording_id=recording_id,half=recording.half,match_status=m['match_status'],
            candidate_id=int(p['transfer_id']) if p else None,BAS_event_id=ref['annotation_id'] if ref else None,
            BAS_class=ref['label'] if ref else None,inspect_time=frame/config.fps,
            sender_id=int(p['sender_id']) if p else ref['player_id'],failure_reason=reason,
            nearest_bas_action=nearest_action(recording.events,frame,config.fps),
            possession_state_counts=window.state.value_counts().to_dict(),
            nearby_candidate_count=len(candidates),flight_audit=same_sender if ref else nearby,
            candidate_outcome=p['outcome'] if p else None,
            interpretation='Diagnostic rule evidence, not verified soccer intent. Unknown possession can reflect tracking availability or heuristic control failure.'))
    return rows,evaluate_transfers(result['passes'],list(refs.values()),config.fps,config.validation_tolerance_seconds)


def write_transfer_diagnostics(rows,class_results,output):
    from .aggregate import aggregate_transfer_classes
    payload=dict(schema_version='1.0',items=rows,counts=dict(Counter(r['match_status'] for r in rows)),
        missed_reference_reasons=dict(Counter(r['failure_reason'] for r in rows if r['match_status']=='false_negative')),
        unmatched_candidate_reasons=dict(Counter(r['failure_reason'] for r in rows if r['match_status']=='false_positive')),
        by_bas_class=aggregate_transfer_classes(class_results),
        limitations=['Class precision/FP/F1 are undefined because predictions do not carry BAS action classes.','No thresholds changed; BAS timing/sender support does not establish pass intent or outcome.'])
    output=Path(output);(output/'transfer_error_analysis.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    pd.DataFrame([{k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows]).to_csv(output/'transfer_error_analysis.csv',index=False)
    return payload
