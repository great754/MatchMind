"""One-to-one timestamp matching; reference labels never create detections."""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def validate_events(predictions, references, fps, tolerance_seconds=1.0, actor_field=None):
    predictions = predictions.to_dict('records') if isinstance(predictions,pd.DataFrame) else predictions
    truth = list(references)
    matches = []
    if predictions and truth:
        dummy = (tolerance_seconds+1)*(len(predictions)+len(truth)+1)
        cost = np.full((len(predictions),len(truth)+len(predictions)),dummy)
        for i,p in enumerate(predictions):
            for j,t in enumerate(truth):
                delta = abs(p['start_frame']-t['frame'])/fps
                actor_ok = actor_field is None or p.get(actor_field) == t['player_id']
                cost[i,j] = delta if delta <= tolerance_seconds and actor_ok else dummy*1000
        pi,ti = linear_sum_assignment(cost)
        matches = [(int(i),int(j)) for i,j in zip(pi,ti) if j < len(truth) and cost[i,j]<dummy]
    p_match = {i:j for i,j in matches}
    t_match = {j:i for i,j in matches}
    rows = []
    for i,p in enumerate(predictions):
        t = truth[p_match[i]] if i in p_match else None
        rows.append({'prediction_index':i,'prediction_frame':p['start_frame'],
                     'annotation_id':t['annotation_id'] if t else None,
                     'annotation_frame':t['frame'] if t else None,
                     'annotation_label':t['label'] if t else None,
                     'timing_error_seconds':(p['start_frame']-t['frame'])/fps if t else None,
                     'match_status':'matched' if t else ('false_positive' if truth else 'unvalidated')})
    for j,t in enumerate(truth):
        if j not in t_match:
            rows.append({'prediction_index':None,'prediction_frame':None,'annotation_id':t['annotation_id'],
                         'annotation_frame':t['frame'],'annotation_label':t['label'],
                         'timing_error_seconds':None,'match_status':'false_negative'})
    tp = len(matches)
    precision = tp/len(predictions) if truth and predictions else (0.0 if truth else None)
    recall = tp/len(truth) if truth else None
    f1 = 2*precision*recall/(precision+recall) if truth and precision+recall else (0.0 if truth else None)
    summary = dict(status='evaluated' if truth else 'no_reference_events',
                   detections=len(predictions),reference_events=len(truth),matched=tp,
                   false_positives=len(predictions)-tp if truth else None,
                   false_negatives=len(truth)-tp if truth else None,
                   precision=precision,recall=recall,f1=f1,
                   tolerance_seconds=tolerance_seconds,actor_required=actor_field is not None,
                   independent_validation=False,
                   caveat='Ball paths are reconstructed from BAS event anchors; this is a consistency check, not independent detector accuracy.')
    columns = ['prediction_index','prediction_frame','annotation_id','annotation_frame','annotation_label',
               'timing_error_seconds','match_status']
    return summary,pd.DataFrame(rows,columns=columns)
