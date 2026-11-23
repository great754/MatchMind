"""Frozen, common center localization metrics; confidence is never ground truth."""
import math
import numpy as np
from scipy.optimize import linear_sum_assignment
from src.evaluation.cv_m19 import counts,iou


def center(box):
    return [box[0]+box[2]/2,box[1]+box[3]/2]


def match(reference,predictions,tolerance):
    """Maximum cardinality first, minimum total distance second, one-to-one."""
    if not reference or not predictions:return []
    distance=np.linalg.norm(np.asarray(reference)[:,None,:]-np.asarray(predictions)[None,:,:],axis=-1)
    penalty=(min(len(reference),len(predictions))+1)*(tolerance+1)
    cost=np.where(distance<=tolerance,distance,penalty)
    return [(int(i),int(j),float(distance[i,j])) for i,j in zip(*linear_sum_assignment(cost)) if distance[i,j]<=tolerance]


def score(references,rows,sources,fraction=.01,native_iou=False,multi=False):
    tp=fp=fn=known=unknown=visible=0;errors=[];selected_errors=[];candidate={k:0 for k in [1,3,5]}
    for ref in references:
        if ref['state']=='uncertain':unknown+=1;continue
        known+=1;visible+=ref['state']=='visible';row=rows[ref['recording_id']][ref['frame']]
        prediction=row['selected'];preds=([prediction] if prediction else []) if not multi else [c for c in row['candidates'] if c['confidence']>=.7]
        truth=[center(ref['bbox'])] if ref['state']=='visible' else []
        if truth and prediction:selected_errors.append(float(np.linalg.norm(np.asarray(truth[0])-prediction['center'])))
        source=sources[ref['recording_id']];tol=math.hypot(source['width'],source['height'])*fraction
        if native_iou:
            matches=[(0,0,float(np.linalg.norm(np.array(truth[0])-prediction['center'])))] if truth and prediction and prediction.get('bbox') and iou(ref['bbox'],prediction['bbox'])>=.25 else []
        else:matches=match(truth,[p['center'] for p in preds],tol)
        tp+=len(matches);fp+=len(preds)-len(matches);fn+=len(truth)-len(matches);errors.extend(m[2] for m in matches)
        for k in candidate:
            candidate[k]+=bool(truth and match(truth,[c['center'] for c in row['candidates'][:k]],tol))
    return {**counts(tp,fp,fn),'reviewed_frames':known,'visible_frames':visible,'uncertain_excluded':unknown,
        'false_detections_per_reviewed_frame':fp/known if known else None,
        'localization_error_pixels':{'mean':float(np.mean(errors)),'median':float(np.median(errors)),'p95':float(np.quantile(errors,.95))} if errors else None,
        'selected_visible_error_pixels_including_wrong':{'mean':float(np.mean(selected_errors)),'median':float(np.median(selected_errors)),'p95':float(np.quantile(selected_errors,.95))} if selected_errors else None,
        'candidate_recall':{str(k):{'hits':n,'visible':visible,'recall':n/visible if visible else None} for k,n in candidate.items()},
        'criterion':'native_IoU_0.25' if native_iou else f'center_diagonal_fraction_{fraction}',
        'selection':'all_strict_peaks_at_least_0.7' if multi else 'selected_observation_only',
        'interval_caution':'Wilson intervals describe sampled frames; temporal correlation and model-assisted selection prevent independent-test/generalization claims.'}


def baseline(row):
    candidates=[{'center':center(c['bbox']),'confidence':c['confidence'],'bbox':c['bbox']} for c in row['candidates']]
    candidates.sort(key=lambda c:-c['confidence'])
    selected=({'center':row['image_position'],'confidence':row['detection_confidence'],'bbox':row['image_bbox']}
        if row['directly_observed'] and row['image_position'] is not None else None)
    return {'frame':row['frame'],'candidates':candidates,'selected':selected}
