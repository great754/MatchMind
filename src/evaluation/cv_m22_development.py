"""Model-assisted human development references; never an independent test API."""
import math
import numpy as np
from .cv_m19 import counts,iou


def ball_metrics(tasks,annotations,balls,sources):
    lookup={t['task_id']:t for t in tasks}
    if len(lookup)!=len(tasks):raise ValueError('Duplicate development task')
    seen=set();tp=fp=fn=tn=known=unknown=0;errors=[]
    for row in annotations:
        if row.get('origin')!='human' or row.get('reviewed_from_source') is not True or row.get('model_assisted') is not True:
            raise ValueError('Explicitly source-reviewed, model-assisted human development labels required')
        if row['task_id'] in seen or row['task_id'] not in lookup:raise ValueError('Duplicate or unknown development reference')
        seen.add(row['task_id']);task=lookup[row['task_id']];name=task['recording_id'];source=sources[name];frame=task['frame']
        timestamp=row.get('timestamp')
        if row.get('source_video_sha256')!=source['video_sha256'] or row.get('frame')!=frame or not isinstance(timestamp,(int,float)) or not math.isfinite(timestamp) or abs(timestamp-frame/source['fps'])>1e-6:
            raise ValueError('Development reference source/clock mismatch')
        if len(balls[name])!=source['frames'] or not 0<=frame<source['frames']:raise ValueError('Prediction source clock mismatch')
        state=row['state']
        if state not in ['visible','not_visible','uncertain']:raise ValueError('Unknown development visibility')
        if state=='uncertain':unknown+=1;continue
        box=row.get('bbox')
        if state=='visible':
            if box is None or len(box)!=4 or any(not math.isfinite(v) for v in box) or min(box[:2])<0 or min(box[2:])<=0 or box[0]+box[2]>source['width'] or box[1]+box[3]>source['height']:
                raise ValueError('Visible reference requires a valid source box')
        elif box is not None:raise ValueError('Absent reference cannot have a ball box')
        predicted=balls[name][frame];observed=predicted['directly_observed'];known+=1
        if observed and predicted.get('image_bbox') is None:raise ValueError('Observed prediction requires source pixels')
        if state=='visible':
            matched=bool(observed and iou(predicted['image_bbox'],box)>=.25)
            tp+=matched;fn+=not matched;fp+=bool(observed and not matched)
            if matched:
                x,y,w,h=predicted['image_bbox'];a,b,bw,bh=box
                errors.append(math.hypot(x+w/2-a-bw/2,y+h/2-b-bh/2))
        else:fp+=bool(observed);tn+=not observed
    return {'reference_role':'development_only','model_assisted':True,'independent_final_test':False,
        'matching_iou':.25,'annotations':len(seen),
        'ball':{**counts(tp,fp,fn),'tn':tn,'reviewed_frames':known,'unknown_visibility_frames':unknown},
        'localization':{'matched_boxes':len(errors),'median_pixel_error':float(np.median(errors)) if errors else None,
            'mean_pixel_error':float(np.mean(errors)) if errors else None,'p90_pixel_error':float(np.quantile(errors,.9)) if errors else None},
        'limits':['Human-confirmed proposals and source corrections are development evidence, not independent final-test truth.',
            'Active-learning selection and model-assisted boxes can bias comparison; source-excluded fits do not remove prior development influence.']}
