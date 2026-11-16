"""Independent human-reference metrics; model candidates and unknown labels are not truth."""
import math
from scipy.optimize import linear_sum_assignment
import numpy as np


def iou(a,b):
    x=max(a[0],b[0]);y=max(a[1],b[1]);r=min(a[0]+a[2],b[0]+b[2]);bottom=min(a[1]+a[3],b[1]+b[3]);inter=max(0,r-x)*max(0,bottom-y)
    return inter/max(1e-12,a[2]*a[3]+b[2]*b[3]-inter)


def wilson(success,total):
    if total==0:return None
    z=1.96;p=success/total;d=1+z*z/total;center=(p+z*z/(2*total))/d;half=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/d
    return [max(0,center-half),min(1,center+half)]


def counts(tp,fp,fn):
    precision=tp/(tp+fp) if tp+fp else None;recall=tp/(tp+fn) if tp+fn else None
    return {'tp':tp,'fp':fp,'fn':fn,'precision':precision,'recall':recall,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,'precision_wilson95':wilson(tp,tp+fp),'recall_wilson95':wilson(tp,tp+fn)}


def evaluate(tasks,annotations,balls,events,protocol,players=None):
    lookup={t['task_id']:t for t in tasks};valid=[];seen=set()
    for row in annotations:
        if row.get('origin')!='human' or row.get('reviewed_from_source') is not True or row.get('model_assisted') is not False:raise ValueError('Independent source-reviewed human labels required')
        if row['task_id'] not in lookup:raise ValueError('Unknown review task')
        if row['task_id'] in seen:raise ValueError('Choose one reviewer or adjudicate duplicate task labels before evaluation')
        seen.add(row['task_id'])
        source=protocol.get('sources',{}).get(lookup[row['task_id']]['recording_id'],{})
        if not source or row.get('source_video_sha256')!=source.get('video_sha256'):raise ValueError('Human reference source hash mismatch')
        valid.append(row)
    ball_tp=ball_fp=ball_fn=known_ball=unknown_ball=0;pass_tp=pass_fp=pass_fn=windows=unknown_windows=0;seconds=0;identity_known=identity_correct=identity_unknown=0
    for label in valid:
        task=lookup[label['task_id']];name=task['recording_id']
        if task['kind']=='ball':
            if label['state']=='uncertain':unknown_ball+=1;continue
            b=balls[name][task['frame']];observed=b['directly_observed'];known_ball+=1
            if label['state']=='visible':
                match=observed and iou(b['image_bbox'],label['bbox'])>=protocol['ball_box_match_iou'];ball_tp+=bool(match);ball_fn+=not match;ball_fp+=bool(observed and not match)
            else:ball_fp+=bool(observed)
        elif task['kind']=='passes':
            if label['state']!='reviewed' or not label['exhaustive'] or any(p['outcome']=='uncertain' for p in label['passes']):unknown_windows+=1;continue
            windows+=1;seconds+=task['end']-task['start'];fps=protocol.get('sources',{}).get(name,{}).get('fps')
            # References contain frame times on the sealed source clock.
            fps=task['fps'] if 'fps' in task else fps
            if fps is None:raise ValueError('Reference task FPS required for pass evaluation')
            refs=[p for p in label['passes'] if p['outcome']=='completed'];pred=[e for e in events.get(name,[]) if e['status']=='estimated' and task['start_frame']<=e['start_frame'] and e['end_frame']<=task['end_frame']]
            costs=np.full((len(refs),len(pred)),1e6);tolerance=protocol['pass_endpoint_tolerance_seconds']
            for i,ref in enumerate(refs):
                for j,event in enumerate(pred):
                    a=abs(ref['release_frame']/fps-event['start']);b=abs(ref['receipt_frame']/fps-event['end'])
                    if max(a,b)<=tolerance:costs[i,j]=a+b
            matched=int(sum(costs[i,j]<1e6 for i,j in zip(*linear_sum_assignment(costs)))) if costs.size else 0
            pass_tp+=matched;pass_fp+=len(pred)-matched;pass_fn+=len(refs)-matched
        elif task['kind']=='identity':
            if label['state']=='uncertain':identity_unknown+=1;continue
            def match(frame,box):
                candidates=[p for p in (players or {}).get(name,{}).get(frame,[]) if iou(p['bbox'],box)>=.5]
                return candidates[0]['entity'] if len(candidates)==1 else None
            a=match(task['frame'],label['bbox']);b=match(task['other_frame'],label['other_bbox'])
            if a is None or b is None:identity_unknown+=1;continue
            identity_known+=1;identity_correct+=(a==b)==(label['state']=='same')
    return {'status':'awaiting_independent_human_labels' if not valid else 'partial_independent_reference_evaluation','independent_label_count':len(valid),'ball':{**counts(ball_tp,ball_fp,ball_fn),'reviewed_frames':known_ball,'unknown_visibility_frames':unknown_ball},'passes':{**counts(pass_tp,pass_fp,pass_fn),'exhaustively_reviewed_windows':windows,'unknown_or_incomplete_windows':unknown_windows,'reviewed_seconds':seconds,'false_positives_per_reviewed_minute':pass_fp*60/seconds if seconds else None},'identity':{'reviewed_matched_pairs':identity_known,'unknown_or_unmatched_pairs':identity_unknown,'correct_pairs':identity_correct,'pair_accuracy':identity_correct/identity_known if identity_known else None,'wilson95':wilson(identity_correct,identity_known),'definition':'Agreement of unique visible M9 identities with independently boxed source-frame pairs; unmatched/ambiguous pairs excluded.'},'limitations':['Uniform-time source tasks; metrics describe only reviewed samples/windows.','One reviewer and few examples do not establish generalization or eliminate label error.','Unknown visibility and incomplete windows are excluded, not negative labels.']}
