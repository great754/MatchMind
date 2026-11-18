"""Reuse sealed M19 human-reference tasks for M19/M20; no labels required for engineering."""
import argparse,json
from pathlib import Path
import numpy as np
from src.serving.cv import CVReview,digest
from src.serving.ball import BallReview
from src.serving.cv_events import CVEventReview
from src.serving.cv_review_labels import HumanReviewStore,HumanAnnotation
from src.evaluation.cv_m19 import evaluate,iou
from src.vision.events.passes import visible_players
from scripts.run_cv_m19 import compact_worlds


def localization(tasks,labels,balls,protocol):
    lookup={t['task_id']:t for t in tasks};errors=[]
    for row in labels:
        t=lookup[row['task_id']]
        if t['kind']!='ball' or row['state']!='visible':continue
        predicted=balls[t['recording_id']][t['frame']]
        if not predicted['directly_observed'] or iou(predicted['image_bbox'],row['bbox'])<protocol['ball_box_match_iou']:continue
        x,y,w,h=row['bbox'];errors.append(float(np.linalg.norm(np.asarray(predicted['image_position'])-[x+w/2,y+h/2])))
    return {'matched_human_boxes':len(errors),'median_pixel_error':float(np.median(errors)) if errors else None,'mean_pixel_error':float(np.mean(errors)) if errors else None,'p90_pixel_error':float(np.quantile(errors,.9)) if errors else None,'definition':'Center error only for selected direct observations matched to independently boxed source balls.'}


def run(root,label_path=None,reviewer=None):
    root=Path(root);store=HumanReviewStore();dataset=store.dataset;protocol=dict(dataset['protocol']);protocol['sources']=dataset['sources'];labels=[]
    if label_path:
        document=json.loads(Path(label_path).read_text())
        if document['dataset_fingerprint']!=dataset['fingerprint']:raise ValueError('Reference task fingerprint changed')
        for row in document['annotations']:
            if reviewer is not None and row['reviewer_id']!=reviewer:continue
            annotation=HumanAnnotation.model_validate({k:v for k,v in row.items() if k!='source_video_sha256'})
            task=store.tasks.get(row['task_id'])
            if task is None:raise ValueError('Unknown source-review task')
            store.validate(task['recording_id'],annotation);labels.append(row)
    players={};ball_sets={'m19':{},'m20':{}};event_sets={'m19':{},'m20':{}}
    for name in dataset['sources']:
        review=CVReview(Path('outputs/world_model')/(name+'_m9_ready'))
        for version,base in [('m19',Path('outputs/m19/run_v3')/name),('m20',root/name)]:
            ball_path=base/('ball_m16' if version=='m19' else 'ball_m20');review.ball_review=BallReview(ball_path,review.manifest)
            # Events pin both the ball source and immutable M9 player evidence.
            event_path=base/('passes_m16' if version=='m19' else 'passes_m20');layer=CVEventReview(event_path,review)
            ball_sets[version][name]=[review.ball_review.frame(i).model_dump() for i in range(review.manifest['frames'])];event_sets[version][name]=layer.events
        players[name]={}
        if labels:
            worlds=compact_worlds(review.data)
            for task in dataset['tasks']:
                if task['recording_id']==name and task['kind']=='identity':
                    for frame in [task['frame'],task['other_frame']]:players[name][frame]=list(visible_players(worlds[frame]).values())
    report={'dataset_fingerprint':dataset['fingerprint'],'annotation_file_supplied':label_path is not None,'independent_accuracy_acceptance':'pending_human_review','annotations_used':len(labels),'existing_saved_annotations_preserved':len(store.labels()),'variants':{}}
    for version in ['m19','m20']:
        result=evaluate(dataset['tasks'],labels,ball_sets[version],event_sets[version],protocol,players)
        result['localization']=localization(dataset['tasks'],labels,ball_sets[version],protocol)
        result.update(independent_reacquisition_success=None,incorrect_ball_associations=None,temporal_ball_reference_limitation='Sparse visibility/box tasks do not provide exhaustive ball identity and reacquisition truth. Temporal ball reference tracks are still required.')
        report['variants'][version]=result
    (root/'independent_evaluation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--labels');p.add_argument('--reviewer');a=p.parse_args();run(a.root,a.labels,a.reviewer)
