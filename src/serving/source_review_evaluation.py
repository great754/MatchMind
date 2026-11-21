"""Validate a completed human-review snapshot against the exact served predictions."""
import json
from pathlib import Path
from src.serving.cv import digest
from src.serving.ball import BallReview
from src.vision.football22.review import DevelopmentReviewStore,DevelopmentLabel
from src.vision.events.passes import signature
from src.evaluation.cv_m22_development import ball_metrics


def completed_review(entry,review,analysis):
    path=Path(entry['path'])
    if digest(path)!=entry['sha256']:raise ValueError('Completed review report changed')
    report=json.loads(path.read_text());name=review.manifest['recording_id']
    if report['schema_version']!='cv-source-review-evaluation-v1' or report['recording_id']!=name or report['independent_final_test'] is not False or report['model_assisted'] is not True:
        raise ValueError('Completed review role/source mismatch')
    if any(report['source'][k]!=review.manifest[k] for k in ['recording_id','video_sha256','frames_sha256','frames','fps','width','height']):
        raise ValueError('Completed review belongs to another source')
    if digest(Path(analysis)/'report.json')!=report['analysis_report_sha256'] or review.ball_review.manifest['perception_records_sha256']!=report['perception_records_sha256']:
        raise ValueError('Completed review inference provenance mismatch')
    snapshot=Path(report['snapshot_root'])
    if snapshot.resolve()!=(path.parent/'snapshot').resolve():raise ValueError('Unexpected completed-review snapshot')
    if digest(snapshot/'tasks.json')!=report['task_manifest_sha256'] or digest(snapshot/'labels.json')!=report['annotation_snapshot_sha256']:
        raise ValueError('Completed review annotations changed')
    store=DevelopmentReviewStore(snapshot/'tasks.json',snapshot,approved_sources={name});labels=store.labels()
    if len({r['reviewer_id'] for r in labels})!=1 or len(labels)!=len(store.tasks) or {r['task_id'] for r in labels}!=set(store.tasks) or report['completed_tasks']!=len(labels):
        raise ValueError('Completed review is incomplete or has unadjudicated reviewers')
    for label in labels:
        valid=store.validate(name,DevelopmentLabel.model_validate({k:v for k,v in label.items() if k in DevelopmentLabel.model_fields}))
        if signature(valid)!=signature(label):raise ValueError('Completed annotation source or derived fields changed')
    if store.dataset['fingerprint']!=report['dataset_fingerprint']:raise ValueError('Completed task fingerprint mismatch')
    systems=[]
    for method in ['full_m23','m21_temporal_offline']:
        layer=review.ball_review if method=='full_m23' else BallReview(Path(analysis)/name/method,review.manifest)
        expected=report['systems'][method]
        if layer.manifest['ball_frames_sha256']!=expected['ball_frames_sha256'] or layer.manifest['detector_sha256']!=expected['model_set_sha256']:
            raise ValueError('Completed review prediction revision mismatch')
        balls=[layer.frame(i).model_dump() for i in range(review.manifest['frames'])]
        measured=ball_metrics(list(store.tasks.values()),labels,{name:balls},store.dataset['sources'])
        if signature(measured)!=signature(expected['metrics']):raise ValueError('Completed review metrics do not reproduce')
        systems.append({'id':method,'ball':measured['ball']})
    try:stale=digest(report['live_labels_path'])!=report['annotation_snapshot_sha256']
    except OSError:stale=True
    return {'completed_tasks':len(labels),'visible_frames':sum(r['state']=='visible' for r in labels),
        'uncertain_frames':sum(r['state']=='uncertain' for r in labels),'absent_frames':sum(r['state']=='not_visible' for r in labels),
        'rejected_regions':sum(len(r['hard_negative_regions']) for r in labels),'systems':systems,
        'failure_counts':report['failure_counts'],'evaluated_at_utc':report['evaluated_at_utc'],
        'annotation_snapshot_sha256':report['annotation_snapshot_sha256'],'labels_updated_since_snapshot':stale,
        'reference_role':'development_only','model_assisted':True,'independent_final_test':False}
