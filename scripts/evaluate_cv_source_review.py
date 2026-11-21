"""Freeze actual completed source reviews and measure unchanged saved predictions."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
from src.serving.cv import CVReview,digest
from src.serving.ball import BallReview
from src.vision.football22.review import DevelopmentReviewStore,DevelopmentLabel
from src.vision.events.passes import signature
from src.vision.intelligent23.inputs import write
from src.evaluation.cv_m22_development import ball_metrics
from src.evaluation.cv_m19 import iou


def failure_traces(tasks,labels,records,balls):
    lookup={t['task_id']:t for t in tasks};traces=[]
    for label in labels:
        task=lookup[label['task_id']];frame=task['frame'];selected=balls[frame]
        if label['state']=='uncertain':continue
        match=label['state']=='visible' and selected['directly_observed'] and iou(selected['image_bbox'],label['bbox'])>=.25
        candidates=records[frame]['methods']['full_m23']['candidates']
        matching=[c for c in candidates if label['state']=='visible' and iou(c['bbox'],label['bbox'])>=.25]
        eligible=[c for c in matching if c['observation_eligible']]
        best=max((c['confidence'] for c in matching),default=None)
        if match:reason='matched'
        elif label['state']=='not_visible':reason='false_selection_on_absent_frame' if selected['directly_observed'] else 'correct_absence'
        elif not matching:reason='no_matching_raw_candidate'
        elif best<.35:reason='matching_candidate_below_fixed_visual_gate'
        elif not eligible:reason='matching_candidate_rejected_by_verification'
        elif all(c['origin']=='complement' for c in eligible):reason='complement_only_gap_policy_did_not_admit_match'
        else:reason='eligible_primary_not_selected_by_offline_association'
        traces.append({'frame':frame,'selection_kind':task['selection_kind'],'reference_state':label['state'],
            'reason':reason,'selected_iou':iou(selected['image_bbox'],label['bbox']) if label['state']=='visible' and selected['directly_observed'] else None,
            'matching_raw_candidates':len(matching),'matching_eligible_candidates':len(eligible),
            'matching_eligible_origins':sorted({c['origin'] for c in eligible}),
            'best_matching_raw_confidence':best})
    return traces


def run(analysis,tasks,labels,output):
    root=Path(output)
    if root.exists():raise ValueError('Completed evaluation snapshots are immutable; choose a fresh output')
    root.mkdir(parents=True);snapshot=root/'snapshot';snapshot.mkdir(mode=0o700)
    for source,name in [(Path(tasks),'tasks.json'),(Path(labels),'labels.json')]:
        payload=source.read_bytes();(snapshot/name).write_bytes(payload)
        if digest(source)!=digest(snapshot/name):raise ValueError('Review changed during snapshot; retry in a fresh directory')
        (snapshot/name).chmod(0o400)
    store=DevelopmentReviewStore(snapshot/'tasks.json',snapshot,approved_sources={'giroud'});rows=store.labels()
    if len({r['reviewer_id'] for r in rows})!=1:raise ValueError('Select or adjudicate one reviewer before evaluation')
    if len(rows)!=len(store.tasks) or {r['task_id'] for r in rows}!=set(store.tasks):raise ValueError('Complete every review task before evaluating')
    for row in rows:
        validated=store.validate('giroud',DevelopmentLabel.model_validate({k:v for k,v in row.items() if k in DevelopmentLabel.model_fields}))
        if signature(validated)!=signature(row):raise ValueError('Stored label source, clock or derived fields differ from validation')
    analysis=Path(analysis);original=json.loads((analysis/'report.json').read_text());review=CVReview(original['world_directory']);meta=review.manifest
    for key in ['recording_id','video_sha256','frames','fps','width','height']:
        if meta[key]!=store.dataset['sources']['giroud'][key]:raise ValueError('Review and prediction source mismatch')
    methods={};predictions={}
    for method in ['full_m23','m21_temporal_offline']:
        layer=BallReview(analysis/'giroud'/method,meta);balls=[layer.frame(i).model_dump() for i in range(meta['frames'])];predictions[method]=balls
        metrics=ball_metrics(list(store.tasks.values()),rows,{'giroud':balls},store.dataset['sources'])
        subsets={}
        for kind in ['uniform_source_sequence','post_hoc_diagnostic']:
            selected=[t for t in store.tasks.values() if t['selection_kind']==kind];ids={t['task_id'] for t in selected}
            subsets[kind]=ball_metrics(selected,[r for r in rows if r['task_id'] in ids],{'giroud':balls},store.dataset['sources'])
        methods[method]={'metrics':metrics,'subsets':subsets,'ball_frames_sha256':layer.manifest['ball_frames_sha256'],
            'model_set_sha256':layer.manifest['detector_sha256']}
    manifest=json.loads((analysis/'giroud/full_m23/manifest.json').read_text());record_path=Path(manifest['perception_records_path'])
    with record_path.open() as stream:records=[json.loads(line) for line in stream]
    traces=failure_traces(list(store.tasks.values()),rows,records,predictions['full_m23'])
    report={'schema_version':'cv-source-review-evaluation-v1','recording_id':'giroud','reference_role':'development_only',
        'model_assisted':True,'independent_final_test':False,'evaluated_at_utc':datetime.now(timezone.utc).isoformat(),
        'source':{k:meta[k] for k in ['recording_id','frames','fps','width','height','video_sha256','frames_sha256']},
        'dataset_fingerprint':store.dataset['fingerprint'],'task_manifest_sha256':digest(snapshot/'tasks.json'),
        'annotation_snapshot_sha256':digest(snapshot/'labels.json'),'snapshot_root':str(snapshot),
        'live_labels_path':str(labels),'analysis_report_sha256':digest(analysis/'report.json'),
        'perception_records_sha256':digest(record_path),'completed_tasks':len(rows),'reviewers':1,
        'annotation_summary':{'states':dict(Counter(r['state'] for r in rows)),
            'accepted_proposals':sum(r['accepted_proposal'] is not None for r in rows),
            'manual_visible_boxes':sum(r['state']=='visible' and r['accepted_proposal'] is None for r in rows),
            'rejected_regions':sum(len(r['hard_negative_regions']) for r in rows)},
        'systems':methods,'failure_counts':dict(Counter(t['reason'] for t in traces)),
        'candidate_audit':{'visible_frames_with_matching_raw_candidate':sum(t['reference_state']=='visible' and t['matching_raw_candidates']>0 for t in traces),
            'visible_frames_with_matching_eligible_candidate':sum(t['reference_state']=='visible' and t['matching_eligible_candidates']>0 for t in traces),
            'matching_rule':'Same reviewed source frame, IoU >= 0.25; diagnostic pool coverage, not a deployed detector score'},
        'new_inference':False,'new_training':False,'predictions_changed':False,'pass_accuracy':None,
        'limits':['Assisted development review; 32 temporally correlated source-spread frames and eight post-hoc diagnostics.',
            'Nine uncertain frames are excluded from known-visibility metrics; no absent-ball references were supplied.',
            'A wrong selected box on a visible-ball frame counts as both a false detection and a missed ball.',
            'Candidate-pool matches diagnose losses; choosing candidates by human labels is not a deployable inference method.',
            'Complementary proposals retain the existing two-sided-primary-anchor policy; eligible pixels alone do not establish a safe trajectory.',
            'No pass references or independent final-test labels were collected by this ball-only review.']}
    write(root/'failure_traces.json',traces);write(root/'report.json',report)
    if digest(labels)!=report['annotation_snapshot_sha256'] or digest(tasks)!=report['task_manifest_sha256']:
        raise ValueError('Live review changed during evaluation; retain this snapshot and evaluate a fresh version')
    print(json.dumps({'completed':len(rows),'states':report['annotation_summary']['states'],
        'systems':{k:v['metrics']['ball'] for k,v in methods.items()},'failures':report['failure_counts']},indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--analysis',type=Path,default=Path('outputs/m23/giroud_v1/analysis'))
    p.add_argument('--tasks',type=Path,default=Path('configs/training/cv_giroud_review.json'))
    p.add_argument('--labels',type=Path,default=Path('outputs/m23/giroud_v1/human_review/labels.json'))
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.analysis,a.tasks,a.labels,a.output)
