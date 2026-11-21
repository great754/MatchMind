"""Fixed development comparisons and exact one-to-one pass-reference diagnostics."""
import json
import numpy as np
from scipy.optimize import linear_sum_assignment
from src.evaluation.cv_m19 import evaluate, iou
from src.evaluation.cv_m22_development import ball_metrics
from src.serving.cv_review_labels import HumanReviewStore, HumanAnnotation
from src.vision.football22.review import DevelopmentReviewStore, DevelopmentLabel
from src.vision.events.passes import visible_players
from src.vision.football22.interactions import pass_trace


def references(inputs):
    original = HumanReviewStore(inputs.path('configs/evaluation/cv_m19_tasks.json'),
        inputs.path('outputs/m19/human_review/labels.json').parent)
    labels = original.labels()
    for row in labels:
        annotation = HumanAnnotation.model_validate({k: v for k, v in row.items() if k != 'source_video_sha256'})
        original.validate(original.tasks[row['task_id']]['recording_id'], annotation)
    reviewed = DevelopmentReviewStore(inputs.path('outputs/m22/completed_review_snapshot_v1/tasks.json'),
        inputs.path('outputs/m22/completed_review_snapshot_v1/labels.json').parent)
    additional = reviewed.labels()
    for row in additional:
        annotation = DevelopmentLabel.model_validate({k: v for k, v in row.items() if k in DevelopmentLabel.model_fields})
        reviewed.validate(reviewed.tasks[row['task_id']]['recording_id'], annotation)
    if len(labels) != 60 or len(additional) != 150 or len({r['reviewer_id'] for r in additional}) != 1:
        raise ValueError('The fixed original/new development reference sample changed')
    return original.dataset, labels, list(reviewed.tasks.values()), additional


def original_metrics(dataset, labels, balls, events, worlds):
    protocol = {**dataset['protocol'], 'sources': dataset['sources']}
    players = {name: {i: list(visible_players(w).values()) for i, w in enumerate(rows)} for name, rows in worlds.items()}
    result = evaluate(dataset['tasks'], labels, balls, events, protocol, players)
    result['status'] = 'development_reference_evaluation'
    result['reference_role'] = 'development_only'
    result['original_review_tasks'] = result.pop('independent_label_count')
    result['independent_final_test'] = False
    return result


def one_to_one_pass_traces(dataset, labels, name, balls, events, evidence, fps, selected=None):
    lookup = {t['task_id']: t for t in dataset['tasks'] if t['recording_id'] == name}
    traces = []
    for row in labels:
        task = lookup.get(row['task_id'])
        if task is None or task['kind'] != 'passes' or row['state'] != 'reviewed' or not row['exhaustive'] or any(p['outcome'] == 'uncertain' for p in row['passes']):
            continue
        refs = [p for p in row['passes'] if p['outcome'] == 'completed']
        predictions = [e for e in events if e['status'] == 'estimated' and task['start_frame'] <= e['start_frame'] and e['end_frame'] <= task['end_frame']]
        costs = np.full((len(refs), len(predictions)), 1e6)
        for i, ref in enumerate(refs):
            for j, event in enumerate(predictions):
                errors = [abs(ref['release_frame']/fps-event['start']), abs(ref['receipt_frame']/fps-event['end'])]
                if max(errors) <= dataset['protocol']['pass_endpoint_tolerance_seconds']:
                    costs[i, j] = sum(errors)
        matched = {i for i, j in zip(*linear_sum_assignment(costs)) if costs[i, j] < 1e6} if costs.size else set()
        for i, reference in enumerate(refs):
            trace = pass_trace(reference, task, balls, evidence, events, fps)
            trace['matched_completed_pass'] = i in matched
            trace['reference_id'] = task['task_id']+'#'+str(i)
            trace['matching_rule'] = 'unchanged_one_to_one_endpoints_inside_exhaustive_window'
            a, b = reference['release_frame'], reference['receipt_frame']
            recovered = [f for f in range(a, b+1) if selected and selected[f] and selected[f].get('origin') == 'guided' and balls[f]['directly_observed']]
            trace['trajectory_guided_current_observation_frames'] = recovered
            trace['receiver_detected_in_M9_near_receipt'] = any(r['candidates'] for r in trace['receiver_evidence'])
            trace['player_team_evidence_is_independent_truth'] = False
            trace['sender_receiver_identity_source'] = 'unchanged_explicit_M9_visible_segments_and_team_support'
            traces.append(trace)
    return traces


def cases(tasks, labels, name, balls, records, method, baseline):
    lookup = {t['task_id']: t for t in tasks if t['recording_id'] == name}
    result = []
    for row in labels:
        task = lookup.get(row['task_id'])
        if task is None or task.get('kind', 'ball') != 'ball' or row['state'] == 'uncertain':
            continue
        frame = task['frame']
        ball = balls[frame]
        match = bool(row['state'] == 'visible' and ball['directly_observed'] and iou(ball['image_bbox'], row['bbox']) >= .25)
        base = baseline[frame]
        base_match = bool(row['state'] == 'visible' and base['directly_observed'] and iou(base['image_bbox'], row['bbox']) >= .25)
        if row['state'] == 'visible' and not match:
            result.append({'kind': 'missed_visible_ball', 'frame': frame})
        if ball['directly_observed'] and not match:
            result.append({'kind': 'false_detection', 'frame': frame})
        if match and not base_match:
            result.append({'kind': 'improved_reviewed_frame', 'frame': frame})
        if base_match and not match:
            result.append({'kind': 'regressed_reviewed_frame', 'frame': frame})
        memory = records[frame]['methods'][method]['memory'] if method in records[frame]['methods'] else None
        if memory and memory['reacquired_with_current_visual_evidence'] and ball['directly_observed']:
            result.append({'kind': 'correct_on_model_reacquisition_frame' if match else 'incorrect_on_model_reacquisition_frame', 'frame': frame})
    return result


def team_goal_failure(tasks, labels, name, balls, records, method, m22_debug):
    lookup = {t['task_id']: t for t in tasks if t['recording_id'] == name and t.get('kind', 'ball') == 'ball'}
    result = []
    for row in labels:
        task = lookup.get(row['task_id'])
        if task is None or row['state'] != 'visible':
            continue
        f = task['frame']
        candidate_set = records[f]['methods'][method]['candidates'] if method in records[f]['methods'] else records[f]['primary_candidates']
        raw = [c for c in candidate_set if iou(c['bbox'], row['bbox']) >= .25]
        eligible = [c for c in raw if c.get('observation_eligible', c['confidence'] >= .35 and c['deterministic']['accepted'])]
        matching = balls[f]['directly_observed'] and iou(balls[f]['image_bbox'], row['bbox']) >= .25
        primary = [c for c in records[f]['primary_candidates'] if iou(c['bbox'], row['bbox']) >= .25]
        regions = records[f]['methods'][method]['memory']['search_regions'] if method in records[f]['methods'] else []
        x, y, w, h = row['bbox']
        covered = any(a <= x+w/2 < a+bw and b <= y+h/2 < b+bh for a, b, bw, bh in [r['bbox'] for r in regions])
        stage = ('MATCHED' if matching else 'ASSOCIATION_OR_REPEAT_SUPPORT' if eligible else
                 'VISUAL_VERIFICATION_OR_THRESHOLD' if raw else 'NO_SEARCH_MEMORY_REGION' if not regions else
                 'SEARCH_REGION_MISSED_REFERENCE' if not covered else 'NO_MATCHING_VISUAL_CANDIDATE_INSIDE_SEARCH')
        result.append({'frame': f, 'matched': bool(matching), 'failure_stage': stage,
            'matching_primary_candidates': len(primary), 'matching_all_current_candidates': len(raw),
            'matching_eligible_candidates': len(eligible), 'search_regions': len(regions), 'reference_inside_search': covered,
            'matching_m22_top5': [{'rank': i+1, 'confidence': c['confidence'],
                'current_only_confidence': c['current_only_confidence'], 'observation_eligible': c['observation_eligible']}
                for i, c in enumerate(m22_debug[f]['candidates'][:5]) if iou(c['bbox'], row['bbox']) >= .25],
            'matching_candidate_rejection_reasons': [c.get('decision_reason', c.get('deterministic', {}).get('reason')) for c in raw if c not in eligible],
            'diagnosis_uses_labels_only_after_perception': True})
    return result
