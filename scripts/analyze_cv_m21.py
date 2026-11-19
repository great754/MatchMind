"""Source-separated development comparisons; offline reconstructions never become observations."""
import argparse
import json
import math
from pathlib import Path
import time
from collections import Counter
import numpy as np
from src.evaluation.cv_m19 import evaluate, iou
from src.serving.cv import digest
from src.vision.ball.schema import BallFrame, canonical
from src.vision.events.passes import visible_players, signature
from src.vision.events.passes_m19 import detect_m19
from src.vision.advanced21.experiment import NAMES, fresh_directory, write_json, source_worlds, frozen_m20
from src.vision.advanced21.temporal import feature_vector, fit_linear, linear_score
from src.vision.advanced21.offline import viterbi, dag_link, reconstruct
from src.vision.advanced21.tracking import causal_rows, offline_rows, imm_rows
from src.vision.advanced21.players import PlayerTracker
from src.vision.advanced21.passes import verify_events, verified_summary
from src.vision.advanced21.physics import fit_ballistic, image_quadratic
from scripts.run_cv_m19 import ball_metrics
from scripts.evaluate_cv_m20 import localization

VARIANTS = ['m20_control', 'dense_current', 'temporal_deterministic', 'temporal_learned',
            'multiframe_detection', 'offline_viterbi', 'offline_dag', 'temporal_offline', 'player_motion', 'combined', 'external_temporal', 'external_offline', 'cross_verified', 'cross_offline', 'combined_cross',
            'imm_control','imm_dense','imm_multiframe','imm_cross','imm_combined']


def decision_reason(variant, candidate, selected):
    if selected:return 'ACTUAL_CURRENT_VISUAL_MEASUREMENT_SELECTED'
    if candidate['confidence']<.35:return 'BELOW_UNCHANGED_VISUAL_THRESHOLD'
    if variant in ['cross_verified','cross_offline','combined_cross','imm_cross','imm_combined'] and not (candidate.get('cross_detector') or {}).get('accepted'):
        return 'OTHER_CURRENT_PIXEL_DETECTOR_DID_NOT_CONFIRM'
    if variant in ['temporal_deterministic','temporal_offline','combined','combined_cross','imm_combined'] and not candidate['deterministic']['accepted']:
        return candidate['deterministic']['reason']
    score=candidate.get('learned_score')
    if variant in ['temporal_learned','combined','combined_cross','imm_combined'] and score is not None and score<.5:
        return 'SOURCE_SEPARATED_TEMPORAL_CLASSIFIER_VETO'
    if variant in ['external_temporal','external_offline'] and candidate['temporal_features'].get('external_temporal_score') is not None and candidate['temporal_features']['external_temporal_score']<.5:
        return 'FROZEN_EXTERNAL_VISUAL_PAIR_VETO'
    return 'NOT_SELECTED_BY_ASSOCIATION_OR_REPEATED_VISUAL_SUPPORT'


def image_fit_summary(rows):
    groups=[];active=[]
    for row in rows:
        if row['directly_observed'] and (not active or row['track_id']==active[-1]['track_id'] and row['camera_epoch']==active[-1]['camera_epoch']):
            active.append(row)
        else:
            if len(active)>=6:groups.append(active)
            active=[row] if row['directly_observed'] else []
    if len(active)>=6:groups.append(active)
    values=[]
    for group in groups:
        for offset in range(0,len(group)-5,20):
            window=group[offset:offset+20]
            fit=image_quadratic([r['timestamp'] for r in window],[r['image_position'] for r in window])
            if fit is not None:values.append(fit['rms_px'])
    return {'image_quadratic_windows':len(values),'median_in_sample_rms_px':float(np.median(values)) if values else None,
            'independent_metric_or_height_accuracy':None,'definition':'Diagnostic 2D fit to selected model observations, not human trajectory ground truth.'}


def supervision(candidate, label):
    """Development labels for the human-identified match ball; ambiguous overlaps are ignored."""
    if label['state'] == 'uncertain':
        return None
    if label['state'] == 'not_visible':
        return 0
    overlap = iou(candidate['bbox'], label['bbox'])
    if overlap >= .25:
        return 1
    a, b = candidate['bbox'], label['bbox']
    distance = math.dist([a[0]+a[2]/2, a[1]+a[3]/2], [b[0]+b[2]/2, b[1]+b[3]/2])
    if overlap <= .01 and distance > 2*math.hypot(b[2], b[3]):
        return 0
    return None


def train_models(inputs, tasks, labels, reference_sha256):
    lookup = {t['task_id']: t for t in tasks}
    examples = []
    for label in labels:
        task = lookup[label['task_id']]
        if task['kind'] != 'ball':
            continue
        row = inputs[task['recording_id']][task['frame']]
        candidates = row['candidate_sets']['multi_frame']
        for candidate in candidates:
            if candidate['confidence'] < .35:
                continue
            target = supervision(candidate, label)
            if target is not None:
                examples.append({'clip': task['recording_id'], 'frame': task['frame'], 'label': target,
                    'features': feature_vector(candidate['temporal_features'], candidate['confidence'], 'external_temporal_score' in candidate['temporal_features']),
                    'weight': 1/max(1, len(candidates))})
    models = {}
    for excluded in [*NAMES, 'none']:
        train = [e for e in examples if e['clip'] != excluded]
        model = fit_linear([e['features'] for e in train], [e['label'] for e in train], [e['weight'] for e in train])
        model.update(excluded_recording=None if excluded == 'none' else excluded,
                     training_recordings=sorted({e['clip'] for e in train}),
                     independent_training_frames=len({(e['clip'], e['frame']) for e in train}),
                     reference_role='development_only', annotations_sha256=reference_sha256,
                     supervision='Human match-ball boxes/known absence; far-away candidate crops are derived development negatives, not new independent annotations.',
                     source_separated_scaling_and_fit=True)
        models[excluded] = model
    return models


def accepted_sets(records, group, method, model):
    result = []
    for row in records:
        accepted = []
        for candidate in row['candidate_sets'][group]:
            if candidate['confidence'] < .35:
                continue
            deterministic = candidate['deterministic']['accepted']
            score = linear_score(model, candidate['temporal_features'], candidate['confidence'])
            learned = score is None or score >= .5
            external = candidate['temporal_features'].get('external_temporal_score')
            cross = (candidate.get('cross_detector') or {}).get('accepted',False)
            if method == 'none' or method == 'deterministic' and deterministic or method == 'learned' and learned or method == 'both' and deterministic and learned or method == 'external' and (external is None or external >= .5) or method == 'cross' and cross or method == 'cross_both' and cross and deterministic and learned:
                accepted.append(candidate)
        result.append(accepted)
    return result


def write_variant(directory, name, metadata, balls, events, summary, evidence, debug, config, detector_hash):
    directory.mkdir(parents=True)
    offsets, debug_offsets = [], []
    with (directory/'ball_frames.jsonl').open('wb') as target, (directory/'video_ball_observations.jsonl').open('w') as stream, (directory/'advanced_frames.jsonl').open('wb') as detail:
        for row, state in zip(debug, balls, strict=True):
            offsets.append(target.tell()); target.write((state.model_dump_json()+'\n').encode())
            stream.write(canonical(name, metadata['fps'], state, detector_hash).model_dump_json()+'\n')
            debug_offsets.append(detail.tell()); detail.write((json.dumps(row, separators=(',', ':'), allow_nan=False)+'\n').encode())
    manifest = {k: metadata[k] for k in ['recording_id', 'frames', 'fps', 'width', 'height', 'video_sha256']}
    manifest.update(version='m21', source_m9_frames_sha256=metadata['frames_sha256'], detector_sha256=detector_hash,
        ball_frames_sha256=digest(directory/'ball_frames.jsonl'), canonical_sha256=digest(directory/'video_ball_observations.jsonl'),
        frame_offsets=offsets, statistics=ball_metrics([b.model_dump() for b in balls]), config=config, config_sha256=signature(config),
        advanced_sha256=digest(directory/'advanced_frames.jsonl'), advanced_frame_offsets=debug_offsets)
    write_json(directory/'manifest.json', manifest)
    event_directory = directory/'passes'
    event_directory.mkdir()
    write_json(event_directory/'events.json', events)
    (event_directory/'evidence_frames.jsonl').write_text(''.join(json.dumps(r, separators=(',', ':'))+'\n' for r in evidence))
    write_json(event_directory/'summary.json', summary)
    return manifest


def run(source, output, audit_root):
    source, audit_root = Path(source), Path(audit_root)
    perception = json.loads((source/'report.json').read_text())
    if set(perception['clips']) != set(NAMES):
        raise ValueError('Finish fresh inference on all three clips first')
    config = json.loads((source/'config.json').read_text())
    if digest(source/'config.json') != perception['config_sha256']:
        raise ValueError('Perception configuration changed')
    audit = json.loads((audit_root/'audit.json').read_text())
    labels_path = audit_root/'development_labels.json'
    if digest(labels_path) != perception['development_annotation_sha256'] or digest(labels_path) != audit['annotations_sha256']:
        raise ValueError('Development labels changed')
    dataset = json.loads((audit_root/'development_tasks.json').read_text())
    labels = json.loads(labels_path.read_text())['annotations']
    protocol = {**dataset['protocol'], 'sources': dataset['sources']}
    inputs, worlds, people, metadata, controls = {}, {}, {}, {}, {}
    for name in NAMES:
        path = source/name/'inputs.jsonl'
        if digest(path) != perception['clips'][name]['inputs_sha256']:
            raise ValueError('Perception cache changed')
        inputs[name] = [json.loads(line) for line in path.open()]
        metadata[name], worlds[name], people[name] = source_worlds(name)
        controls[name], _ = frozen_m20(name, perception['m20_reference'])
    root = fresh_directory(output)
    models = train_models(inputs, dataset['tasks'], labels, audit['annotations_sha256'])
    write_json(root/'temporal_models.json', models)
    report = {'schema_version': 'cv-m21-analysis-v1', 'perception_source': str(source), 'perception_report_sha256': digest(source/'report.json'),
        'config': config, 'reference_role': 'development_only', 'annotations_sha256': audit['annotations_sha256'],
        'dataset_fingerprint': dataset['fingerprint'], 'final_test_status': 'pending_new_source_disjoint_human_labels',
        'temporal_verifier': {'architecture': 'L2 regularized logistic temporal visual baseline; ten features plus two frozen-pair features when available',
            'fold_policy': 'Leave one complete source clip out of feature scaling, weighting and fitting.',
            'not_untouched_test': True, 'detector_confidence_never_inflated': True}, 'variants': {}, 'clips': {}}
    measurements = {v: {'balls': {}, 'events': {}, 'players': {}} for v in VARIANTS}
    for name in NAMES:
        m, records, source_world = metadata[name], inputs[name], worlds[name]
        cameras = [row['camera'] for row in source_world]
        model = models[name]
        sets = {
            'dense_current': accepted_sets(records, 'dense', 'none', model),
            'temporal_deterministic': accepted_sets(records, 'm20_candidates', 'deterministic', model),
            'temporal_learned': accepted_sets(records, 'm20_candidates', 'learned', model),
            'multiframe_detection': accepted_sets(records, 'multi_frame', 'none', model),
            'offline_viterbi': accepted_sets(records, 'm20_candidates', 'none', model),
            'offline_dag': accepted_sets(records, 'm20_candidates', 'none', model),
            'temporal_offline': accepted_sets(records, 'multi_frame', 'deterministic', model),
            'combined': accepted_sets(records, 'multi_frame', 'both', model),
            'external_temporal': accepted_sets(records, 'multi_frame', 'external', model),
            'external_offline': accepted_sets(records, 'multi_frame', 'external', model),
            'cross_verified': accepted_sets(records, 'multi_frame', 'cross', model),
            'cross_offline': accepted_sets(records, 'multi_frame', 'cross', model),
            'combined_cross': accepted_sets(records, 'multi_frame', 'cross_both', model),
            'imm_control': accepted_sets(records,'m20_candidates','none',model),
            'imm_dense': accepted_sets(records,'dense','none',model),
            'imm_multiframe': accepted_sets(records,'multi_frame','none',model),
            'imm_cross': accepted_sets(records,'multi_frame','cross',model),
            'imm_combined': accepted_sets(records,'multi_frame','cross_both',model)}
        player_baseline, player_compensated = [], []
        for compensated in [False, True]:
            tracker = PlayerTracker(m['fps'], config['players'], compensated, appearance=compensated)
            rows = [tracker.step(i, cameras[i], records[i]['player_observations']) for i in range(m['frames'])]
            (player_compensated if compensated else player_baseline).extend(rows)
        original_players = {i: list(visible_players(w).values()) for i, w in enumerate(source_world)}
        alternate_players = {i: rows for i, rows in enumerate(player_compensated)}
        report['clips'][name] = {'camera': {'accepted_relative_refinements': sum(r['camera_refinement']['accepted'] for r in records),
                'frames': m['frames'], 'absolute_calibration_changed': False, 'independent_reprojection_error_px': None,
                'original_m8_valid_calibration_frames': sum(c['calibration_valid'] for c in cameras)},
            'physics': fit_ballistic([], [], calibration_verified=False),
            'players': {'manual_identity_changes': 0, 'new_identity_namespace': 'M21',
                'source': 'Existing observed M9 boxes and appearance; no new player-detector inference.',
                'identity_negative_pairs': 0}}
        for variant in VARIANTS:
            started = time.perf_counter()
            inferred = [None]*m['frames']
            group = 'm20_candidates' if variant in ['m20_control', 'temporal_deterministic', 'temporal_learned', 'offline_viterbi', 'offline_dag', 'player_motion','imm_control'] else 'dense' if variant in ['dense_current','imm_dense'] else 'multi_frame'
            offline = variant in ['offline_viterbi', 'offline_dag', 'temporal_offline', 'combined', 'external_offline', 'cross_offline', 'combined_cross']
            if variant in ['m20_control', 'player_motion']:
                balls = [BallFrame.model_validate({**b, 'version': 'm21'}) for b in controls[name]]
            elif variant.startswith('imm_'):
                balls = imm_rows(sets[variant],source_world,records,m)
            elif offline:
                linker = viterbi if variant == 'offline_viterbi' else dag_link
                selected = linker(sets[variant], cameras, m['fps'], math.hypot(m['width'], m['height']))
                inferred = reconstruct(selected, cameras, m['fps'])
                balls = offline_rows(selected, inferred, sets[variant], source_world, m)
            else:
                balls = causal_rows(sets[variant], source_world, records, m)
            raw_balls = [b.model_dump() for b in balls]
            base_events, summary, evidence = detect_m19(name, source_world, raw_balls, m['fps'])
            # Events always use the original explicit M9 participants. New anonymous tracks cannot fabricate attribution.
            events, event_checks = verify_events(base_events, evidence, m['fps']) if variant in ['combined','combined_cross','imm_combined'] else (base_events, [])
            summary = verified_summary(summary, events)
            debug = []
            for i, ball in enumerate(balls):
                selected_id = next((c.candidate_id for c in ball.candidates if ball.directly_observed and c.bbox == ball.image_bbox), None)
                candidates = records[i]['candidate_sets'][group]
                decisions = []
                for candidate in candidates:
                    score = linear_score(model, candidate['temporal_features'], candidate['confidence'])
                    selected_here = ball.directly_observed and tuple(candidate['bbox']) == ball.image_bbox and candidate['confidence'] == ball.detection_confidence
                    decisions.append({'candidate_id': candidate['candidate_id'], 'bbox': candidate['bbox'],
                        'visual_confidence': candidate['confidence'], 'inference_view': candidate['inference_view'],
                        'selected': selected_here, 'deterministic': candidate['deterministic'], 'temporal_score': score,
                        'temporal_features': candidate['temporal_features'], 'cross_detector':candidate.get('cross_detector'), 'decision_reason':decision_reason(variant,{**candidate,'learned_score':score},selected_here), 'score_calibrated': False})
                debug.append({'schema_version': 'cv-m21-debug-v1', 'variant': variant, 'frame': i,
                    'timestamp': ball.timestamp, 'camera_epoch': ball.camera_epoch,
                    'association_mode': 'offline' if offline or variant.startswith('imm_') or variant in ['m20_control', 'player_motion'] else 'causal_ball_only',
                    'uses_future_for_ball_association': offline or variant.startswith('imm_') or variant in ['m20_control', 'player_motion'],
                    'event_analysis_mode': 'offline_with_original_M9_participants',
                    'selected_candidate_id': selected_id, 'directly_observed': ball.directly_observed,
                    'candidates': decisions, 'reconstruction': inferred[i], 'camera_refinement': records[i]['camera_refinement'],
                    'player_tracks': player_compensated[i] if variant in ['combined', 'player_motion'] else player_baseline[i],
                    'manual_identity_propagated': False})
            directory = root/name/variant
            manifest = write_variant(directory, name, m, balls, events, summary, evidence, debug,
                {**config, 'variant': variant, 'temporal_model': model, 'event_second_stage': variant in ['combined','combined_cross','imm_combined']},
                signature({'m20': json.loads(Path('configs/vision/cv_m20.json').read_text())['detector']['sha256'], 'generic': perception['checkpoint_sha256']}))
            write_json(directory/'passes/second_stage.json', event_checks)
            measurements[variant]['balls'][name] = raw_balls
            measurements[variant]['events'][name] = events
            measurements[variant]['players'][name] = alternate_players if variant in ['player_motion', 'combined'] else original_players
            tasks = [t for t in dataset['tasks'] if t['recording_id'] == name]
            ids = {t['task_id'] for t in tasks}
            refs = [r for r in labels if r['task_id'] in ids]
            metrics = evaluate(tasks, refs, {name: raw_balls}, {name: events}, protocol, {name: measurements[variant]['players'][name]})
            metrics['localization'] = localization(tasks, refs, {name: raw_balls}, protocol)
            report['variants'].setdefault(variant, {'per_recording': {}})['per_recording'][name] = {
                **metrics, 'engineering': manifest['statistics'], 'passes_before_second_stage': sum(e['status'] == 'estimated' for e in base_events),
                'passes_after_second_stage': summary['completed_pass_estimates'], 'analysis_wall_seconds': time.perf_counter()-started, 'image_space_fit':image_fit_summary(raw_balls),
                'incorrect_ball_associations': None, 'independent_reacquisition_accuracy': None}
            print('M21 ANALYSIS', name, variant, metrics['ball']['tp'], metrics['ball']['fp'], metrics['ball']['fn'], flush=True)
    for variant, sources in measurements.items():
        total = evaluate(dataset['tasks'], labels, sources['balls'], sources['events'], protocol, sources['players'])
        total['localization'] = localization(dataset['tasks'], labels, sources['balls'], protocol)
        report['variants'][variant]['aggregate'] = total
    baseline = report['variants']['m20_control']['aggregate']['ball']
    baseline_passes = report['variants']['m20_control']['aggregate']['passes']
    supported = [v for v in VARIANTS if report['variants'][v]['aggregate']['ball']['fp'] <= baseline['fp']
                 and (report['variants'][v]['aggregate']['ball']['precision'] or 0) >= (baseline['precision'] or 0)
                 and report['variants'][v]['aggregate']['passes']['tp'] >= baseline_passes['tp']
                 and report['variants'][v]['aggregate']['passes']['fp'] <= baseline_passes['fp']]
    selected = max(supported, key=lambda v: (report['variants'][v]['aggregate']['ball']['f1'] or 0,
                  report['variants'][v]['aggregate']['ball']['precision'] or 0, v == 'm20_control'))
    report['selection'] = {'development_selected_variant': selected, 'selection_policy': 'Maximum development F1 with no increase in ball/pass false detections, no loss of ball precision and no loss of pass matches against M20; fixed .35 visual and .5 learned thresholds.',
        'application_default_changed': False, 'final_accuracy_acceptance': 'not_established_without_new_test_labels',
        'training_or_selection_on_untouched_test': False, 'original_m14_m16_m19_m20_preserved': True}
    write_json(root/'report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--audit', required=True)
    args = parser.parse_args()
    run(args.source, args.output, args.audit)
