"""Fixed four-way M23 integration analysis; development truth is read only here."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import resource
import time
import cv2
from src.serving.cv import digest
from src.evaluation.cv_m19 import iou
from src.evaluation.cv_m22_development import ball_metrics as assisted_metrics
from src.vision.ball.schema import BallFrame, VideoBallObservation, canonical
from src.vision.ball20.tracker import appearance
from src.vision.advanced21.offline import dag_link, reconstruct
from src.vision.advanced21.tracking import offline_rows
from src.vision.football22.interactions import interaction_frames
from src.vision.events.passes import PassConfig, VideoPass, signature
from src.vision.events.passes_m19 import detect_m19
from src.vision.intelligent23.inputs import Inputs, NAMES, write
from src.vision.intelligent23.offline import reconstruct_camera, history, appearance_edge, anchored_complements
from src.vision.intelligent23.schema import IntelligentFrame
from src.vision.intelligent23.evaluation import references, original_metrics, one_to_one_pass_traces, cases, team_goal_failure
from scripts.run_cv_m19 import ball_metrics as engineering_metrics

METHODS = ('m21_temporal_offline', 'guided_search', 'guided_verified', 'full_m23')
BASELINES = ('m20_control', 'm21_temporal_offline', 'm22_reviewed')


def read_baseline(inputs, version, name, meta):
    directory = (Path('outputs/m20/run_v1')/name/'ball_m20' if version == 'm20_control' else
                 Path('outputs/m21/analysis_final_v1')/name/'temporal_offline' if version == 'm21_temporal_offline' else
                 Path('outputs/m22/analysis_review_v1')/name)
    manifest = inputs.json(directory/'manifest.json')
    for key in ['recording_id', 'frames', 'fps', 'width', 'height', 'video_sha256']:
        if meta[key] != manifest[key]:
            raise ValueError('Frozen baseline source/clock mismatch')
    if meta['frames_sha256'] != manifest['source_m9_frames_sha256']:
        raise ValueError('Frozen baseline world mismatch')
    ball_path = inputs.path(directory/'ball_frames.jsonl')
    canonical_path = inputs.path(directory/'video_ball_observations.jsonl')
    if digest(ball_path) != manifest['ball_frames_sha256'] or digest(canonical_path) != manifest['canonical_sha256']:
        raise ValueError('Frozen baseline evidence hash changed')
    rows = []
    with ball_path.open('rb') as source, canonical_path.open() as observations:
        for i in range(meta['frames']):
            if source.tell() != manifest['frame_offsets'][i]:
                raise ValueError('Frozen baseline byte index mismatch')
            row = BallFrame.model_validate_json(source.readline())
            observation = VideoBallObservation.model_validate_json(observations.readline())
            if row.frame != i or abs(row.timestamp-i/meta['fps']) > 1e-6 or canonical(name, meta['fps'], row, manifest['detector_sha256']) != observation:
                raise ValueError('Frozen canonical observation mismatch')
            rows.append(row.model_dump())
        if source.read(1) or observations.read(1):
            raise ValueError('Unindexed baseline source')
    event_root = Path('outputs/m20/run_v1')/name/'passes_m20' if version == 'm20_control' else directory/'passes'
    event_manifest = inputs.json(event_root/'manifest.json')
    event_path = inputs.path(event_root/'events.json')
    if digest(event_path) != event_manifest['events_sha256'] or event_manifest['ball_frames_sha256'] != manifest['ball_frames_sha256']:
        raise ValueError('Frozen event source mismatch')
    events = [VideoPass.model_validate(e).model_dump() for e in json.loads(event_path.read_text())]
    return rows, events, manifest


def minimal(rows):
    keys = ['frame', 'timestamp', 'directly_observed', 'image_position', 'image_bbox', 'track_id']
    return [{k: r[k] for k in keys} for r in rows]


def selected_candidates(records, method):
    if method == 'm21_temporal_offline':
        return [[c for c in row['primary_candidates'] if c['confidence'] >= .35 and c['deterministic']['accepted']] for row in records]
    return [[c for c in row['methods'][method]['candidates'] if c['observation_eligible']] for row in records]


def debug_candidates(record, method, chosen):
    raw = record['primary_candidates'] if method == 'm21_temporal_offline' else record['methods'][method]['candidates']
    result = []
    for c in raw:
        selected = chosen is not None and c['candidate_id'] == chosen['candidate_id']
        eligible = c['confidence'] >= .35 and c['deterministic']['accepted']
        values = {**c, 'selected': selected}
        if method == 'm21_temporal_offline':
            values.update(support_views=[{'origin': 'primary', 'bbox': c['bbox'], 'confidence': c['confidence'], 'source_region': None}],
                observation_eligible=eligible, decision_reason='M21_UNCHANGED_TEMPORAL_PRIMARY' if eligible else 'M21_VISUAL_GATE_REJECTED',
                joint_memory_evidence=None, blur_supported=False, blur_definition='Not classified by this control',
                m22_support=[], association_adjustment=0., detector_confidence_unchanged=True,
                confidence_is_calibrated_probability=False)
        result.append(values)
    return result


def write_artifacts(root, name, method, meta, balls, selected, inferred, records, cameras, config, model_components, model_set_sha, source_config_sha, perception_path):
    directory = root/name/method
    directory.mkdir(parents=True)
    raw = [b.model_dump() for b in balls]
    events, summary, evidence = detect_m19(name, meta['_worlds'], raw, meta['fps'])
    interactions = interaction_frames(evidence, events)
    ball_offsets, debug_offsets, states = [], [], Counter()
    with (directory/'ball_frames.jsonl').open('wb') as bf, (directory/'video_ball_observations.jsonl').open('w') as observations, (directory/'intelligent_frames.jsonl').open('wb') as debug:
        for i, b in enumerate(balls):
            ball_offsets.append(bf.tell())
            bf.write(b.model_dump_json().encode()+b'\n')
            observations.write(canonical(name, meta['fps'], b, model_set_sha).model_dump_json()+'\n')
            state = 'OBSERVED' if b.directly_observed else 'PREDICTED' if b.predicted_image_position is not None else 'AMBIGUOUS' if b.tracking_status == 'AMBIGUOUS' else 'UNKNOWN'
            states[state] += 1
            frame = IntelligentFrame(schema_version='cv-m23-frame-v1', recording_id=name, variant=method,
                frame=i, timestamp=b.timestamp, camera_epoch=b.camera_epoch, state=state,
                directly_observed=b.directly_observed, selected_candidate_id=selected[i]['candidate_id'] if selected[i] else None,
                candidates=debug_candidates(records[i], method, selected[i]),
                memory=records[i]['methods'][method]['memory'] if method in records[i]['methods'] else None,
                reconstruction=inferred[i], trajectory_history=history(selected, cameras, i, meta['fps']),
                interaction=interactions[i], source_pixels_sha256=records[i]['source_pixels_sha256'],
                model_set_sha256=model_set_sha, config_sha256=source_config_sha,
                uses_future_for_association=True, future_visual_frames_used=False,
                m22_context_frames=records[i]['m22_context_frames'] if method == 'full_m23' else [], reasons=[b.reason])
            debug_offsets.append(debug.tell())
            debug.write(frame.model_dump_json().encode()+b'\n')
    manifest = {k: meta[k] for k in ['recording_id', 'frames', 'fps', 'width', 'height', 'video_sha256']}
    manifest.update(version='m23', source_m9_frames_sha256=meta['frames_sha256'], detector_sha256=model_set_sha,
        frame_offsets=ball_offsets, ball_frames_sha256=digest(directory/'ball_frames.jsonl'),
        canonical_sha256=digest(directory/'video_ball_observations.jsonl'),
        intelligent_sha256=digest(directory/'intelligent_frames.jsonl'), intelligent_frame_offsets=debug_offsets,
        config={**config, 'method': method}, config_sha256=signature({**config, 'method': method}),
        perception_config_sha256=source_config_sha, model_components=model_components,
        perception_records_path=str(perception_path), perception_records_sha256=digest(perception_path),
        statistics={**engineering_metrics(raw), 'states': dict(states), 'independent_reacquisition_accuracy': None,
                    'true_ball_identity_switches': None, 'true_trajectory_fragmentation': None})
    write(directory/'manifest.json', manifest)
    passes = directory/'passes'
    passes.mkdir()
    write(passes/'events.json', events)
    (passes/'evidence_frames.jsonl').write_text(''.join(json.dumps(e, separators=(',', ':'))+'\n' for e in evidence))
    write(passes/'summary.json', summary)
    cfg = PassConfig().model_dump()
    event_manifest = {k: meta[k] for k in ['recording_id', 'frames', 'fps', 'width', 'height', 'video_sha256']}
    event_manifest.update(schema_version='cv-passes-v1', algorithm_version='cv-pass-m23-v1',
        world_frames_sha256=meta['frames_sha256'], ball_version='m23', ball_frames_sha256=manifest['ball_frames_sha256'],
        ball_detector_sha256=model_set_sha, events_sha256=digest(passes/'events.json'), evidence_sha256=digest(passes/'evidence_frames.jsonl'),
        config=cfg, config_sha256=signature(cfg), statistics=summary,
        independent_reference_status='development_only_no_final_test',
        unavailable=['verified_touches', 'verified_possession', 'goals', 'shots', 'tackles', 'interceptions', 'pass_attempts', 'pass_accuracy'])
    write(passes/'manifest.json', event_manifest)
    return raw, events, summary, evidence, manifest


def run(inputs_root, perception_root, output):
    inputs, source, root = Inputs(inputs_root), Path(perception_root), Path(output)
    if root.exists():
        raise ValueError('Use a fresh immutable M23 analysis root')
    perception = json.loads((source/'report.json').read_text())
    if tuple(perception['clips']) != NAMES or perception['input_manifest_sha256'] != inputs.sha256:
        raise ValueError('Complete all three pinned M23 inference sources first')
    dataset, original, reviewed_tasks, reviewed = references(inputs)
    config = perception['config']
    root.mkdir(parents=True)
    report = {'schema_version': 'cv-m23-analysis-v1', 'reference_role': 'development_only',
        'analysis_source_sha256': {p: digest(p) for p in ['scripts/analyze_cv_m23.py',
            'src/vision/intelligent23/schema.py', 'src/vision/intelligent23/offline.py', 'src/vision/intelligent23/evaluation.py',
            'src/vision/advanced21/offline.py', 'src/vision/ball/schema.py']},
        'input_manifest_sha256': inputs.sha256, 'inputs_root': str(inputs_root),
        'perception_report_sha256': digest(source/'report.json'), 'perception': perception,
        'original_label_sha256': inputs.manifest['original_labels_sha256'], 'm22_label_sha256': inputs.manifest['m22_labels_sha256'],
        'variants': {m: {'per_recording': {}} for m in METHODS}, 'baselines': {v: {'per_recording': {}} for v in BASELINES},
        'baseline_reproduction': {}, 'independent_final_test': 'not_available_no_new_approved_source',
        'promotion': {'status': 'not_promoted', 'ordinary_default': 'm14', 'experimental_control': 'm20',
            'reason': 'Development-influenced clips and sparse labels; independent source approval, overlap audit and frozen labels required.'}}
    balls_all = {m: {} for m in [*METHODS, *['frozen_'+b for b in BASELINES]]}
    events_all = {m: {} for m in balls_all}
    worlds_all = {}
    for name in NAMES:
        started = time.perf_counter()
        meta, worlds, people = inputs.worlds(name)
        worlds_all[name] = worlds
        path = source/name/'inputs.jsonl'
        if digest(path) != perception['clips'][name]['inputs_sha256']:
            raise ValueError('M23 perception inputs changed')
        records = [json.loads(line) for line in path.open()]
        if len(records) != meta['frames'] or any(r['frame'] != i or abs(r['timestamp']-i/meta['fps']) > 1e-6 for i, r in enumerate(records)):
            raise ValueError('M23 source clock changed')
        frozen_balls = {}
        for baseline in BASELINES:
            rows, events, manifest = read_baseline(inputs, baseline, name, meta)
            frozen_balls[baseline] = rows
            balls_all['frozen_'+baseline][name], events_all['frozen_'+baseline][name] = minimal(rows), events
        old_records = [json.loads(line) for line in inputs.path(Path(inputs.manifest['m21_perception_root'])/name/'inputs.jsonl').open()]
        frozen_sets = [[c for c in r['candidate_sets']['multi_frame'] if c['confidence'] >= .35 and c['deterministic']['accepted']] for r in old_records]
        world_cameras = [w['camera'] for w in worlds]
        old_selected = dag_link(frozen_sets, world_cameras, meta['fps'], math.hypot(meta['width'], meta['height']))
        old_inferred = reconstruct(old_selected, world_cameras, meta['fps'])
        reproduced = offline_rows(old_selected, old_inferred, frozen_sets, worlds, meta)
        sha = hashlib.sha256(b''.join(b.model_dump_json().encode()+b'\n' for b in reproduced)).hexdigest()
        expected = inputs.json(Path('outputs/m21/analysis_final_v1')/name/'temporal_offline/manifest.json')['ball_frames_sha256']
        if sha != expected:
            raise ValueError('Frozen M21 default DAG no longer reproduces exactly')
        report['baseline_reproduction'][name] = {'frozen_m21_ball_sha256': sha, 'byte_identical': True,
            'fresh_source_pixel_hashes_match': all(a['source_pixels_sha256'] == b['source_pixels_sha256'] for a, b in zip(records, old_records, strict=True))}
        if not report['baseline_reproduction'][name]['fresh_source_pixel_hashes_match']:
            raise ValueError('Fresh neural run source pixels differ from the frozen reference')
        cameras = [r['causal_camera'] for r in records]
        # Pairwise appearance for offline links is recomputed from actual current pixels.
        descriptors = {}
        capture = cv2.VideoCapture(meta['source_video'])
        try:
            for i, record in enumerate(records):
                ok, image = capture.read()
                if not ok or hashlib.sha256(image.tobytes()).hexdigest() != record['source_pixels_sha256']:
                    raise ValueError('Offline appearance source mismatch')
                for method in METHODS:
                    for c in (record['primary_candidates'] if method == 'm21_temporal_offline' else record['methods'][method]['candidates']):
                        descriptors[(i, c['candidate_id'])] = appearance(image, c['bbox'])
        finally:
            capture.release()
        m22_debug = [json.loads(line) for line in inputs.path(Path('outputs/m22/analysis_review_v1')/name/'football_frames.jsonl').open()]
        for method in METHODS:
            method_started = time.perf_counter()
            candidate_sets = selected_candidates(records, method)
            # Candidate IDs repeat across frames only in the original control.
            descriptor_by_object = {id(c): descriptors[(i, c['candidate_id'])] for i, row in enumerate(candidate_sets) for c in row}
            def edge(a, b, dt):
                x, y = descriptor_by_object.get(id(a)), descriptor_by_object.get(id(b))
                if x is None or y is None:return 0.
                import numpy as np
                return .1*(1-float(np.clip(np.sqrt(x*y).sum(), 0, 1)))
            selected = dag_link(candidate_sets, world_cameras, meta['fps'], math.hypot(meta['width'], meta['height']),
                maximum_gap_seconds=config['offline']['maximum_gap_seconds'],
                node_adjustment=(lambda c, f: c['association_adjustment']) if method in ['guided_verified', 'full_m23'] else None,
                edge_adjustment=edge if method in ['guided_verified', 'full_m23'] else None)
            if method == 'full_m23' and config['offline'].get('primary_anchored_complement_only'):
                primary_sets=[[c for c in row if c.get('origin')!='complement'] for row in candidate_sets]
                primary_selected=dag_link(primary_sets,world_cameras,meta['fps'],math.hypot(meta['width'],meta['height']),
                    maximum_gap_seconds=config['offline']['maximum_gap_seconds'],
                    node_adjustment=lambda c,f:c['association_adjustment'],edge_adjustment=edge)
                selected=anchored_complements(primary_selected,candidate_sets,cameras,meta['fps'],
                    math.hypot(meta['width'],meta['height']),descriptors,config['offline']['maximum_gap_seconds'])
            inferred = (reconstruct(selected, world_cameras, meta['fps']) if method == 'm21_temporal_offline' else
                        reconstruct_camera(selected, cameras, meta['fps'], meta['width'], meta['height'], config['offline']['reconstruction_gap_seconds']))
            legacy_rows = offline_rows(selected, inferred, candidate_sets, worlds, meta)
            balls = []
            for i, row in enumerate(legacy_rows):
                values = {**row.model_dump(), 'version': 'm23', 'coordinate_provenance': 'M23_CURRENT_PIXEL_EVIDENCE; NO_VERIFIED_HEIGHT',
                    'motion_model': 'M20_IMM_MEMORY_M21_OFFLINE_DAG; OFFLINE_ASSOCIATION', 'search_stage': 'M23_'+method.upper()}
                if not row.directly_observed and row.predicted_image_position is None and len(candidate_sets[i]) > 1:
                    values.update(tracking_status='AMBIGUOUS', ambiguity_reason='UNRESOLVED_CURRENT_CANDIDATES_AFTER_OFFLINE_LINKING')
                balls.append(BallFrame.model_validate(values))
            components = [{'role': 'generic_primary', 'path': str(inputs.path('outputs/ml/ball_m16/yolo11s.pt')), 'sha256': perception['clips'][name]['generic_checkpoint_sha256']},
                          {'role': 'soccer_frozen_global_measurements', 'path': str(inputs.path('outputs/ml/ball_m16/training/soccer11s/weights/best.pt')), 'sha256': inputs.json('configs/vision/cv_m20.json')['detector']['sha256']}]
            if method == 'full_m23':
                trained = inputs.json('outputs/m22/analysis_review_v1/report.json')['training'][name]
                components.append({'role': 'm22_complementary', 'path': str(inputs.path(trained['checkpoint'])), 'sha256': trained['checkpoint_sha256']})
            for component in components:
                if digest(component['path']) != component['sha256']:
                    raise ValueError('Frozen component checkpoint mismatch')
            model_sha = signature({'components': components, 'config': perception['config_sha256'], 'variant': method})
            raw, events, summary, evidence, manifest = write_artifacts(root, name, method, {**meta, '_worlds': worlds}, balls,
                selected, inferred, records, cameras, config, components, model_sha, perception['config_sha256'], path)
            balls_all[method][name], events_all[method][name] = minimal(raw), events
            ids = {t['task_id'] for t in dataset['tasks'] if t['recording_id'] == name}
            old_labels = [r for r in original if r['task_id'] in ids]
            old_dataset = {**dataset, 'tasks': [t for t in dataset['tasks'] if t['recording_id'] == name]}
            orig = original_metrics(old_dataset, old_labels, {name: raw}, {name: events}, {name: worlds})
            new_tasks = [t for t in reviewed_tasks if t['recording_id'] == name]
            new_ids = {t['task_id'] for t in new_tasks}
            new_labels = [r for r in reviewed if r['task_id'] in new_ids]
            new = assisted_metrics(new_tasks, new_labels, {name: raw}, dataset['sources'])
            traces = one_to_one_pass_traces(dataset, original, name, raw, events, evidence, meta['fps'], selected)
            if sum(t['matched_completed_pass'] for t in traces) != orig['passes']['tp']:
                raise ValueError('One-to-one pass diagnostics disagree with the frozen evaluation rule')
            navigation = cases(reviewed_tasks, reviewed, name, raw, records, method, frozen_balls['m21_temporal_offline'])
            navigation += [{**c, 'reference_set': 'original'} for c in cases(dataset['tasks'], original, name, raw, records, method, frozen_balls['m21_temporal_offline'])]
            navigation += [{'kind': 'ball_disappeared', 'frame': i} for i in range(1, len(raw)) if raw[i-1]['directly_observed'] and not raw[i]['directly_observed']]
            navigation += [{'kind': 'reference_pass_matched' if t['matched_completed_pass'] else 'reference_pass_failed',
                            'frame': t['release_frame'], 'reference_id': t['reference_id']} for t in traces]
            diagnostic = {'original_references': orig, 'new_150': new, 'engineering': manifest['statistics'],
                'full_clip_passes': summary, 'pass_traces': traces, 'navigation': navigation,
                'analysis_seconds': time.perf_counter()-method_started,
                'reviewed_reacquisition_identity_accuracy': None,
                'reacquisition_limitation': 'Isolated human frames verify ball boxes, not continuous ball identity across an occlusion.'}
            if name == 'team_goal':
                diagnostic['original_failure_stages'] = team_goal_failure(dataset['tasks'], original, name, raw, records, method, m22_debug)
                diagnostic['new_failure_stages'] = team_goal_failure(reviewed_tasks, reviewed, name, raw, records, method, m22_debug)
            report['variants'][method]['per_recording'][name] = diagnostic
            write(root/name/method/'pass_reference_traces.json', traces)
            print('M23 ANALYSIS', name, method, 'new150', new['ball'], 'original', orig['ball'], 'passes', orig['passes']['tp'], flush=True)
        report.setdefault('analysis_timing', {})[name] = {'seconds': time.perf_counter()-started,
            'peak_rss_bytes_macos': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    for method in METHODS:
        report['variants'][method]['original_references'] = original_metrics(dataset, original, balls_all[method], events_all[method], worlds_all)
        report['variants'][method]['new_150'] = assisted_metrics(reviewed_tasks, reviewed, balls_all[method], dataset['sources'])
    for baseline in BASELINES:
        key = 'frozen_'+baseline
        report['baselines'][baseline]['original_references'] = original_metrics(dataset, original, balls_all[key], events_all[key], worlds_all)
        report['baselines'][baseline]['new_150'] = assisted_metrics(reviewed_tasks, reviewed, balls_all[key], dataset['sources'])
        for name in NAMES:
            ids = {t['task_id'] for t in reviewed_tasks if t['recording_id'] == name}
            report['baselines'][baseline]['per_recording'][name] = {'new_150': assisted_metrics(
                [t for t in reviewed_tasks if t['recording_id'] == name], [r for r in reviewed if r['task_id'] in ids], balls_all[key], dataset['sources'])}
    expected = {'m20_control': (41, 3, 104), 'm21_temporal_offline': (95, 14, 50), 'm22_reviewed': (64, 27, 81)}
    for method, values in expected.items():
        actual = report['baselines'][method]['new_150']['ball']
        if tuple(actual[k] for k in ['tp', 'fp', 'fn']) != values:
            raise ValueError('Frozen baseline development counts changed')
    full = report['variants']['full_m23']
    b, p = full['new_150']['ball'], full['original_references']['passes']
    targets = {'precision': (b['precision'] or 0) >= .85, 'recall': (b['recall'] or 0) >= .75,
        'f1': (b['f1'] or 0) >= .80, 'correct_balls': b['tp'] >= 109,
        'known_pass_preserved': p['tp'] >= 1, 'aspirational_passes': p['tp'] >= 3, 'no_false_passes_in_reviewed_windows': p['fp'] == 0}
    report['research_targets'] = {'achieved': targets, 'primary_all_met': all(targets[k] for k in ['precision', 'recall', 'f1', 'correct_balls']),
        'independent_accuracy_acceptance': False}
    report['all_seven_passes_diagnosed'] = sum(len(full['per_recording'][n]['pass_traces']) for n in NAMES) == 7
    if not report['all_seven_passes_diagnosed']:
        raise ValueError('Seven original reference passes required')
    write(root/'report.json', report)
    print('M23 COMPLETE', full['new_150']['ball'], 'passes', p, 'targets', targets, flush=True)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', required=True)
    p.add_argument('--perception', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    run(a.inputs, a.perception, a.output)
