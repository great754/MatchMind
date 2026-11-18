"""Fresh trajectory-guided crop inference and reproducible frozen-input replay."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import cv2
import numpy as np
from scripts.run_cv_m19 import compact_worlds, ball_metrics
from scripts.process_cv_passes import export_tables
from src.serving.cv import digest
from src.vision.ball.schema import canonical
from src.vision.ball19.tracker import VisualBallTracker, AssociationConfig
from src.vision.ball20.tracker import TrajectoryBallTracker
from src.vision.ball20.detector import YoloCropAdapter, merge_candidates
from src.vision.events.passes import signature
from src.vision.events.passes_m19 import detect_m19

NAMES = ['test_match', 'team_goal', 'southampton']


def timestamps(video):
    binary = shutil.which('ffprobe')
    if not binary: raise ValueError('ffprobe is required to verify the source presentation clock')
    raw = subprocess.check_output([binary, '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                                   'frame=best_effort_timestamp_time', '-of', 'json', str(video)])
    rows = json.loads(raw)['frames']
    return [float(row['best_effort_timestamp_time']) for row in rows]


def write_ball(directory, name, metadata, rows, detector_hash, version, extras=None):
    directory.mkdir(); offsets = []
    with (directory/'ball_frames.jsonl').open('wb') as target, (directory/'video_ball_observations.jsonl').open('w') as stream:
        for state in rows:
            offsets.append(target.tell()); target.write((state.model_dump_json()+'\n').encode())
            stream.write(canonical(name, metadata['fps'], state, detector_hash).model_dump_json()+'\n')
    manifest = {k: metadata[k] for k in ['recording_id', 'frames', 'fps', 'width', 'height', 'video_sha256']}
    manifest.update(version=version, source_m9_frames_sha256=metadata['frames_sha256'], detector_sha256=detector_hash,
                    ball_frames_sha256=digest(directory/'ball_frames.jsonl'), canonical_sha256=digest(directory/'video_ball_observations.jsonl'),
                    frame_offsets=offsets, statistics=ball_metrics([r.model_dump() for r in rows]), **(extras or {}))
    (directory/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def event_outputs(directory, name, worlds, rows, metadata):
    directory.mkdir(); events, summary, evidence = detect_m19(name, worlds, [r.model_dump() for r in rows], metadata['fps'])
    (directory/'events.json').write_text(json.dumps(events, indent=2)+'\n')
    (directory/'evidence_frames.jsonl').write_text(''.join(json.dumps(r, separators=(',', ':'))+'\n' for r in evidence))
    export_tables(name, directory, events, evidence, metadata['fps'])
    return summary


def trajectory_metrics(debug, metadata, frozen_m19):
    rows = [row.model_dump() for row in debug]
    candidates = [c for r in rows for c in r['candidates']]
    regions = [region for r in rows for region in r['search_regions']]
    proxy_frames = proxy_with_regions = proxy_hits = 0
    for row, reference in zip(rows, frozen_m19, strict=True):
        if not reference['directly_observed']: continue
        proxy_frames += 1
        if row['search_regions']: proxy_with_regions += 1
        x, y = reference['image_position']
        proxy_hits += any(a <= x <= a+w and b <= y <= b+h for a, b, w, h in [r['bbox'] for r in row['search_regions']])
    inference = np.array([r['inference_seconds'] for r in rows])
    return {'states': dict(Counter(r['state'] for r in rows)), 'events': dict(Counter(e for r in rows for e in r['events'])),
            'candidate_decisions': dict(Counter(c['decision'] for c in candidates)),
            'candidate_reasons': dict(Counter(c['reason'] for c in candidates)),
            'ambiguity_frame_fraction': sum(r['state']=='AMBIGUOUS' for r in rows)/len(rows),
            'global_detection_frames': sum(r['global_inference'] for r in rows), 'real_crop_views': len(regions),
            'frames_with_regions': sum(bool(r['search_regions']) for r in rows),
            'mean_sum_crop_area_fraction': sum(r['bbox'][2]*r['bbox'][3] for r in regions)/(len(rows)*metadata['width']*metadata['height']),
            'mean_prediction_radius_px': float(np.mean([h['radius90_px'] for r in rows for h in r['hypotheses']])) if any(r['hypotheses'] for r in rows) else None,
            'inference_seconds': float(inference.sum()), 'inference_median_seconds': float(np.median(inference)),
            'inference_p95_seconds': float(np.quantile(inference, .95)),
            'crop_inference_seconds': sum(r['crop_inference_seconds'] for r in rows),
            'tracker_seconds': sum(r['tracker_seconds'] for r in rows),
            'next_frame_search_proxy': {'reference': 'frozen_M19_model_observations_not_human_truth',
                                        'reference_frames': proxy_frames, 'reference_frames_with_regions': proxy_with_regions,
                                        'hits': proxy_hits, 'hit_fraction_all_reference_frames': proxy_hits/proxy_frames if proxy_frames else None,
                                        'hit_fraction_when_regions_exist': proxy_hits/proxy_with_regions if proxy_with_regions else None},
            'independent_detection_precision': None, 'independent_detection_recall': None, 'independent_localization_error': None,
            'independent_reacquisition_success': None, 'incorrect_ball_associations': None, 'pass_precision_recall_f1': None}


def run(output, replay=None, recordings=None):
    root = Path(output)
    if root.exists(): raise ValueError('Use a new output directory; prior experiments are immutable')
    config = json.loads(Path('configs/vision/cv_m20.json').read_text())
    if digest(Path(config['detector']['checkpoint'])) != config['detector']['sha256']: raise ValueError('Checkpoint hash mismatch')
    root.mkdir(parents=True)
    detector = config['detector']; model = adapter = None
    import torch
    torch.set_num_threads(4); cv2.setNumThreads(4)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    load_started = time.perf_counter()
    if replay is None:
        from ml.ball16.detector import make_model
        model = make_model(detector['checkpoint']); adapter = YoloCropAdapter(model, detector['crop_resolution'], device)
    load_seconds = time.perf_counter()-load_started
    report = {'version': 'cv-m20-v1', 'config': config, 'config_sha256': signature(config), 'clips': {},
              'runtime': {'torch': torch.__version__, 'opencv': cv2.__version__, 'device': device, 'model_load_seconds': load_seconds},
              'detector_inference': 'fresh_global_and_prediction_crops' if replay is None else 'hash_checked_visual_input_replay',
              'independent_accuracy': 'pending_human_review', 'default_changed': False}
    for name in recordings or NAMES:
        print('M20 VALIDATE', name, flush=True)
        base = Path('outputs/world_model')/(name+'_m9_ready'); metadata = json.loads((base/'manifest.json').read_text())
        if digest(base/'frames.jsonl') != metadata['frames_sha256'] or digest(Path(metadata['source_video'])) != metadata['video_sha256']: raise ValueError('Original source changed')
        world = compact_worlds(base/'frames.jsonl'); pts = timestamps(metadata['source_video'])
        if len(pts) != len(world) or any(abs(a-b['timestamp']) > 2e-6 for a, b in zip(pts, world)): raise ValueError('Source PTS/world clock mismatch')
        directory = root/name; directory.mkdir()
        old_m19 = Path('outputs/m19/run_v3')/name/'ball_m16'
        old_manifest = json.loads((old_m19/'manifest.json').read_text())
        if digest(old_m19/'ball_frames.jsonl') != old_manifest['ball_frames_sha256']: raise ValueError('M19 baseline changed')
        frozen_m19 = [json.loads(line) for line in (old_m19/'ball_frames.jsonl').open()]
        prior_metrics = {}
        for version in ['m14', 'm16']:
            path = Path('outputs/world_model')/('ball_'+version)/name; manifest = json.loads((path/'manifest.json').read_text())
            if digest(path/'ball_frames.jsonl') != manifest['ball_frames_sha256']: raise ValueError('Ball baseline changed')
            with (path/'ball_frames.jsonl').open() as source: prior_metrics[version] = ball_metrics([json.loads(line) for line in source])
        prior_metrics['m19'] = ball_metrics(frozen_m19)
        baseline = VisualBallTracker(metadata['width'], metadata['height'], detector['threshold'], AssociationConfig(**config['association']))
        tracker = TrajectoryBallTracker(metadata['width'], metadata['height'], config)
        balls = []; baseline_balls = []; debug_rows = []; global_seconds = []
        replay_source = None
        if replay:
            old = Path(replay)/name; manifest = json.loads((old/'ball_m20/manifest.json').read_text())
            if manifest['config_sha256'] != signature(config) or digest(old/'detector_inputs.jsonl') != manifest['detector_inputs_sha256'] or manifest['video_sha256'] != metadata['video_sha256']: raise ValueError('Inference replay hash/config/source mismatch')
            replay_source = (old/'detector_inputs.jsonl').open()
        capture = cv2.VideoCapture(metadata['source_video']); started = time.perf_counter()
        try:
            with (directory/'detector_inputs.jsonl').open('w') as cache:
                for frame, (w, timestamp) in enumerate(zip(world, pts, strict=True)):
                    ok, image = capture.read()
                    if not ok: raise ValueError('Source video ended before its verified frame clock')
                    people = [{'bbox': e['image_bbox']} for e in w['entities']]+[{'bbox': e['observation']['bbox']} for e in w['unassigned']]
                    regions, global_needed = tracker.prepare(frame, timestamp, w['camera'], people)
                    pixels_sha = hashlib.sha256(image.tobytes()).hexdigest()
                    tick = time.perf_counter()
                    if replay_source:
                        record = json.loads(replay_source.readline())
                        for key, expected in [('frame', frame), ('timestamp', timestamp), ('regions', regions), ('source_pixels_sha256', pixels_sha), ('global_needed', global_needed)]:
                            if record[key] != expected: raise ValueError('Prediction/crop/source replay mismatch: '+key)
                        global_candidates = record['global_candidates']; local_candidates = record['local_candidates']
                        global_timing = {'seconds': 0.}; local_timing = {'seconds': 0.}
                    else:
                        from ml.ball16.detector import infer
                        # Interval one is the predeclared fair global-detector control.
                        if not global_needed: raise ValueError('This matched-control experiment requires global_interval_frames=1')
                        global_candidates, global_timing = infer(model, image, detector['global_resolution'])
                        local_candidates, local_timing = adapter.infer(image, regions)
                        record = {'frame': frame, 'timestamp': timestamp, 'regions': regions, 'global_needed': global_needed,
                                  'source_pixels_sha256': pixels_sha, 'global_candidates': global_candidates, 'local_candidates': local_candidates,
                                  'global_inference_seconds': global_timing['seconds'], 'crop_inference_seconds': local_timing['seconds']}
                    cache.write(json.dumps(record, separators=(',', ':'))+'\n')
                    combined = merge_candidates(global_candidates, local_candidates)
                    ball, debug = tracker.consume(image, combined, inference_seconds=global_timing['seconds']+local_timing['seconds'], crop_seconds=local_timing['seconds'])
                    control = baseline.step(frame, timestamp, global_candidates, w['camera'], people)
                    balls.append(ball); baseline_balls.append(control); debug_rows.append(debug); global_seconds.append(global_timing['seconds'])
                    if frame % 100 == 0: print('M20 FRAME', name, frame, debug.state, 'regions', len(regions), 'seconds', round(time.perf_counter()-tick, 3), flush=True)
                if capture.read()[0]: raise ValueError('Unindexed source video frames')
                if replay_source and replay_source.read(1): raise ValueError('Unindexed detector replay frames')
        finally:
            capture.release()
            if replay_source: replay_source.close()
        wall = time.perf_counter()-started
        debug_path = directory/'trajectory_frames.jsonl'; debug_offsets = []
        with debug_path.open('wb') as target:
            for row in debug_rows: debug_offsets.append(target.tell()); target.write((row.model_dump_json()+'\n').encode())
        extras = {'config': config, 'config_sha256': signature(config), 'detector_inputs_sha256': digest(directory/'detector_inputs.jsonl'),
                  'trajectory_sha256': digest(debug_path), 'trajectory_frame_offsets': debug_offsets, 'source_pts_sha256': signature(pts),
                  'inference_reused': replay is not None, 'independent_accuracy': 'pending_human_review'}
        installed = write_ball(directory/'ball_m20', name, metadata, balls, detector['sha256'], 'm20', extras)
        write_ball(directory/'ball_m19_control', name, metadata, baseline_balls, detector['sha256'], 'm19')
        summary = event_outputs(directory/'passes_m20', name, world, balls, metadata)
        control_summary = event_outputs(directory/'passes_m19_control', name, world, baseline_balls, metadata)
        clip = {'source': {k: metadata[k] for k in ['video_sha256', 'frames_sha256', 'frames', 'fps', 'width', 'height']},
                'source_pts_sha256': signature(pts), 'frozen_baselines': prior_metrics,
                'm19_same_global_inputs': {'ball': ball_metrics([r.model_dump() for r in baseline_balls]), 'passes': control_summary, 'inference_seconds': sum(global_seconds)},
                'm20': {'ball': installed['statistics'], 'passes': summary, 'trajectory': trajectory_metrics(debug_rows, metadata, frozen_m19),
                        'processing_wall_seconds': wall, 'processing_fps': len(world)/wall}}
        report['clips'][name] = clip
        (root/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print('M20 RESULT', name, json.dumps({'observed': installed['statistics']['direct_observed_frames'], 'passes': summary['completed_pass_estimates'], 'candidates': summary['transfer_candidates'], 'wall_seconds': wall}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True); parser.add_argument('--replay-inference'); parser.add_argument('--recording', choices=NAMES, action='append')
    args = parser.parse_args(); run(args.output, args.replay_inference, args.recording)
