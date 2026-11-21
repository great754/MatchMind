"""Fresh M21 visual inference and one integrated, controlled M23 search campaign."""
import argparse
import hashlib
import json
import resource
import time
from pathlib import Path
import cv2
import numpy as np
from src.serving.cv import digest
from src.vision.intelligent23.inputs import Inputs, NAMES, write
from src.vision.intelligent23.detection import TemporalPrimary, region_candidates
from src.vision.intelligent23.memory import BallMemory
from src.vision.intelligent23.verification import combine, verify
from src.vision.advanced21.camera import causal_camera
from src.vision.advanced21.temporal import batch_visual_features, deterministic_verdict
from src.vision.football22.model import context_indices

METHODS = ('guided_search', 'guided_verified', 'full_m23')


def run(inputs_root, output, resume=False):
    import torch
    from ultralytics import YOLO
    from src.vision.football22.inference import FootballDetector
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    inputs = Inputs(inputs_root)
    config_path = Path('configs/vision/cv_m23.json')
    config = json.loads(config_path.read_text())
    m20 = inputs.json('configs/vision/cv_m20.json')
    m22_config = inputs.json('configs/vision/cv_m22.json')
    if 'complement_models' not in inputs.manifest or 'm22_global_root' not in inputs.manifest:
        raise ValueError('M23 corrected integration requires the explicitly pinned complementary visual cache')
    root = Path(output)
    if root.exists() and not resume:
        raise ValueError('Use a fresh immutable M23 campaign or explicit same-campaign resume')
    root.mkdir(parents=True, exist_ok=True)
    source_files = ['scripts/run_cv_m23.py',
        *['src/vision/intelligent23/'+p+'.py' for p in ['inputs', 'detection', 'memory', 'verification']],
        *['src/vision/advanced21/'+p+'.py' for p in ['detection', 'temporal', 'camera']],
        *['src/vision/ball20/'+p+'.py' for p in ['tracker', 'motion', 'detector']],
        'src/vision/ball19/tracker.py', 'src/vision/ball/geometry.py',
        'src/vision/football22/inference.py', 'src/vision/football22/model.py', 'configs/vision/cv_m23.json']
    frozen = {'schema_version': 'cv-m23-perception-v1', 'input_manifest_sha256': inputs.sha256,
        'config': config, 'config_sha256': digest(config_path), 'source_sha256': {p: digest(p) for p in source_files},
        'device': 'mps' if torch.backends.mps.is_available() else 'cpu', 'torch': torch.__version__, 'opencv': cv2.__version__,
        'reference_role': 'development_only', 'new_training': False, 'fresh_primary_current_pixel_inference': True,
        'm20_global_candidates': 'Explicit hash-pinned frozen M20 visual measurements reused as in M21',
        'labels_read_during_perception': False, 'clips': {}}
    report_path = root/'report.json'
    if report_path.exists():
        previous = json.loads(report_path.read_text())
        if any(previous[k] != v for k, v in frozen.items() if k != 'clips'):
            raise ValueError('M23 resume source/config/runtime mismatch')
        frozen = previous
    write(report_path, frozen)
    import shutil
    for source in source_files:
        destination = root/'source_snapshot'/source
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            destination.chmod(0o400)
        if digest(destination) != frozen['source_sha256'][source]:
            raise ValueError('Perception source snapshot changed')
    generic = inputs.path('outputs/ml/ball_m16/yolo11s.pt')
    generic_sha = digest(generic)
    if generic_sha != '85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5':
        raise ValueError('Frozen M21 generic weights changed')
    model = YOLO(generic)
    if model.names[32] != 'sports ball':
        raise ValueError('Wrong primary visual class')
    for name in NAMES:
        directory = root/name
        directory.mkdir(exist_ok=True)
        path = directory/'inputs.jsonl'
        if name in frozen['clips']:
            if digest(path) != frozen['clips'][name]['inputs_sha256']:
                raise ValueError('Completed M23 perception changed')
            print('M23 verified completed inference', name, flush=True)
            continue
        meta, worlds, people = inputs.worlds(name)
        control = [json.loads(line) for line in inputs.path(Path('outputs/m20/run_v1')/name/'ball_m20/ball_frames.jsonl').open()]
        primary = TemporalPrimary(model, generic_sha, meta['width'], meta['height'], frozen['device'])
        memories = {method: BallMemory(meta['width'], meta['height'], m20, config) for method in METHODS}
        trained = inputs.manifest['complement_models'][name]
        global_m22_path = inputs.path(Path(inputs.manifest['m22_global_root'])/name/'detections.jsonl')
        global_m22 = [json.loads(line) for line in global_m22_path.open()]
        if len(global_m22) != meta['frames'] or any(r['frame'] != i or r['checkpoint_sha256'] != trained['checkpoint_sha256'] or
            len(r['context_frames']) != 3 or r['context_frames'][-1] != i or sorted(r['context_frames']) != r['context_frames'] or
            min(r['context_frames']) < 0 for i, r in enumerate(global_m22)):
            raise ValueError('Complementary global source/checkpoint clock mismatch')
        complement = FootballDetector(inputs.path(trained['checkpoint']), trained['checkpoint_sha256'], m22_config, device=frozen['device'])
        if complement.metadata['split']['excluded_development_source'] != name:
            raise ValueError('M22 target source entered fitting/selection')
        previous_records = []
        if path.exists():
            complete = []
            for line in path.read_bytes().splitlines(keepends=True):
                try:
                    row = json.loads(line)
                except ValueError:
                    break
                if not line.endswith(b'\n'):
                    break
                if row['frame'] != len(complete) or row['config_sha256'] != frozen['config_sha256']:
                    raise ValueError('Invalid partial M23 perception')
                complete.append(line)
                previous_records.append(row)
            path.write_bytes(b''.join(complete))
        capture = cv2.VideoCapture(meta['source_video'])
        epochs = []
        epoch = 0
        for i, world in enumerate(worlds):
            if i and (world['camera']['epoch'] != worlds[i-1]['camera']['epoch'] or world['camera'].get('continuity') == 'cut'):
                epoch += 1
            epochs.append(epoch)
        history = {}
        started = time.perf_counter()
        totals = {'primary_neural_seconds': 0., 'primary_seconds': 0., 'search_neural_seconds': 0.,
                  'm22_neural_seconds': 0., 'verification_memory_seconds': 0., 'extra_crop_views': 0, 'm22_views': 0, 'primary_views': 0}
        try:
            with path.open('a') as stream:
                for i, world in enumerate(worlds):
                    tick = time.perf_counter()
                    ok, image = capture.read()
                    if not ok:
                        raise ValueError('Approved source video ended before its pinned clock')
                    pixel_sha = hashlib.sha256(image.tobytes()).hexdigest()
                    if global_m22[i]['current_frame_pixels_sha256'] != pixel_sha:
                        raise ValueError('Complementary model cache does not describe these actual source pixels')
                    history[i] = image
                    for old in list(history):
                        if old < i-6:
                            del history[old]
                    old_image = primary.previous
                    if i < len(previous_records):
                        row = previous_records[i]
                        if row['source_pixels_sha256'] != pixel_sha:
                            raise ValueError('Resumed source pixels changed')
                        for method, memory in memories.items():
                            memory.prepare(i, i/meta['fps'], row['causal_camera'], people[i])
                            memory.consume(image, [c for c in row['methods'][method]['candidates'] if c['observation_eligible']])
                        # Rebuild causal fusion from the actual replayed source frames.
                        step = np.asarray(row['relative_camera_homography']) if row['relative_camera_homography'] is not None else None
                        primary.history_motion = [step@h if step is not None and h is not None else None for h in primary.history_motion]
                        if i and (world['camera']['epoch'] != worlds[i-1]['camera']['epoch'] or world['camera'].get('continuity') == 'cut'):
                            primary.history, primary.history_motion = [], []
                        primary.history = (primary.history+[image.copy()])[-2:]
                        primary.history_motion = (primary.history_motion+[np.eye(3)])[-2:]
                        primary.previous, primary.previous_camera, primary.previous_people = image, world['camera'], people[i]
                        for key in totals:
                            totals[key] += row['timing'][key]
                        continue
                    refinement, step, camera_source = primary.camera(image, world['camera'])
                    old_image = primary.previous
                    camera = causal_camera(world['camera'], refinement)
                    plans = {method: memory.prepare(i, i/meta['fps'], camera, people[i]) for method, memory in memories.items()}
                    current, timing = primary.detect(image, world['camera'], people[i], control[i]['candidates'], refinement, step)
                    requested = {method: [tuple(r['bbox']) for r in plans[method]['regions']] for method in METHODS}
                    unique_regions = sorted(set(r for regions in requested.values() for r in regions))
                    local, local_timing = region_candidates(primary.detector, image, unique_regions)
                    flat = [c for rows in local.values() for c in rows]
                    local_features = batch_visual_features(old_image, image, flat, step)
                    for c, feature in zip(flat, local_features, strict=True):
                        c.update(temporal_features=feature, deterministic=deterministic_verdict(feature, c['confidence']))
                    m22_regions = requested['full_m23'][:config['verification']['maximum_m22_regions']]
                    if not m22_regions and sum(c['confidence'] >= .35 for c in current) > 1:
                        m22_regions = [(0, 0, meta['width'], meta['height'])]
                    m22 = {'candidates': [], 'seconds': 0., 'inference_views': 0}
                    if m22_regions:
                        ids = context_indices(i, meta['fps'], epochs)
                        m22 = complement.infer([history[j] for j in ids], i, regions=m22_regions)
                    complementary_pool = [*m22['candidates'], *global_m22[i]['candidates']]
                    complements = []
                    from src.evaluation.cv_m19 import iou
                    full_local = [c for region in requested['full_m23'] for c in local[region]]
                    existing_eligible = [c for c in [*current, *full_local] if c['confidence'] >= .35 and c['deterministic']['accepted']]
                    for c in complementary_pool:
                        if not c['observation_eligible'] or any(iou(c['bbox'], p['bbox']) >= .1 for p in existing_eligible):
                            continue
                        complements.append({'candidate_id': c['candidate_id'], 'bbox': c['bbox'], 'confidence': c['confidence'],
                            'current_only_confidence': c['current_only_confidence'], 'inference_view': 'M23_M22_TEMPORAL_WITH_ACTUAL_CURRENT_PIXEL_CHECK',
                            'source': 'VIDEO_CV_DETECTOR', 'origin': 'complement', 'visual_evidence': 'CURRENT_PIXELS',
                            'source_model_sha256': trained['checkpoint_sha256']})
                    complement_features = batch_visual_features(old_image, image, complements, step)
                    for c, feature in zip(complements, complement_features, strict=True):
                        c.update(temporal_features=feature, deterministic=deterministic_verdict(feature, c['confidence']))
                    verify_started = time.perf_counter()
                    methods = {}
                    for method, memory in memories.items():
                        discovery = {tuple(r['bbox']) for r in plans[method]['regions'] if r['kind'] == 'global_discovery'}
                        local_for_method = [{**c, 'origin': 'discovery' if region in discovery else 'guided'}
                            for region in requested[method] for c in local[region]]
                        candidates = combine(current, local_for_method+
                            (complements if method == 'full_m23' else []), i)
                        checked = [verify(c, memory, image, config, additional_verification=method != 'guided_search',
                            m22_candidates=complementary_pool if method == 'full_m23' else ()) for c in candidates]
                        ball, snapshot = memory.consume(image, [c for c in checked if c['observation_eligible']])
                        methods[method] = {'candidates': checked, 'memory': snapshot,
                            'causal_selected_bbox': list(ball.image_bbox) if ball.image_bbox is not None else None,
                            'requested_crop_views': len(requested[method])}
                    clock = {'primary_neural_seconds': timing['neural_seconds'], 'primary_seconds': timing['seconds'],
                        'search_neural_seconds': local_timing['seconds'], 'm22_neural_seconds': m22['seconds'],
                        'verification_memory_seconds': time.perf_counter()-verify_started, 'extra_crop_views': local_timing['views'],
                        'm22_views': m22['inference_views'], 'primary_views': timing['views']}
                    for key in totals:
                        totals[key] += clock[key]
                    row = {'frame': i, 'timestamp': i/meta['fps'], 'source_pixels_sha256': pixel_sha,
                        'config_sha256': frozen['config_sha256'], 'primary_candidates': current,
                        'camera_refinement': refinement, 'causal_camera': camera, 'methods': methods,
                        'relative_camera_homography': step.tolist() if step is not None else None,
                        'm22_context_frames': context_indices(i, meta['fps'], epochs) if m22_regions else [],
                        'm22_checkpoint_sha256': trained['checkpoint_sha256'], 'm22_candidates': m22['candidates'],
                        'm22_frozen_global_source_sha256': inputs.manifest['files'][str(Path(inputs.manifest['m22_global_root'])/name/'detections.jsonl')]['sha256'],
                        'timing': clock, 'frame_wall_seconds': time.perf_counter()-tick}
                    stream.write(json.dumps(row, separators=(',', ':'), allow_nan=False)+'\n')
                    stream.flush()
                    if i % 100 == 0:
                        print('M23 INFER', name, i, '/', meta['frames'], 'guided crops', clock['extra_crop_views'], flush=True)
            if capture.read()[0]:
                raise ValueError('Unindexed source video frames')
        finally:
            capture.release()
        elapsed = time.perf_counter()-started
        actual_rows = [json.loads(line) for line in path.open()]
        processing_seconds = sum(r['frame_wall_seconds'] for r in actual_rows)
        frozen['clips'][name] = {'frames': meta['frames'], 'source_video_sha256': meta['video_sha256'],
            'source_world_sha256': meta['frames_sha256'], 'inputs_sha256': digest(path), 'timing': totals,
            'wall_seconds_this_process': elapsed, 'frames_per_second_this_process': meta['frames']/elapsed,
            'recorded_processing_seconds': processing_seconds, 'processing_fps': meta['frames']/processing_seconds,
            'm22_checkpoint_sha256': trained['checkpoint_sha256'], 'generic_checkpoint_sha256': generic_sha,
            'peak_rss_bytes_macos': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'mps_allocated_bytes': torch.mps.current_allocated_memory() if frozen['device'] == 'mps' else None}
        write(report_path, frozen)
        print('M23 INFER COMPLETE', name, frozen['clips'][name], flush=True)
        del complement
        if frozen['device'] == 'mps':
            torch.mps.empty_cache()
    return frozen


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    run(a.inputs, a.output, a.resume)
