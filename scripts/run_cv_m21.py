"""Fresh small-ball/temporal inference campaign; experiments remain opt-in and isolated."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import time
import cv2
import numpy as np

from src.serving.cv import digest
from src.vision.ball.geometry import camera_motion
from src.vision.advanced21.camera import refine_motion, causal_camera
from src.vision.advanced21.detection import DetectorBackend, tile_regions, merge_candidates, past_frame_fusion, bounded_region
from src.vision.advanced21.experiment import NAMES, fresh_directory, write_json, source_worlds, frozen_m20
from src.vision.advanced21.temporal import batch_visual_features, deterministic_verdict


def run(output, m20_reference, audit_root):
    config = json.loads(Path('configs/vision/cv_m21.json').read_text())
    audit = json.loads((Path(audit_root)/'audit.json').read_text())
    if not (Path(audit_root)/'baseline_reproduction.json').exists():
        raise ValueError('Reproduce the M20 baseline before new experiments')
    root = fresh_directory(output)
    write_json(root/'config.json', config)
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    checkpoint = config['detectors']['generic']
    if Path(checkpoint).stat().st_blocks == 0:
        raise ValueError('Download the existing checkpoint from iCloud first')
    model = YOLO(checkpoint)
    if model.names[32] != 'sports ball':
        raise ValueError('Generic detector class mismatch')
    detector = DetectorBackend(model, digest(checkpoint), 32, device=device)
    report = {'schema_version': 'cv-m21-perception-v1', 'config_sha256': digest(root/'config.json'),
        'm20_reference': str(m20_reference), 'development_annotation_sha256': audit['annotations_sha256'],
        'dataset_fingerprint': audit['dataset_fingerprint'], 'reference_role': 'development_only',
        'final_test_used': False, 'training': 'Frozen existing YOLO11s; no detector fine-tuning on the 36 ball tasks.',
        'runtime': {'torch': torch.__version__, 'opencv': cv2.__version__, 'device': device,
                    'platform': platform.platform(), 'cpu_count': os.cpu_count()},
        'checkpoint_sha256': detector.checkpoint_sha256, 'clips': {}}
    write_json(root/'report.json', report)
    for name in NAMES:
        metadata, worlds, people = source_worlds(name)
        baseline, _ = frozen_m20(name, m20_reference)
        if len(baseline) != len(worlds):
            raise ValueError('M20 control frame mismatch')
        # Development pilot selected 320px generic crops; holdout labels are not accessed.
        regions = tile_regions(metadata['width'], metadata['height'], 320, config['detectors']['tile_overlap'])
        directory = root/name
        directory.mkdir()
        capture = cv2.VideoCapture(metadata['source_video'])
        previous = previous_camera = None
        previous_people = []
        history, history_motion = [], []
        started = time.perf_counter()
        timings = {'dense_seconds': 0., 'fusion_seconds': 0., 'current_verification_seconds': 0., 'temporal_camera_seconds': 0.}
        view_counts = {'dense': 0, 'fusion': 0, 'current_verification': 0}
        try:
            with (directory/'inputs.jsonl').open('w') as cache:
                for i, world in enumerate(worlds):
                    ok, image = capture.read()
                    if not ok:
                        raise ValueError('Source video ended before its verified clock')
                    camera = world['camera']
                    epoch_changed = previous_camera is not None and (camera['epoch'] != previous_camera['epoch'] or camera.get('continuity') == 'cut')
                    if epoch_changed:
                        history, history_motion = [], []
                        previous = None
                    tick = time.perf_counter()
                    motion = refine_motion(previous, image, previous_camera, camera, previous_people)
                    original_motion, _ = camera_motion(causal_camera(previous_camera) if previous_camera else None, causal_camera(camera))
                    step = np.asarray(motion['homography']) if motion['accepted'] else original_motion
                    history_motion = [step@old if old is not None and step is not None else None for old in history_motion]
                    fused, history_used = past_frame_fusion(image, history, history_motion)
                    timings['temporal_camera_seconds'] += time.perf_counter()-tick
                    dense, dt = detector.infer(image, regions, 'current_generic_dense320')
                    timings['dense_seconds'] += dt['seconds']; view_counts['dense'] += dt['views']
                    proposals = []
                    if history_used:
                        proposals, ft = detector.infer(fused, [(0, 0, metadata['width'], metadata['height'])], 'PAST_FRAME_FUSION_PROPOSAL_NOT_OBSERVED')
                        timings['fusion_seconds'] += ft['seconds']; view_counts['fusion'] += ft['views']
                    proposed_regions = list(dict.fromkeys(bounded_region((c['bbox'][0]+c['bbox'][2]/2, c['bbox'][1]+c['bbox'][3]/2),
                        metadata['width'], metadata['height'], config['detectors']['current_verification_region'])
                        for c in proposals if c['confidence'] >= config['detectors']['visual_threshold']))[:4]
                    confirmed, vt = detector.infer(image, proposed_regions, 'current_pixels_after_temporal_proposal')
                    timings['current_verification_seconds'] += vt['seconds']; view_counts['current_verification'] += vt['views']
                    groups = {'m20_candidates': baseline[i]['candidates'],
                              'dense': merge_candidates(baseline[i]['candidates'], dense),
                              'multi_frame': merge_candidates(baseline[i]['candidates'], dense, confirmed)}
                    unique = {(tuple(c['bbox']), c['confidence']): c for candidates in groups.values() for c in candidates}
                    computed = dict(zip(unique, batch_visual_features(previous, image, list(unique.values()), step), strict=True))
                    for candidates in groups.values():
                        for c in candidates:
                            key = (tuple(c['bbox']), c['confidence'])
                            c['temporal_features'] = computed[key]
                            c['deterministic'] = deterministic_verdict(computed[key], c['confidence'])
                    cache.write(json.dumps({'frame': i, 'timestamp': world['timestamp'], 'camera_epoch': camera['epoch'],
                        'source_pixels_sha256': hashlib.sha256(image.tobytes()).hexdigest(),
                        'candidate_sets': groups, 'fusion_history_frames': history_used,
                        'fusion_proposals_not_direct_evidence': proposals, 'current_verification_regions': proposed_regions,
                        'camera_refinement': motion, 'player_observations': people[i]}, separators=(',', ':'), allow_nan=False)+'\n')
                    history = (history+[image.copy()])[-2:]
                    history_motion = (history_motion+[np.eye(3)])[-2:]
                    previous, previous_camera, previous_people = image, camera, people[i]
                    if i % 100 == 0:
                        cache.flush()
                        print('M21 PERCEPTION', name, i, '/', len(worlds), 'seconds', round(time.perf_counter()-started, 1), flush=True)
                if capture.read()[0]:
                    raise ValueError('Unindexed source frames')
        finally:
            capture.release()
        elapsed = time.perf_counter()-started
        report['clips'][name] = {'frames': len(worlds), 'fps': metadata['fps'], 'source_video_sha256': metadata['video_sha256'],
            'world_frames_sha256': metadata['frames_sha256'], 'inputs_sha256': digest(directory/'inputs.jsonl'),
            'wall_seconds': elapsed, 'processing_fps': len(worlds)/elapsed, 'timings': timings, 'detector_views': view_counts,
            'max_rss_bytes_macos': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'mps_allocated_bytes': torch.mps.current_allocated_memory() if device == 'mps' else None}
        write_json(root/'report.json', report)
        print('M21 PERCEPTION COMPLETE', name, json.dumps(report['clips'][name]), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--m20-reference', required=True)
    parser.add_argument('--audit', required=True)
    args = parser.parse_args()
    run(args.output, args.m20_reference, args.audit)
