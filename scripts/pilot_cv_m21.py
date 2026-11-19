"""Bounded small-ball detector selection on EXPOSED development frames, not final TEST."""
import argparse
import json
from pathlib import Path
import resource
import time
import cv2
import numpy as np
from src.serving.cv import digest
from src.evaluation.cv_m19 import counts, iou
from src.vision.advanced21.detection import DetectorBackend, tile_regions


def run(output):
    root = Path(output)
    if root.exists():
        raise ValueError('Use a fresh experiment directory')
    root.mkdir(parents=True)
    config = json.loads(Path('configs/vision/cv_m21.json').read_text())
    dataset = json.loads(Path('configs/evaluation/cv_m19_tasks.json').read_text())
    labels_path = Path('outputs/m21/audit_v1/development_labels.json')
    labels = json.loads(labels_path.read_text())['annotations']
    lookup = {r['task_id']: r for r in labels}
    tasks = [t for t in dataset['tasks'] if t['kind'] == 'ball']
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    results = {'reference_role': 'development_only', 'dataset_fingerprint': dataset['fingerprint'],
               'labels_sha256': digest(labels_path), 'config': config, 'variants': {},
               'final_test_used': False, 'runtime': {'torch': torch.__version__, 'opencv': cv2.__version__, 'device': device}}
    for name, class_id in [('soccer', 0), ('generic', 32)]:
        path = config['detectors'][name]
        model = YOLO(path)
        if name == 'generic' and model.names[class_id] != 'sports ball':
            raise ValueError('Unexpected generic model class map')
        detector = DetectorBackend(model, digest(path), class_id, device=device)
        for size in config['detectors']['tile_sizes']:
            variant = f'{name}_tiles{size}'
            frames = []
            started = time.perf_counter()
            for task in tasks:
                clip = task['recording_id']
                capture = cv2.VideoCapture('data/raw/'+clip+'.mp4')
                try:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, task['frame'])
                    ok, image = capture.read()
                finally:
                    capture.release()
                if not ok:
                    raise ValueError('Development source frame unavailable')
                candidates, timing = detector.infer(image, tile_regions(image.shape[1], image.shape[0], size, config['detectors']['tile_overlap']))
                row = {'task_id': task['task_id'], 'recording_id': clip, 'frame': task['frame'], 'candidates': candidates, 'timing': timing}
                frames.append(row)
            metrics = {}
            for clip in [*dataset['sources'], 'all']:
                tp = fp = fn = known = unknown = possible = 0
                for row in frames:
                    if clip != 'all' and row['recording_id'] != clip:
                        continue
                    label = lookup[row['task_id']]
                    if label['state'] == 'uncertain':
                        unknown += 1
                        continue
                    known += 1
                    predictions = [c for c in row['candidates'] if c['confidence'] >= config['detectors']['visual_threshold']]
                    selected = predictions[0] if predictions else None
                    if label['state'] == 'visible':
                        good = selected is not None and iou(selected['bbox'], label['bbox']) >= .25
                        tp += good; fn += not good; fp += selected is not None and not good
                        possible += any(iou(c['bbox'], label['bbox']) >= .25 for c in predictions)
                    else:
                        fp += selected is not None
                metrics[clip] = {**counts(tp, fp, fn), 'known_frames': known, 'unknown_frames': unknown,
                                 'candidate_pool_reference_matches': possible,
                                 'selection': 'highest_visual_confidence_without_temporal_tracking'}
            elapsed = time.perf_counter()-started
            results['variants'][variant] = {'checkpoint_sha256': detector.checkpoint_sha256,
                       'metrics': metrics, 'wall_seconds': elapsed,
                       'inference_seconds': sum(r['timing']['seconds'] for r in frames),
                       'views': sum(r['timing']['views'] for r in frames)}
            (root/(variant+'.json')).write_text(json.dumps(frames, separators=(',', ':'))+'\n')
            (root/'report.json').write_text(json.dumps(results, indent=2)+'\n')
            print(variant, json.dumps(metrics['all']), 'seconds', round(elapsed, 2), flush=True)
    results['runtime']['max_rss_bytes_macos'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    results['runtime']['mps_allocated_bytes'] = torch.mps.current_allocated_memory() if device == 'mps' else None
    (root/'report.json').write_text(json.dumps(results, indent=2)+'\n')
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    run(parser.parse_args().output)
