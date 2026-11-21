"""Private source-image error sheets; labels are used only after inference."""
import argparse
from collections import Counter
import json
from pathlib import Path
import cv2
import numpy as np
from src.evaluation.cv_m19 import iou
from src.vision.intelligent23.inputs import Inputs, NAMES, write
from src.vision.intelligent23.evaluation import references, team_goal_failure


def run(input_root, perception, analysis, output):
    inputs = Inputs(input_root)
    dataset, original, new_tasks, labels = references(inputs)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    report = {}
    for name in NAMES:
        meta = inputs.json(Path('outputs/world_model')/(name+'_m9_ready')/'manifest.json')
        records = [json.loads(l) for l in (Path(perception)/name/'inputs.jsonl').open()]
        full = [json.loads(l) for l in (Path(analysis)/name/'full_m23/ball_frames.jsonl').open()]
        guided = [json.loads(l) for l in (Path(analysis)/name/'guided_search/ball_frames.jsonl').open()]
        debug = [json.loads(l) for l in inputs.path(Path('outputs/m22/analysis_review_v1')/name/'football_frames.jsonl').open()]
        stages = team_goal_failure(new_tasks, labels, name, full, records, 'full_m23', debug)
        report[name] = {'failure_stages': dict(Counter(r['failure_stage'] for r in stages)), 'details': stages, 'lost_after_verification': []}
        tasks = {t['task_id']: t for t in new_tasks if t['recording_id'] == name}
        examples = []
        for label in labels:
            task = tasks.get(label['task_id'])
            if task is None:
                continue
            f = task['frame']
            if label['state'] != 'visible' and not full[f]['directly_observed']:
                continue
            match = label['state'] == 'visible' and full[f]['directly_observed'] and iou(full[f]['image_bbox'], label['bbox']) >= .25
            previously = label['state'] == 'visible' and guided[f]['directly_observed'] and iou(guided[f]['image_bbox'], label['bbox']) >= .25
            if previously and not match:
                report[name]['lost_after_verification'].append({'frame': f, 'guided_bbox': guided[f]['image_bbox'],
                    'matching_candidates': [{k: c[k] for k in ['bbox', 'confidence', 'origin', 'observation_eligible', 'decision_reason', 'joint_memory_evidence', 'support_views']}
                        for c in records[f]['methods']['full_m23']['candidates'] if iou(c['bbox'], label['bbox']) >= .25]})
            if not match:
                examples.append((f, label, 'FALSE' if full[f]['directly_observed'] else 'MISS'))
        examples = sorted(examples, key=lambda r: (r[2] != 'FALSE', r[0]))[:12]
        capture = cv2.VideoCapture(str(inputs.path(meta['source_video'])))
        tiles = []
        try:
            for f, label, kind in examples:
                capture.set(cv2.CAP_PROP_POS_FRAMES, f)
                ok, image = capture.read()
                if not ok:raise ValueError('Missing diagnostic source frame')
                source = image.copy()
                for box, color in [(label['bbox'], (50, 240, 70)), (full[f]['image_bbox'], (60, 60, 255))]:
                    if box is not None:
                        x, y, w, h = map(round, box)
                        cv2.rectangle(source, (x-2, y-2), (x+w+2, y+h+2), color, 2)
                thumbnail = cv2.resize(source, (480, 270))
                box = label['bbox'] or full[f]['image_bbox']
                x, y, w, h = box
                patch = cv2.getRectSubPix(source, (80, 80), (float(x+w/2), float(y+h/2)))
                patch = cv2.resize(patch, (160, 160))
                tile = np.full((325, 640, 3), 20, np.uint8)
                tile[35:305, :480] = thumbnail
                tile[60:220, 480:] = patch
                cv2.putText(tile, f'{name} frame {f} / {kind}', (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .6, (240, 240, 240), 1)
                cv2.putText(tile, 'Green: human / red: model', (8, 321), cv2.FONT_HERSHEY_SIMPLEX, .5, (240, 240, 240), 1)
                tiles.append(tile)
        finally:
            capture.release()
        if tiles:
            while len(tiles) % 2:tiles.append(np.zeros_like(tiles[0]))
            sheet = np.vstack([np.hstack(tiles[i:i+2]) for i in range(0, len(tiles), 2)])
            if not cv2.imwrite(str(root/(name+'.jpg')), sheet):raise ValueError('Diagnostic write failed')
    write(root/'report.json', report)
    print({n: {'failure_stages': v['failure_stages'], 'lost_after_verification': v['lost_after_verification']} for n, v in report.items()}, flush=True)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', required=True)
    p.add_argument('--perception', required=True)
    p.add_argument('--analysis', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    run(a.inputs, a.perception, a.analysis, a.output)
