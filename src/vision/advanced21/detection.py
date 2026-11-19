"""Small-ball search and past-frame fusion with current-pixel re-verification."""
from dataclasses import dataclass
import hashlib
import time
import cv2
import numpy as np


def tile_regions(width, height, size=256, overlap=.2):
    if min(width, height, size) <= 0 or not 0 <= overlap < 1:
        raise ValueError('Positive dimensions and bounded overlap required')
    def axis(length):
        extent = min(size, length)
        return sorted(set([*range(0, max(1, length-extent+1), max(1, round(size*(1-overlap)))), length-extent]))
    return [(x, y, min(size, width), min(size, height)) for y in axis(height) for x in axis(width)]


def bounded_region(center, width, height, size=192):
    w, h = min(size, width), min(size, height)
    return (max(0, min(width-w, round(center[0]-w/2))),
            max(0, min(height-h, round(center[1]-h/2))), w, h)


def merge_candidates(*groups, iou_threshold=.3, limit=80):
    """Keep actual winning boxes/confidences; merging never raises a visual score."""
    from src.evaluation.cv_m19 import iou
    candidates = sorted([dict(c) for group in groups for c in group],
                        key=lambda c: (-c['confidence'], tuple(c['bbox']), c.get('inference_view', '')))
    retained = []
    for row in candidates:
        if not any(iou(row['bbox'], c['bbox']) > iou_threshold for c in retained):
            row['candidate_id'] = f"c{len(retained)}"
            retained.append(row)
        if len(retained) >= limit:
            break
    return retained


@dataclass
class DetectorBackend:
    model: object
    checkpoint_sha256: str
    ball_class: int = 0
    resolution: int = 640
    device: str = 'cpu'
    batch_size: int = 8

    def infer(self, image, regions, view='current_dense'):
        """Ultralytics outputs are already in crop pixels, before source translation."""
        start = time.perf_counter()
        h, w = image.shape[:2]
        candidates = []
        regions = list(dict.fromkeys(tuple(map(int, r)) for r in regions))
        for x, y, cw, ch in regions:
            if min(x, y) < 0 or min(cw, ch) <= 0 or x+cw > w or y+ch > h:
                raise ValueError('Detector crop outside source')
        for offset in range(0, len(regions), self.batch_size):
            batch = regions[offset:offset+self.batch_size]
            outputs = self.model.predict([image[y:y+ch, x:x+cw] for x, y, cw, ch in batch],
                       imgsz=self.resolution, conf=.02, iou=.4, classes=[self.ball_class],
                       verbose=False, device=self.device, max_det=40)
            for region, output in zip(batch, outputs, strict=True):
                for raw, confidence in zip(output.boxes.xyxy.cpu().numpy(), output.boxes.conf.cpu().numpy(), strict=True):
                    a, b, c, d = raw + np.array([region[0], region[1], region[0], region[1]])
                    a, b, c, d = max(0., float(a)), max(0., float(b)), min(float(w), float(c)), min(float(h), float(d))
                    if min(c-a, d-b) < 1 or max(c-a, d-b) > .15*max(w, h):
                        continue
                    candidates.append({'candidate_id': str(len(candidates)), 'bbox': [a, b, c-a, d-b],
                                       'confidence': float(confidence), 'inference_view': view,
                                       'source': 'VIDEO_CV_DETECTOR'})
        return merge_candidates(candidates), {'seconds': time.perf_counter()-start, 'views': len(regions),
                    'source_pixels_sha256': hashlib.sha256(image.tobytes()).hexdigest(),
                    'checkpoint_sha256': self.checkpoint_sha256}


def past_frame_fusion(current, history, motion_to_current):
    """Causal search image, not direct evidence. A separate current-image detector must confirm."""
    if len(history) != len(motion_to_current):
        raise ValueError('One camera transform per history image required')
    h, w = current.shape[:2]
    total = current.astype(np.float32)*.6
    weight = np.full((h, w, 1), .6, np.float32)
    used = 0
    for old, transform, value in zip(history[-2:][::-1], motion_to_current[-2:][::-1], [.25, .15]):
        if transform is None:
            continue
        transform = np.asarray(transform, float)
        if transform.shape != (3, 3) or not np.isfinite(transform).all():
            continue
        warped = cv2.warpPerspective(old, transform, (w, h))
        mask = cv2.warpPerspective(np.ones(old.shape[:2], np.uint8), transform, (w, h), flags=cv2.INTER_NEAREST)
        total += warped.astype(np.float32)*mask[..., None]*value
        weight += mask[..., None]*value
        used += 1
    return np.clip(total/weight, 0, 255).astype(np.uint8), used
