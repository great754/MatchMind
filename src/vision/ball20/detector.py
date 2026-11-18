"""Fresh YOLO crop inference with explicit source-coordinate mapping.

The existing M16 checkpoint/global path is reused. No weights are downloaded.
"""
import time
import numpy as np
from src.vision.ball.schema import BallCandidate


def source_candidate(box, confidence, region, frame_size, key, resolution):
    """Ultralytics boxes already undo its internal letterbox/resize transform."""
    x, y, width, height = region
    left, top, right, bottom = map(float, box)
    left = max(0., min(width, left)); right = max(0., min(width, right))
    top = max(0., min(height, top)); bottom = max(0., min(height, bottom))
    full_width, full_height = frame_size
    if right-left < 1 or bottom-top < 1 or max(right-left, bottom-top) > .15*max(full_width, full_height):
        return None
    return BallCandidate(candidate_id=key, bbox=(x+left, y+top, right-left, bottom-top),
                         confidence=float(confidence), inference_view=f'M20_CROP_{resolution}@{x},{y},{width},{height}').model_dump()


class YoloCropAdapter:
    def __init__(self, model, resolution=640, device='cpu'):
        if resolution < 320 or resolution > 1280 or resolution % 32:
            raise ValueError('Use a documented 32-aligned crop input resolution')
        self.model = model; self.resolution = resolution; self.device = device

    def infer(self, image, regions):
        started = time.perf_counter()
        if not regions:
            return [], {'seconds': 0., 'views': 0}
        height, width = image.shape[:2]
        crops = []
        for row in regions:
            x, y, w, h = row['bbox']
            if x < 0 or y < 0 or w <= 0 or h <= 0 or x+w > width or y+h > height:
                raise ValueError('Crop must be inside the actual source frame')
            crops.append(np.ascontiguousarray(image[y:y+h, x:x+w]))
        # Real detector invocation on current image crops, never cached filtering.
        outputs = self.model.predict(crops, imgsz=self.resolution, conf=.02, iou=.4,
                                     verbose=False, device=self.device, max_det=40)
        result = []
        for i, (row, out) in enumerate(zip(regions, outputs, strict=True)):
            for j, (box, confidence) in enumerate(zip(out.boxes.xyxy.cpu().numpy(), out.boxes.conf.cpu().numpy(), strict=True)):
                candidate = source_candidate(box, confidence, row['bbox'], (width, height), f'local_{i}_{j}', self.resolution)
                if candidate: result.append(candidate)
        return result, {'seconds': time.perf_counter()-started, 'views': len(crops)}


def merge_candidates(global_candidates, local_candidates):
    """NMS retains original neural confidence; no trajectory-based score inflation."""
    from ml.ball14.evaluate import iou
    ordered = sorted([dict(c) for c in global_candidates+local_candidates],
                     key=lambda c: (-c['confidence'], c['inference_view'], c['candidate_id']))
    kept = []
    for candidate in ordered:
        if not any(iou(candidate['bbox'], row['bbox']) > .3 for row in kept):
            candidate['candidate_id'] = f'm20_c{len(kept)}'; kept.append(candidate)
    return kept[:160]
