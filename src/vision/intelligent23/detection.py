"""The existing M21 visual primary and bounded, fresh current-image ROI calls."""
import hashlib
import time
import cv2
import numpy as np
from src.serving.cv import digest
from src.vision.advanced21.camera import refine_motion, causal_camera
from src.vision.advanced21.detection import DetectorBackend, tile_regions, past_frame_fusion, bounded_region, merge_candidates
from src.vision.advanced21.temporal import batch_visual_features, deterministic_verdict
from src.vision.ball.geometry import camera_motion
from src.vision.ball20.detector import source_candidate


class TemporalPrimary:
    def __init__(self, model, checkpoint_sha256, width, height, device):
        self.detector = DetectorBackend(model, checkpoint_sha256, 32, device=device)
        self.regions = tile_regions(width, height, 320, .2)
        self.width, self.height = width, height
        self.history, self.history_motion = [], []
        self.previous = self.previous_camera = None
        self.previous_people = []

    def camera(self, image, world_camera):
        cut = self.previous_camera is not None and (world_camera['epoch'] != self.previous_camera['epoch'] or world_camera.get('continuity') == 'cut')
        if cut:
            self.previous = None
            self.history, self.history_motion = [], []
        refinement = refine_motion(self.previous, image, self.previous_camera, world_camera, self.previous_people)
        step, source = camera_motion(causal_camera(self.previous_camera) if self.previous_camera else None, causal_camera(world_camera, refinement))
        return refinement, step, source

    def detect(self, image, camera, people, m20_candidates, refinement, step):
        started = time.perf_counter()
        self.history_motion = [step@old if old is not None and step is not None else None for old in self.history_motion]
        fused, history_used = past_frame_fusion(image, self.history, self.history_motion)
        dense, dt = self.detector.infer(image, self.regions, 'current_generic_dense320')
        proposed, ft = (self.detector.infer(fused, [(0, 0, self.width, self.height)], 'PAST_FRAME_FUSION_PROPOSAL_NOT_OBSERVED')
                        if history_used else ([], {'seconds': 0., 'views': 0}))
        regions = list(dict.fromkeys(bounded_region((c['bbox'][0]+c['bbox'][2]/2, c['bbox'][1]+c['bbox'][3]/2),
            self.width, self.height, 192) for c in proposed if c['confidence'] >= .35))[:4]
        confirmed, vt = self.detector.infer(image, regions, 'current_pixels_after_temporal_proposal')
        candidates = merge_candidates(m20_candidates, dense, confirmed)
        features = batch_visual_features(self.previous, image, candidates, step)
        for candidate, values in zip(candidates, features, strict=True):
            candidate.update(temporal_features=values, deterministic=deterministic_verdict(values, candidate['confidence']),
                             origin='primary', visual_evidence='CURRENT_PIXELS')
        self.history = (self.history+[image.copy()])[-2:]
        self.history_motion = (self.history_motion+[np.eye(3)])[-2:]
        self.previous, self.previous_camera, self.previous_people = image, camera, people
        return candidates, {'seconds': time.perf_counter()-started, 'neural_seconds': dt['seconds']+ft['seconds']+vt['seconds'],
            'views': dt['views']+ft['views']+vt['views'], 'fusion_history_frames': history_used,
            'camera_refinement': refinement, 'source_pixels_sha256': hashlib.sha256(image.tobytes()).hexdigest()}


def region_candidates(detector, image, regions):
    """Keep each requested crop's output separate for controlled ablations."""
    started = time.perf_counter()
    regions = list(dict.fromkeys(tuple(map(int, r)) for r in regions))
    width, height = image.shape[1], image.shape[0]
    for x, y, w, h in regions:
        if min(x, y) < 0 or min(w, h) <= 0 or x+w > width or y+h > height:
            raise ValueError('Guided crop outside original source pixels')
    result = {r: [] for r in regions}
    for offset in range(0, len(regions), detector.batch_size):
        batch = regions[offset:offset+detector.batch_size]
        images = [np.ascontiguousarray(image[y:y+h, x:x+w]) for x, y, w, h in batch]
        outputs = detector.model.predict(images, imgsz=detector.resolution, conf=.02, iou=.4,
            classes=[detector.ball_class], max_det=40, verbose=False, device=detector.device)
        for r, output in zip(batch, outputs, strict=True):
            for j, (box, confidence) in enumerate(zip(output.boxes.xyxy.cpu().numpy(), output.boxes.conf.cpu().numpy(), strict=True)):
                value = source_candidate(box, confidence, r, (width, height), f'roi_{r}_{j}', detector.resolution)
                if value:
                    value.update(origin='guided', source_region=list(r), visual_evidence='CURRENT_PIXELS',
                                 source_model_sha256=detector.checkpoint_sha256)
                    result[r].append(value)
    return result, {'seconds': time.perf_counter()-started, 'views': len(regions)}
