"""Observed-source ball rows and separate offline reconstruction provenance."""
import math
import json
from pathlib import Path
import cv2
from src.vision.ball.schema import BallFrame
from src.vision.ball19.tracker import VisualBallTracker, AssociationConfig
from src.vision.advanced21.camera import causal_camera
from src.vision.advanced21.offline import point

CANDIDATE_FIELDS = ['candidate_id', 'bbox', 'confidence', 'inference_view', 'source']


def clean(candidate):
    return {k: candidate[k] for k in CANDIDATE_FIELDS if k in candidate}


def causal_rows(candidate_sets, worlds, inputs, metadata, threshold=.35):
    tracker = VisualBallTracker(metadata['width'], metadata['height'], threshold, AssociationConfig())
    result = []
    for i, candidates in enumerate(candidate_sets):
        camera = causal_camera(worlds[i]['camera'], inputs[i]['camera_refinement'])
        people = [{'bbox': p['bbox']} for p in inputs[i]['player_observations']]
        ball = tracker.step(i, i/metadata['fps'], [clean(c) for c in candidates], camera, people)
        result.append(BallFrame.model_validate({**ball.model_dump(), 'version': 'm21',
                      'coordinate_provenance': 'M21_CURRENT_PIXEL_EVIDENCE; NO_VERIFIED_HEIGHT',
                      'motion_model': 'CAUSAL_IMAGE_ASSOCIATION', 'search_stage': 'M21_CURRENT_SOURCE_PIXELS'}))
    return result


def imm_rows(candidate_sets, worlds, inputs, metadata):
    """Matched M20 IMM association on alternative actual current-pixel candidate pools.

    M8's precomputed camera context is retained, so this experiment is offline;
    the separate causal_rows implementation never reads that absolute context.
    """
    from src.vision.ball20.tracker import TrajectoryBallTracker
    config = json.loads(Path('configs/vision/cv_m20.json').read_text())
    tracker = TrajectoryBallTracker(metadata['width'],metadata['height'],config)
    capture = cv2.VideoCapture(metadata['source_video'])
    result = []
    try:
        for i,candidates in enumerate(candidate_sets):
            ok,image = capture.read()
            if not ok:raise ValueError('Verified source frame unavailable for IMM replay')
            people = [{'bbox':p['bbox']} for p in inputs[i]['player_observations']]
            tracker.prepare(i,i/metadata['fps'],worlds[i]['camera'],people)
            state,_ = tracker.consume(image,[clean(c) for c in candidates])
            result.append(BallFrame.model_validate({**state.model_dump(),'version':'m21'}))
    finally:capture.release()
    return result


def offline_rows(selected, inferred, candidate_sets, worlds, metadata):
    result = []
    last_frame = last_track = last_point = None
    track = 0
    previous_epoch = None
    for i, (chosen, reconstructed, candidates) in enumerate(zip(selected, inferred, candidate_sets, strict=True)):
        epoch = worlds[i]['camera']['epoch']
        if epoch != previous_epoch or worlds[i]['camera'].get('continuity') == 'cut':
            last_frame = last_track = last_point = None
        previous_epoch = epoch
        shared = {'version': 'm21', 'frame': i, 'timestamp': i/metadata['fps'], 'camera_epoch': epoch,
                  'candidates': [clean(c) for c in candidates], 'candidate_count': len(candidates),
                  'motion_model': 'OFFLINE_GLOBAL_IMAGE_PATH; USES_FUTURE',
                  'coordinate_provenance': 'M21_IMAGE_EVIDENCE_ONLY; METRIC_HEIGHT_UNAVAILABLE',
                  'coordinate_state': 'HEIGHT_UNKNOWN', 'search_stage': 'M21_OFFLINE_GLOBAL_LINKING'}
        if chosen is not None:
            gap = (i-last_frame)/metadata['fps'] if last_frame is not None else math.inf
            if gap > .35:
                track += 1
            xy = point(chosen).tolist()
            shared.update(directly_observed=True, detection_status='DETECTED',
                tracking_status='TRACKED' if last_track == track else 'DETECTED', image_position=xy,
                image_bbox=chosen['bbox'], detection_confidence=chosen['confidence'],
                tracking_confidence=chosen['confidence'], track_id=track,
                last_confirmed_image=xy, last_confirmed_timestamp=i/metadata['fps'],
                reason='OFFLINE_SELECTED_ACTUAL_CURRENT_VISUAL_CANDIDATE')
            last_frame, last_track, last_point = i, track, xy
        elif reconstructed is not None and last_frame is not None:
            shared.update(tracking_status='PREDICTED', predicted_image_position=reconstructed['image'],
                prediction_age_seconds=(i-last_frame)/metadata['fps'], track_id=last_track,
                last_confirmed_image=last_point, last_confirmed_timestamp=last_frame/metadata['fps'],
                reason='OFFLINE_FUTURE_SUPPORTED_RECONSTRUCTION_NOT_OBSERVED')
        else:
            shared.update(reason='NO_CURRENT_BALL_EVIDENCE', last_confirmed_image=last_point,
                          last_confirmed_timestamp=last_frame/metadata['fps'] if last_frame is not None else None)
        result.append(BallFrame.model_validate(shared))
    return result
