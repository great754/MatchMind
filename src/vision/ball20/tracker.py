"""M20 motion-guided verification around the unchanged-default M19 matcher."""
import math
import time
import cv2
import numpy as np
from src.vision.ball.geometry import camera_motion
from src.vision.ball.schema import BallFrame
from src.vision.ball19.tracker import VisualBallTracker, AssociationConfig
from .motion import MotionBank
from .schema import TrajectoryFrame


def appearance(image, box):
    x, y, width, height = box
    patch = image[max(0, int(y)):min(image.shape[0], math.ceil(y+height)),
                  max(0, int(x)):min(image.shape[1], math.ceil(x+width))]
    if not patch.size: return None
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [1, 2], None, [4, 8], [0, 256, 0, 256]).ravel().astype(float)
    return histogram / max(histogram.sum(), 1e-12)


def similarity(a, b):
    return None if a is None or b is None else float(np.clip(np.sqrt(a*b).sum(), 0, 1))


class TrajectoryBallTracker:
    def __init__(self, width, height, config):
        self.width = width; self.height = height; self.cfg = config
        self.diagonal = math.hypot(width, height)
        self.core = VisualBallTracker(width, height, config['detector']['threshold'], AssociationConfig(**config['association']))
        self.banks = {}; self.templates = {}; self.previous_camera = None
        self.plan = None; self.previous_state = 'UNKNOWN'; self.ever_observed = False

    def prepare(self, frame, timestamp, camera, people=()):
        """Predict before detector inference: crops cannot consult current candidates."""
        if self.plan is not None: raise ValueError('Consume the previous prediction before preparing another')
        if not math.isfinite(timestamp) or timestamp < 0 or self.core.previous_time is not None and timestamp <= self.core.previous_time:
            raise ValueError('Source timestamps must increase')
        cut = self.previous_camera is not None and (camera['epoch'] != self.previous_camera['epoch'] or camera.get('continuity') == 'cut')
        if cut:
            self.banks.clear(); self.templates.clear(); self.previous_state = 'UNKNOWN'; self.ever_observed = False
        motion, source = camera_motion(self.previous_camera, camera)
        self.previous_camera = camera
        active = {h['id']: h for h in self.core.hypotheses if timestamp-h['observed_at'] <= self.core.cfg.prediction_seconds}
        self.banks = {key: bank for key, bank in self.banks.items() if key in active}
        predictions = {}
        for key, bank in self.banks.items():
            point = bank.combined()[0][:2]
            contact = any((box := p.get('bbox')) and box[0] <= point[0] <= box[0]+box[2]
                          and box[1]+.5*box[3] <= point[1] <= box[1]+1.2*box[3] for p in people)
            bank.predict(timestamp, motion, source, contact)
            predictions[key] = {**bank.public(), 'track_id': key, 'confirmed': active[key]['confirmed']}
        regions = []
        ordered = sorted(predictions.values(), key=lambda p: (not p['confirmed'], p['age_seconds'], p['track_id']))
        for prediction in ordered:
            if not prediction['confirmed'] or prediction['radius90_px'] > self.diagonal*self.cfg['motion']['maximum_covariance_radius_diagonals']:
                continue
            x, y = prediction['mean']
            if not 0 <= x < self.width or not 0 <= y < self.height: continue
            size = int(np.clip(2*prediction['radius90_px']+32, self.cfg['detector']['minimum_region_px'], self.cfg['detector']['maximum_region_px']))
            width = min(size, self.width); height = min(size, self.height)
            left = max(0, min(self.width-width, round(x-width/2)))
            top = max(0, min(self.height-height, round(y-height/2)))
            box = [left, top, width, height]
            if any(row['bbox'] == box for row in regions): continue
            regions.append({'track_id': prediction['track_id'], 'bbox': box, 'radius90_px': prediction['radius90_px']})
            if len(regions) >= self.cfg['detector']['maximum_regions']: break
        global_needed = cut or not regions or frame % self.cfg['detector']['global_interval_frames'] == 0 or self.previous_state in ['UNKNOWN', 'AMBIGUOUS']
        self.plan = dict(frame=frame, timestamp=timestamp, camera=camera, people=people, predictions=predictions,
                         regions=regions, global_inference=global_needed, cut=cut)
        return regions, global_needed

    def consume(self, image, candidates, *, inference_seconds=0., crop_seconds=0.):
        started = time.perf_counter()
        if self.plan is None: raise ValueError('Prepare a source-frame prediction first')
        plan = self.plan; self.plan = None
        if image.shape[:2] != (self.height, self.width): raise ValueError('Actual source dimensions required')
        feature = {c['candidate_id']: appearance(image, c['bbox']) for c in candidates}
        scores = {}
        priors = {key: (p['mean'], max(4., p['radius90_px']/math.sqrt(4.605170186))) for key, p in plan['predictions'].items()}

        def cost(hypothesis, candidate, default):
            key = hypothesis['id']; bank = self.banks.get(key)
            if bank is None: return default
            innovation = bank.innovation(candidate.center)
            app = similarity(self.templates.get(key), feature[candidate.candidate_id])
            accepted = innovation <= self.cfg['motion']['maximum_mahalanobis_squared'] and (app is None or app >= self.cfg['motion']['minimum_appearance_similarity'])
            rank = .65*candidate.confidence + .20*math.exp(-min(innovation, 1e3)/2) + .15*(app if app is not None else .5)
            scores[(key, candidate.candidate_id)] = {'mahalanobis_squared': innovation, 'appearance_similarity': app,
                                                    'association_score': rank, 'passed': accepted}
            return 1-rank if accepted else 1e6

        ball = self.core.step(plan['frame'], plan['timestamp'], candidates, plan['camera'], plan['people'],
                              association_priors=priors, association_cost=cost)
        retained = {h['id'] for h in self.core.hypotheses}
        self.banks = {key: bank for key, bank in self.banks.items() if key in retained}
        self.templates = {key: value for key, value in self.templates.items() if key in retained}
        current_hypotheses = []
        for h in self.core.hypotheses:
            key = h['id']; candidate = h['candidate']
            if candidate is not None:
                if key not in self.banks:
                    self.banks[key] = MotionBank(candidate.center, plan['timestamp'], self.diagonal, self.cfg['motion'])
                else: self.banks[key].update(candidate.center, plan['timestamp'])
                visual = feature[candidate.candidate_id]
                if visual is not None:
                    old = self.templates.get(key)
                    self.templates[key] = visual if old is None else .8*old+.2*visual
            if key in plan['predictions']: current_hypotheses.append(plan['predictions'][key])
            elif key in self.banks: current_hypotheses.append({**self.banks[key].public(), 'track_id': key, 'confirmed': h['confirmed']})
        selected = next((c['candidate_id'] for c in candidates if ball.directly_observed and tuple(c['bbox']) == ball.image_bbox), None)
        updates = dict(version='m20', motion_model='M20_IMM_IMAGE_MOTION; PHYSICAL_STATE_UNVERIFIED',
                       search_stage='GLOBAL_PLUS_REAL_TRAJECTORY_CROPS' if plan['regions'] else 'GLOBAL_REACQUISITION',
                       coordinate_provenance='M20_IMAGE_EVIDENCE_ONLY; NO_VERIFIED_BALL_HEIGHT',
                       estimated_velocity=None, velocity_provenance='UNKNOWN', prediction_region=None)
        prior = plan['predictions'].get(ball.track_id)
        if prior is not None:
            updates.update(estimated_velocity=prior['velocity'], velocity_provenance='M20_IMM_IMAGE_PX_PER_SECOND; NOT_METRIC')
            if prior['radius90_px'] <= self.diagonal*self.cfg['motion']['maximum_covariance_radius_diagonals']:
                updates['prediction_region'] = {'center': prior['mean'], 'radius90_px': prior['radius90_px'], 'calibration': 'Gaussian mixture moment proxy; not empirically calibrated'}
        if not ball.directly_observed and ball.tracking_status in ['PREDICTED', 'OCCLUDED']:
            if updates['prediction_region'] is not None and prior is not None and 0 <= prior['mean'][0] < self.width and 0 <= prior['mean'][1] < self.height:
                updates.update(predicted_image_position=prior['mean'], prediction_uncertainty_px=prior['radius90_px'], prediction_age_seconds=prior['age_seconds'])
            else:
                updates.update(tracking_status='LOST', predicted_image_position=None, prediction_age_seconds=None, prediction_uncertainty_px=None)
        ball = BallFrame.model_validate({**ball.model_dump(), **updates})
        state = 'OBSERVED' if ball.directly_observed else 'AMBIGUOUS' if ball.tracking_status == 'AMBIGUOUS' else 'PREDICTED' if ball.predicted_image_position is not None else 'UNKNOWN'
        decisions = []
        for candidate in candidates:
            options = [value for (key, name), value in scores.items() if name == candidate['candidate_id']]
            best = max(options, key=lambda row: row['association_score']) if options else {}
            chosen = selected == candidate['candidate_id']
            decision = 'OBSERVED' if chosen else 'REJECTED' if candidate['confidence'] < self.core.threshold or options and not any(o['passed'] for o in options) else 'COMPETING'
            reason = 'ACCEPTED_CURRENT_VISUAL_MEASUREMENT' if chosen else 'BELOW_FROZEN_VISUAL_THRESHOLD' if candidate['confidence'] < self.core.threshold else 'INNOVATION_OR_APPEARANCE_GATE' if decision == 'REJECTED' else 'WAIT_VISUAL_BIRTH_OR_RESOLVE_COMPETITION'
            decisions.append({'candidate_id': candidate['candidate_id'], 'decision': decision, 'reason': reason,
                              'visual_confidence': candidate['confidence'], **{k: best.get(k) for k in ['mahalanobis_squared', 'appearance_similarity', 'association_score']}})
        events = ['CAMERA_EPOCH_RESET'] if plan['cut'] else []
        if state == 'OBSERVED' and self.previous_state != 'OBSERVED' and self.ever_observed: events.append('REACQUIRED_WITH_FRESH_VISUAL_EVIDENCE')
        if state == 'UNKNOWN' and self.previous_state in ['OBSERVED', 'PREDICTED']: events.append('TRACK_LOST')
        if state == 'AMBIGUOUS': events.append('COMPETING_VISUAL_HYPOTHESES')
        self.ever_observed |= state == 'OBSERVED'; self.previous_state = state
        debug = TrajectoryFrame(frame=ball.frame, timestamp=ball.timestamp, camera_epoch=ball.camera_epoch, state=state,
                                selected_candidate_id=selected, hypotheses=current_hypotheses, search_regions=plan['regions'],
                                candidates=decisions, global_inference=plan['global_inference'], inference_seconds=inference_seconds,
                                crop_inference_seconds=crop_seconds, tracker_seconds=time.perf_counter()-started, events=events,
                                prediction_only=state == 'PREDICTED', tracking_confidence=ball.tracking_confidence)
        return ball, debug
