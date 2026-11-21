"""Bounded football search memory extends the existing M20 IMM and M19 matcher."""
import math
import copy
import numpy as np
from src.evaluation.cv_m19 import iou
from src.vision.ball20.tracker import TrajectoryBallTracker, appearance, similarity
from src.vision.ball.geometry import camera_motion
from src.vision.advanced21.detection import bounded_region, tile_regions


class BallMemory:
    def __init__(self, width, height, m20_config, config):
        self.cfg = config
        self.core = TrajectoryBallTracker(width, height, copy.deepcopy(m20_config))
        self.width, self.height = width, height
        self.diagonal = math.hypot(width, height)
        self.archive = {}
        self.previous_epoch = None
        self.plan = None
        self.last_observation = None
        self.last_state = 'UNKNOWN'
        self.discovery_grid = tile_regions(width, height, config['memory']['discovery_region_px'], .2)

    def prepare(self, frame, timestamp, camera, people):
        if self.plan is not None:
            raise ValueError('Consume the preceding source frame first')
        cut = self.previous_epoch is not None and (camera['epoch'] != self.previous_epoch or camera.get('continuity') == 'cut')
        field = camera.get('motion_evidence', {}).get('field_motion', {})
        incompatible = False
        if field.get('accepted') and field.get('homography') is not None:
            h = np.asarray(field['homography'], float)
            incompatible = h.shape != (3, 3) or not np.isfinite(h).all() or abs(np.linalg.det(h)) < 1e-8
            if not incompatible:
                singular = np.linalg.svd(h[:2, :2], compute_uv=False)
                incompatible = singular.min() < .2 or singular.max() > 5
        if incompatible:
            self.core = TrajectoryBallTracker(self.width, self.height, copy.deepcopy(self.core.cfg))
            self.archive.clear()
            self.last_observation = None
            self.last_state = 'UNKNOWN'
            camera = {**camera, 'motion_evidence': {'field_motion': {}}}
            cut = True
        if cut:
            self.archive.clear()
            self.last_observation = None
            self.last_state = 'UNKNOWN'
        self.previous_epoch = camera['epoch']
        previous = {h['id']: h for h in self.core.core.hypotheses}
        for key, bank in ([] if cut else self.core.banks.items()):
            h = previous.get(key)
            if h and h['confirmed']:
                self.archive[key] = {'bank': bank, 'template': self.core.templates.get(key),
                    'bbox': list(h['candidate'].bbox) if h['candidate'] is not None else self.archive.get(key, {}).get('bbox'),
                    'frame': h['last_frame'], 'stationary_frames': self.archive.get(key, {}).get('stationary_frames', 0)}
        self.archive = {key: v for key, v in self.archive.items() if timestamp-v['bank'].last_observed_at <= self.cfg['memory']['maximum_age_seconds']}
        self.core.prepare(frame, timestamp, camera, people)
        motion, source = camera_motion(
            self._old_camera if hasattr(self, '_old_camera') else None, camera)
        self._old_camera = camera
        predictions = dict(self.core.plan['predictions'])
        for key, value in self.archive.items():
            bank = value['bank']
            if key not in predictions:
                if bank.time < timestamp:
                    bank.predict(timestamp, motion, source)
                predictions[key] = {**bank.public(), 'track_id': key, 'confirmed': True, 'archived': True}
        regions = []
        maximum = self.cfg['memory']['maximum_regions']
        tentative = 0
        for key, prediction in sorted(predictions.items(), key=lambda p: (not p[1]['confirmed'], p[1]['age_seconds'], p[0])):
            if not prediction['confirmed']:
                if tentative >= self.cfg['memory']['maximum_tentative_regions'] or prediction['age_seconds'] > self.cfg['memory']['tentative_age_seconds']:
                    continue
                tentative += 1
            if prediction['radius90_px'] > self.diagonal*self.cfg['memory']['maximum_radius_diagonals']:
                continue
            point = prediction['mean']
            box = self.archive.get(key, {}).get('bbox')
            ball_size = max(box[2:]) if box else 6.
            size = int(np.clip(2*prediction['radius90_px']+self.cfg['memory']['measurement_size_margin']*ball_size,
                self.cfg['memory']['minimum_region_px'], self.cfg['memory']['maximum_region_px']))
            w, h = min(size, self.width), min(size, self.height)
            # A just-offscreen prediction may still overlap the visible image.
            if point[0]+size/2 < 0 or point[0]-size/2 >= self.width or point[1]+size/2 < 0 or point[1]-size/2 >= self.height:
                continue
            x, y = max(0, min(self.width-w, round(point[0]-w/2))), max(0, min(self.height-h, round(point[1]-h/2)))
            region = [x, y, w, h]
            if any(iou(region, r['bbox']) >= self.cfg['memory']['duplicate_region_iou'] for r in regions):
                continue
            regions.append({'bbox': region, 'track_id': key, 'radius90_px': prediction['radius90_px'], 'planned_before_current_candidates': True,
                            'kind': 'trajectory' if prediction['confirmed'] else 'tentative_visual'})
            bank = self.core.banks.get(key) or self.archive.get(key, {}).get('bank')
            if prediction['confirmed'] and bank is not None:
                modes = sorted(range(len(bank.mu)), key=lambda j: -bank.mu[j])[:self.cfg['memory']['maximum_motion_modes_per_track']]
                for j in modes:
                    if bank.mu[j] < self.cfg['memory']['minimum_motion_mode_probability'] or np.linalg.norm(bank.x[j, :2]-point) < 8:
                        continue
                    radius = math.sqrt(4.605170186*np.linalg.eigvalsh(bank.p[j, :2, :2]).max())
                    extent = int(np.clip(2*radius+6*ball_size, self.cfg['memory']['minimum_region_px'], self.cfg['memory']['maximum_region_px']))
                    extra = list(bounded_region(bank.x[j, :2], self.width, self.height, extent))
                    if len(regions) < maximum and not any(iou(extra, r['bbox']) >= self.cfg['memory']['duplicate_region_iou'] for r in regions):
                        regions.append({'bbox': extra, 'track_id': key, 'radius90_px': radius,
                            'planned_before_current_candidates': True, 'kind': 'motion_mode',
                            'motion_mode': bank.names[j], 'mode_probability': float(bank.mu[j])})
            if len(regions) >= maximum:
                break
        # Independent discovery remains bounded and does not inherit the old ball's location.
        # Current M9 person boxes propose areas to inspect; they never establish a touch.
        if self.last_state != 'OBSERVED' or frame % 5 == 0:
            proposals = []
            if people:
                person = people[frame % len(people)]
                x, y, w, h = person['bbox']
                proposals.append(bounded_region((x+w/2, y+h), self.width, self.height, self.cfg['memory']['discovery_region_px']))
            proposals.append(self.discovery_grid[frame % len(self.discovery_grid)])
            for region in proposals[:self.cfg['memory']['discovery_regions_per_frame']]:
                if not any(iou(region, r['bbox']) >= self.cfg['memory']['duplicate_region_iou'] for r in regions):
                    regions.append({'bbox': list(region), 'track_id': -1, 'radius90_px': 0.,
                        'planned_before_current_candidates': True, 'kind': 'global_discovery'})
        self.plan = {'frame': frame, 'timestamp': timestamp, 'camera': camera, 'people': people,
                     'predictions': predictions, 'regions': regions, 'camera_motion_source': source,
                     'camera_homography': motion.tolist() if motion is not None else None, 'cut': cut}
        return copy.deepcopy(self.plan)

    def evidence(self, image, candidate):
        if self.plan is None:
            raise ValueError('Predict before inspecting current candidates')
        box = candidate['bbox']
        point = np.array([box[0]+box[2]/2, box[1]+box[3]/2])
        visual = appearance(image, box)
        options = []
        for key, prediction in self.plan['predictions'].items():
            value = self.archive.get(key, {})
            bank = self.core.banks.get(key) or value.get('bank')
            if bank is None or (not prediction['confirmed'] and prediction['age_seconds'] > self.cfg['memory']['tentative_age_seconds']):
                continue
            distance = bank.innovation(point)
            model_distances = []
            for weight, mean, covariance in zip(bank.mu, bank.x, bank.p):
                if weight < self.cfg['memory']['minimum_motion_mode_probability']:
                    continue
                delta = point-mean[:2]
                model_distances.append(float(delta@np.linalg.solve(covariance[:2, :2]+np.eye(2)*bank.cfg['measurement_sigma_px']**2, delta)))
            distance = min([distance, *model_distances])
            template = self.core.templates.get(key)
            if template is None:
                template = value.get('template')
            app = similarity(template, visual)
            previous_box = value.get('bbox')
            size_ratio = math.sqrt(box[2]*box[3]/max(previous_box[2]*previous_box[3], 1.)) if previous_box else 1.
            options.append({'track_id': key, 'mahalanobis_squared': distance, 'appearance_similarity': app,
                'size_ratio': size_ratio, 'velocity': prediction['velocity'],
                'joint_support': math.exp(-min(distance, 100)/2)*(app if app is not None else .5),
                'stationary_frames': value.get('stationary_frames', 0), 'confirmed_history': prediction['confirmed']})
        return max(options, key=lambda v: v['joint_support']) if options else None

    def consume(self, image, candidates):
        if self.plan is None:
            raise ValueError('Prepare a source prediction first')
        plan = self.plan
        # Reuse the unchanged M19 repeated-birth/one-to-one gate and M20 IMM.
        clean = [{k: c[k] for k in ['candidate_id', 'bbox', 'confidence', 'inference_view', 'source'] if k in c} for c in candidates]
        ball, debug = self.core.consume(image, clean)
        for h in self.core.core.hypotheses:
            if h['candidate'] is None or not h['confirmed']:
                continue
            c = next(c for c in candidates if c['candidate_id'] == h['candidate'].candidate_id)
            features = c.get('temporal_features', {})
            static = features.get('camera_residual_px') is not None and features['camera_residual_px'] < 1
            previous = self.archive.get(h['id'], {})
            self.archive[h['id']] = {'bank': self.core.banks[h['id']], 'template': self.core.templates.get(h['id']),
                'bbox': list(h['candidate'].bbox), 'frame': plan['frame'],
                'stationary_frames': previous.get('stationary_frames', 0)+1 if static else 0}
        state = 'OBSERVED' if ball.directly_observed else 'AMBIGUOUS' if ball.tracking_status == 'AMBIGUOUS' else 'PREDICTED' if ball.predicted_image_position is not None else 'UNKNOWN'
        recovered = bool(ball.directly_observed and self.last_observation is not None and self.last_state != 'OBSERVED')
        if ball.directly_observed:
            self.last_observation = {'position': list(ball.image_position), 'frame': ball.frame, 'timestamp': ball.timestamp,
                                     'bbox': list(ball.image_bbox), 'trajectory_hypothesis': ball.track_id}
        snapshot = {'state': state, 'last_direct_observation': copy.deepcopy(self.last_observation),
            'missed_frames': plan['frame']-self.last_observation['frame'] if self.last_observation else None,
            'missed_seconds': plan['timestamp']-self.last_observation['timestamp'] if self.last_observation else None,
            'hypotheses': [{**p, 'last_observed_frame': self.archive.get(key, {}).get('frame'),
                'appearance_template': self.archive[key]['template'].tolist() if self.archive.get(key, {}).get('template') is not None else None}
                for key, p in plan['predictions'].items()], 'search_regions': plan['regions'],
            'active_trajectory_hypothesis': ball.track_id, 'camera_motion_source': plan['camera_motion_source'],
            'camera_homography': plan['camera_homography'],
            'camera_cut': plan['cut'], 'reacquired_with_current_visual_evidence': recovered,
            'predicted_image_position': list(ball.predicted_image_position) if ball.predicted_image_position is not None else None,
            'reason': ball.reason, 'appearance_is_calibrated': False,
            'search_memory_is_ball_identity_truth': False}
        self.last_state = state
        self.plan = None
        return ball, snapshot
