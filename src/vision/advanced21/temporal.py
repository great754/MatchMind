"""Consecutive visual evidence: deterministic checks versus a frozen learned verifier."""
import math
import cv2
import numpy as np


def center(box):
    x, y, w, h = box
    return np.array([x+w/2, y+h/2], dtype=float)


def patch(image, point, radius=12, size=48):
    return cv2.getRectSubPix(image, (2*radius, 2*radius), tuple(map(float, point))) if size is None else cv2.resize(
        cv2.getRectSubPix(image, (2*radius, 2*radius), tuple(map(float, point))), (size, size))


def visual_features(previous, current, candidate, motion=None):
    point = center(candidate['bbox'])
    box = candidate['bbox']
    radius = max(5, min(24, math.ceil(max(box[2:]))))
    region = patch(current, point, radius)
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY).astype(float)/255
    dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    tensor = np.array([[np.mean(dx*dx), np.mean(dx*dy)], [np.mean(dx*dy), np.mean(dy*dy)]])
    eigen = np.linalg.eigvalsh(tensor)
    coherence = float((eigen[1]-eigen[0])/max(eigen.sum(), 1e-9))
    features = {'aspect_ratio': float(max(box[2:])/max(min(box[2:]), 1e-6)),
                'edge_coherence': coherence, 'contrast': float(gray.std()),
                'forward_backward_error_px': None, 'camera_residual_px': None,
                'patch_correlation': None, 'history_available': False}
    previous_point = None
    if previous is not None:
        pgray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
        cgray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
        p = np.float32(point).reshape(1, 1, 2)
        old, status, _ = cv2.calcOpticalFlowPyrLK(cgray, pgray, p, None, winSize=(19, 19), maxLevel=2)
        if status is not None and status[0, 0] and old is not None and np.isfinite(old).all():
            back, checked, _ = cv2.calcOpticalFlowPyrLK(pgray, cgray, old, None, winSize=(19, 19), maxLevel=2)
            q = old.reshape(2)
            if checked is not None and checked[0, 0] and back is not None and np.isfinite(back).all() and 0 <= q[0] < previous.shape[1] and 0 <= q[1] < previous.shape[0]:
                features.update(history_available=True, forward_backward_error_px=float(np.linalg.norm(back.reshape(2)-point)))
                previous_point = q
                a = cv2.cvtColor(patch(previous, q, radius), cv2.COLOR_BGR2GRAY).astype(float).ravel()
                b = gray.ravel()
                if a.std() > 1e-6 and b.std() > 1e-6:
                    features['patch_correlation'] = float(np.clip(np.corrcoef(a, b)[0, 1], -1, 1))
                if motion is not None:
                    x = np.asarray(motion, float) @ np.r_[q, 1.]
                    if np.isfinite(x).all() and abs(x[2]) > 1e-9:
                        features['camera_residual_px'] = float(np.linalg.norm(x[:2]/x[2]-point))
    return features, previous_point


def deterministic_verdict(features, confidence):
    """Conservative veto, never a calibrated probability or confidence inflation."""
    line_like = features['aspect_ratio'] > 3 and features['edge_coherence'] > .9
    static = features['camera_residual_px'] is not None and features['camera_residual_px'] < 1
    unsupported_flow = features['forward_backward_error_px'] is not None and features['forward_backward_error_px'] > 4
    rejected = confidence < .8 and ((line_like and static) or (line_like and unsupported_flow))
    return {'accepted': not rejected, 'score': None, 'reason': 'LINE_LIKE_WITH_TEMPORAL_CONTRADICTION' if rejected else 'NO_CONCLUSIVE_VISUAL_VETO'}


def batch_visual_features(previous, current, candidates, motion=None):
    """Share source-image flow pyramids across all candidate centers in a frame."""
    results = [visual_features(None, current, candidate)[0] for candidate in candidates]
    if previous is None or not candidates:
        return results
    pgray, cgray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY), cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    points = np.float32([center(c['bbox']) for c in candidates]).reshape(-1, 1, 2)
    old, status, _ = cv2.calcOpticalFlowPyrLK(cgray, pgray, points, None, winSize=(19, 19), maxLevel=2)
    if old is None or status is None:
        return results
    back, reverse, _ = cv2.calcOpticalFlowPyrLK(pgray, cgray, old, None, winSize=(19, 19), maxLevel=2)
    if back is None or reverse is None:
        return results
    for i, (candidate, features) in enumerate(zip(candidates, results, strict=True)):
        q, p = old[i, 0], points[i, 0]
        if not status[i, 0] or not reverse[i, 0] or not np.isfinite(q).all() or not np.isfinite(back[i]).all() or not (0 <= q[0] < previous.shape[1] and 0 <= q[1] < previous.shape[0]):
            continue
        radius = max(5, min(24, math.ceil(max(candidate['bbox'][2:]))))
        a = cv2.cvtColor(patch(previous, q, radius), cv2.COLOR_BGR2GRAY).astype(float).ravel()
        b = cv2.cvtColor(patch(current, p, radius), cv2.COLOR_BGR2GRAY).astype(float).ravel()
        features.update(history_available=True, forward_backward_error_px=float(np.linalg.norm(back[i, 0]-p)))
        if min(a.std(), b.std()) > 1e-6:
            features['patch_correlation'] = float(np.clip(np.corrcoef(a, b)[0, 1], -1, 1))
        if motion is not None:
            x = np.asarray(motion, float)@np.r_[q, 1.]
            if np.isfinite(x).all() and abs(x[2]) > 1e-9:
                features['camera_residual_px'] = float(np.linalg.norm(x[:2]/x[2]-p))
    return results


FEATURE_NAMES = ['detector_logit', 'aspect_log', 'edge_coherence', 'contrast',
                 'forward_backward_error', 'camera_residual', 'patch_correlation',
                 'history_missing', 'camera_missing', 'correlation_missing']


def feature_vector(features, confidence, extended=False):
    score = float(np.clip(confidence, .001, .999))
    values = [math.log(score/(1-score)), math.log(max(1, features['aspect_ratio'])),
            features['edge_coherence'], features['contrast'],
            min(features['forward_backward_error_px'] or 0, 20)/20,
            min(features['camera_residual_px'] or 0, 40)/40,
            features['patch_correlation'] or 0, float(not features['history_available']),
            float(features['camera_residual_px'] is None), float(features['patch_correlation'] is None)]
    if extended:
        values.extend([features.get('external_temporal_score') or 0, float(features.get('external_temporal_score') is None)])
    return values


def fit_linear(features, labels, weights=None, penalty=2.):
    """Strongly regularized tiny baseline, not a neural model trained on 36 frames.

    Callers split by source recording BEFORE fitting scaling or class weights.
    Candidate negatives must be explicitly documented as development supervision
    derived from the human-identified match ball, not independent negative truth.
    """
    from scipy.optimize import minimize
    x, y = np.asarray(features, float), np.asarray(labels, float)
    if x.ndim != 2 or x.shape[1] not in [len(FEATURE_NAMES),len(FEATURE_NAMES)+2] or len(x) != len(y) or not np.isfinite(x).all() or set(y) != {0., 1.}:
        return {'available': False, 'reason': 'BOTH_CLASSES_REQUIRED_IN_TRAINING_SOURCES'}
    mean, scale = x.mean(axis=0), np.maximum(x.std(axis=0), .05)
    z = np.c_[np.clip((x-mean)/scale, -10, 10), np.ones(len(x))]
    weights = np.ones(len(y)) if weights is None else np.asarray(weights, float)
    for label in [0, 1]:
        weights[y == label] /= max(weights[y == label].sum(), 1e-9)
    weights /= weights.sum()
    def objective(theta):
        logits = z@theta
        likelihood = np.logaddexp(0, logits)-y*logits
        loss = np.sum(weights*likelihood)+penalty*np.sum(theta[:-1]**2)/len(x)
        p = 1/(1+np.exp(-np.clip(logits, -40, 40)))
        gradient = z.T@(weights*(p-y))
        gradient[:-1] += 2*penalty*theta[:-1]/len(x)
        return loss, gradient
    fit = minimize(objective, np.zeros(z.shape[1]), jac=True, method='L-BFGS-B')
    if not fit.success:
        return {'available': False, 'reason': 'OPTIMIZATION_DID_NOT_CONVERGE'}
    names = FEATURE_NAMES+(['external_temporal_score','external_temporal_missing'] if x.shape[1] > len(FEATURE_NAMES) else [])
    return {'available': True, 'feature_names': names, 'mean': mean.tolist(),
            'scale': scale.tolist(), 'weights': fit.x.tolist(), 'l2_penalty': penalty,
            'training_candidates': len(y), 'training_positives': int(y.sum()),
            'threshold': .5, 'score_calibrated': False}


def linear_score(model, features, confidence):
    if not model['available']:
        return None
    extended = model['feature_names'] == FEATURE_NAMES+['external_temporal_score','external_temporal_missing']
    if model['feature_names'] != FEATURE_NAMES and not extended:
        raise ValueError('Temporal feature schema changed')
    x = np.asarray(feature_vector(features, confidence, extended))
    z = np.clip((x-np.asarray(model['mean']))/np.asarray(model['scale']), -10, 10)
    logit = np.r_[z, 1]@np.asarray(model['weights'])
    return float(1/(1+np.exp(-np.clip(logit, -40, 40))))


class LearnedVerifier:
    """Reuse the existing external-training VisualPair weights; no demo-label training."""
    def __init__(self, checkpoint, threshold=.5):
        import torch
        from ml.ball16.temporal import VisualPair
        self.torch = torch
        self.model = VisualPair()
        self.model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)['model'])
        self.model.eval()
        self.threshold = threshold

    def verify(self, previous, current, candidate, previous_point):
        if previous is None or previous_point is None:
            return {'accepted': True, 'score': None, 'reason': 'NO_VALID_CONSECUTIVE_HISTORY; VERIFIER_ABSTAINS'}
        from ml.ball16.temporal import pair
        a = cv2.cvtColor(patch(previous, previous_point), cv2.COLOR_BGR2RGB)
        b = cv2.cvtColor(patch(current, center(candidate['bbox'])), cv2.COLOR_BGR2RGB)
        with self.torch.inference_mode():
            score = float(self.model(pair(a, b)[None]).sigmoid()[0])
        return {'accepted': score >= self.threshold, 'score': score,
                'reason': 'FROZEN_VISUAL_PAIR_ACCEPT' if score >= self.threshold else 'FROZEN_VISUAL_PAIR_REJECT'}
