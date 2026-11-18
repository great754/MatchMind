"""Continuous-time IMM mixing and image-space Kalman motion hypotheses.

Models describe image motion compatible with rolling/curved flight/deflections;
their probabilities do not classify physical ball height or ground contact.
"""
import math
import numpy as np
from scipy.linalg import expm
from scipy.special import logsumexp
from src.vision.ball.geometry import warp_point


def positive(matrix):
    """Keep symmetric covariance numerically positive without hiding NaNs."""
    matrix = (matrix + matrix.T) / 2
    if not np.isfinite(matrix).all():
        raise ValueError('Nonfinite motion covariance')
    values, vectors = np.linalg.eigh(matrix)
    return (vectors * np.maximum(values, 1e-6)) @ vectors.T


def jacobian(h, point):
    x, y = point
    denominator = h[2, 0]*x + h[2, 1]*y + h[2, 2]
    target = warp_point(h, point)
    if target is None or abs(denominator) < 1e-6:
        return None
    j = (h[:2, :2] - np.outer(target, h[2, :2])) / denominator
    singular = np.linalg.svd(j, compute_uv=False)
    return j if singular.min() >= .2 and singular.max() <= 5 else None


class MotionBank:
    def __init__(self, point, timestamp, diagonal, config):
        self.cfg = config
        self.diagonal = diagonal
        self.names = list(config['models'])
        if len(self.names) != 4 or config['measurement_sigma_px'] <= 0:
            raise ValueError('Four compatible motion models and positive visual noise required')
        self.mu = np.asarray(config['initial_probabilities'], float)
        if self.mu.shape != (4,) or np.any(self.mu <= 0):
            raise ValueError('Positive initial motion probabilities required')
        self.mu /= self.mu.sum()
        self.x = np.zeros((4, 6)); self.x[:, :2] = point
        sigmas = [config['initial_position_sigma_px']]*2 + [diagonal*config['initial_velocity_diagonals']]*2 + [diagonal]*2
        self.p = np.tile(np.diag(np.square(sigmas)), (4, 1, 1))
        self.time = timestamp
        self.last_observed_at = timestamp
        self.camera_status = 'INITIAL_IMAGE_FRAME'

    def combined(self):
        mean = self.mu @ self.x
        covariance = sum(w*(p + np.outer(x-mean, x-mean)) for w, x, p in zip(self.mu, self.x, self.p))
        return mean, positive(covariance)

    def predict(self, timestamp, homography=None, camera_status='UNKNOWN', contact=False):
        dt = timestamp - self.time
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError('Motion prediction requires increasing finite timestamps')
        self.time = timestamp
        # Warp posterior state and uncertainty into the current image frame first.
        supported = homography is not None
        if supported:
            transformed = []
            for x, p in zip(self.x, self.p):
                j = jacobian(homography, x[:2])
                point = warp_point(homography, x[:2])
                if j is None or point is None:
                    supported = False; break
                transform = np.zeros((6, 6))
                for k in [0, 2, 4]: transform[k:k+2, k:k+2] = j
                state = transform @ x; state[:2] = point
                transformed.append((state, positive(transform @ p @ transform.T)))
            if supported:
                self.x = np.array([v[0] for v in transformed]); self.p = np.array([v[1] for v in transformed])
        self.camera_status = camera_status if supported else 'UNKNOWN_OR_REJECTED_CAMERA_WARP'
        # Row i -> column j transition probabilities, using actual elapsed time.
        rates = np.ones((4, 4)) * self.cfg['switch_rate_per_second'] / 3
        np.fill_diagonal(rates, -self.cfg['switch_rate_per_second'])
        transition = expm(rates * dt)
        prior = self.mu @ transition
        mixed_x = []; mixed_p = []
        for j in range(4):
            weights = self.mu * transition[:, j] / max(prior[j], 1e-12)
            mean = weights @ self.x
            covariance = sum(w*(p + np.outer(x-mean, x-mean)) for w, x, p in zip(weights, self.x, self.p))
            mixed_x.append(mean); mixed_p.append(covariance)
        self.mu = prior / prior.sum()
        for i, name in enumerate(self.names):
            f = np.eye(6); f[0, 2] = f[1, 3] = dt
            if name == 'constant_velocity':
                f[4, 4] = f[5, 5] = math.exp(-12*dt)
            else:
                f[0, 4] = f[1, 5] = .5*dt*dt
                f[2, 4] = f[3, 5] = dt
                if name != 'curved_acceleration': f[4, 4] = f[5, 5] = math.exp(-5*dt)
            scale = self.cfg['process_acceleration_diagonals'][i] * self.diagonal
            if contact and name in ['maneuver', 'occlusion_diffuse']: scale *= 1.5
            q = np.diag([(.5*dt*dt*scale)**2]*2 + [(dt*scale)**2]*2 + [(math.sqrt(dt)*scale)**2]*2)
            # Camera-plane warp is approximate for unknown-height balls. Inflate it.
            inflation = 1.5 if supported else self.cfg['camera_unknown_inflation_px']
            q[:2, :2] += np.eye(2) * inflation**2 * dt * 25
            self.x[i] = f @ mixed_x[i]
            self.p[i] = positive(f @ mixed_p[i] @ f.T + q)
        return self.combined()

    def innovation(self, point):
        mean, covariance = self.combined()
        s = covariance[:2, :2] + np.eye(2)*self.cfg['measurement_sigma_px']**2
        delta = np.asarray(point) - mean[:2]
        return float(delta @ np.linalg.solve(s, delta))

    def update(self, point, timestamp):
        if abs(timestamp-self.time) > 1e-8:
            raise ValueError('Measurement must match predicted frame timestamp')
        r = np.eye(2)*self.cfg['measurement_sigma_px']**2
        h = np.zeros((2, 6)); h[:, :2] = np.eye(2)
        log_likelihood = []
        for i in range(4):
            residual = np.asarray(point) - self.x[i, :2]
            s = positive(h @ self.p[i] @ h.T + r)
            gain = np.linalg.solve(s, h @ self.p[i]).T
            self.x[i] += gain @ residual
            # Joseph update retains positive covariance under finite precision.
            a = np.eye(6)-gain@h
            self.p[i] = positive(a @ self.p[i] @ a.T + gain @ r @ gain.T)
            log_likelihood.append(-.5*(residual @ np.linalg.solve(s, residual) + np.linalg.slogdet(s)[1] + 2*math.log(2*math.pi)))
        log_weights = np.log(np.maximum(self.mu, 1e-300)) + log_likelihood
        self.mu = np.exp(log_weights-logsumexp(log_weights))
        self.last_observed_at = timestamp

    def public(self):
        mean, covariance = self.combined()
        radius = math.sqrt(4.605170186 * np.linalg.eigvalsh(covariance[:2, :2]).max())
        return {'mean': mean[:2].tolist(), 'velocity': mean[2:4].tolist(), 'acceleration': mean[4:].tolist(),
                'covariance': covariance[:2, :2].tolist(), 'radius90_px': radius,
                'motion_probabilities': dict(zip(self.names, map(float, self.mu))),
                'age_seconds': self.time-self.last_observed_at, 'camera_status': self.camera_status}
