"""Research-only metric flight fitting. A ground homography is deliberately insufficient."""
import numpy as np
from scipy.optimize import least_squares


def image_quadratic(times, pixels):
    times, pixels = np.asarray(times, float), np.asarray(pixels, float)
    if len(times) < 3 or pixels.shape != (len(times), 2) or not np.isfinite(times).all() or not np.isfinite(pixels).all() or np.ptp(times) <= 0:
        return None
    coefficients = np.polynomial.polynomial.polyfit(times-times[0], pixels, 2)
    predicted = np.polynomial.polynomial.polyval(times-times[0], coefficients).T
    return {'rms_px': float(np.sqrt(np.mean(np.sum((pixels-predicted)**2, axis=1)))),
            'coefficients': coefficients.tolist(), 'metric_position': None, 'directly_observed': False}


def fit_ballistic(times, pixels, projections=None, *, calibration_verified=False, maximum_rms_px=4):
    result = {'accepted': False, 'experimental': True, 'directly_observed': False,
              'eligible_for_pass_gates': False, 'authoritative_ball_height': None}
    if not calibration_verified or projections is None:
        return {**result, 'reason': 'VERIFIED_FULL_3D_CAMERA_PROJECTIONS_REQUIRED; GROUND_HOMOGRAPHY_INSUFFICIENT'}
    t, uv, projection = np.asarray(times, float), np.asarray(pixels, float), np.asarray(projections, float)
    if len(t) < 6 or uv.shape != (len(t), 2) or projection.shape != (len(t), 3, 4) or not all(np.isfinite(a).all() for a in [t, uv, projection]) or np.any(np.diff(t) <= 0) or np.ptp(t) < .25:
        return {**result, 'reason': 'INSUFFICIENT_FINITE_TIME_SEPARATED_OBSERVATIONS'}
    tau = t-t[0]
    gravity = np.column_stack([np.zeros(len(t)), np.zeros(len(t)), -.5*9.81*tau**2])
    matrix, target = [], []
    for dt, point, p, g in zip(tau, uv, projection, gravity, strict=True):
        for k in [0, 1]:
            a = p[k, :3]-point[k]*p[2, :3]
            matrix.append(np.r_[a, dt*a])
            target.append(-a@g-p[k, 3]+point[k]*p[2, 3])
    matrix, target = np.asarray(matrix), np.asarray(target)
    if np.linalg.matrix_rank(matrix) < 6 or np.linalg.cond(matrix) > 1e8:
        return {**result, 'reason': 'DEPTH_OR_VELOCITY_NOT_IDENTIFIABLE'}
    initial = np.linalg.lstsq(matrix, target, rcond=None)[0]
    def positions(theta):
        return theta[:3]+tau[:, None]*theta[3:]+gravity
    def residual(theta):
        points = np.c_[positions(theta), np.ones(len(t))]
        projected = np.einsum('nij,nj->ni', projection, points)
        depth = projected[:, 2:]
        return ((projected[:, :2]/np.where(np.abs(depth) < 1e-8, 1e-8, depth))-uv).ravel()
    fit = least_squares(residual, initial, loss='soft_l1', f_scale=2, max_nfev=100)
    xyz = positions(fit.x)
    rms = float(np.sqrt(np.mean(np.sum(residual(fit.x).reshape(-1, 2)**2, axis=1))))
    points = np.c_[xyz, np.ones(len(t))]
    depth = np.einsum('nij,nj->ni', projection, points)[:, 2]
    if not fit.success or not np.isfinite(xyz).all() or rms > maximum_rms_px or np.any(depth <= 0) or np.min(xyz[:, 2]) < -.1 or np.max(xyz[:, 2]) > 30 or np.linalg.norm(fit.x[3:]) > 60:
        return {**result, 'reason': 'PHYSICAL_OR_REPROJECTION_CHECK_FAILED', 'rms_px': rms}
    covariance = np.linalg.pinv(fit.jac.T@fit.jac)*max(rms*rms, 1e-6)
    return {**result, 'accepted': True, 'reason': 'EXPERIMENTAL_GRAVITY_CONSTRAINED_REPROJECTION_FIT',
            'p0_m': fit.x[:3].tolist(), 'v0_m_s': fit.x[3:].tolist(), 'covariance': covariance.tolist(),
            'rms_px': rms, 'observations': len(t), 'image_2d_comparison': image_quadratic(t, uv)}
