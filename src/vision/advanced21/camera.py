"""Field-masked bidirectional flow/RANSAC proposals alongside unchanged M8 mapping."""
import cv2
import numpy as np
from src.vision.ball.geometry import camera_motion


def causal_camera(camera, refinement=None):
    """Do not feed M8's offline absolute/bidirectional calibration into a live layer."""
    field = refinement if refinement and refinement.get('accepted') else camera.get('motion_evidence', {}).get('field_motion', {})
    field = field if field.get('accepted') and field.get('interval_frames') == 1 else {}
    return {'epoch': camera['epoch'], 'continuity': camera['continuity'],
            'calibration_valid': False, 'calibration': None, 'motion_evidence': {'field_motion': field}}


def refine_motion(previous, current, previous_camera, camera, previous_people=(), include_support=False):
    unavailable = {'accepted': False, 'homography': None, 'independent_pitch_reprojection_error_px': None,
                   'absolute_calibration_changed': False, 'pitch_dimensions_m': [105, 68]}
    if previous is None or previous_camera is None or previous_camera['epoch'] != camera['epoch'] or camera.get('continuity') == 'cut':
        return {**unavailable, 'reason': 'NO_SAME_EPOCH_HISTORY'}
    h, w = current.shape[:2]
    scale = min(1., 640/w)
    size = (round(w*scale), round(h*scale))
    a, b = cv2.resize(previous, size), cv2.resize(current, size)
    hsv = cv2.cvtColor(a, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (25, 30, 20), (100, 255, 255))
    for person in previous_people:
        x, y, bw, bh = [round(v*scale) for v in person['bbox']]
        cv2.rectangle(mask, (x, y), (x+bw, y+bh), 0, -1)
    gray_a, gray_b = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    points = cv2.goodFeaturesToTrack(gray_a, 250, .01, 8, mask=mask)
    if points is None or len(points) < 12:
        return {**unavailable, 'reason': 'INSUFFICIENT_FIELD_FEATURES'}
    q, status, _ = cv2.calcOpticalFlowPyrLK(gray_a, gray_b, points, None, winSize=(21, 21), maxLevel=3)
    if q is None:
        return {**unavailable, 'reason': 'FORWARD_FLOW_FAILED'}
    back, reverse, _ = cv2.calcOpticalFlowPyrLK(gray_b, gray_a, q, None, winSize=(21, 21), maxLevel=3)
    if back is None:
        return {**unavailable, 'reason': 'BACKWARD_FLOW_FAILED'}
    valid = status.ravel().astype(bool)&reverse.ravel().astype(bool)&(np.linalg.norm(points-back, axis=2).ravel() <= 1.5)
    src, dst = points.reshape(-1, 2)[valid], q.reshape(-1, 2)[valid]
    if len(src) < 12:
        return {**unavailable, 'reason': 'INSUFFICIENT_BIDIRECTIONAL_SUPPORT'}
    fit, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 2)
    if fit is None or inliers is None or not np.isfinite(fit).all():
        return {**unavailable, 'reason': 'ROBUST_FIELD_FIT_FAILED'}
    inliers = inliers.ravel().astype(bool)
    support = int(inliers.sum())
    coverage = cv2.contourArea(cv2.convexHull(src[inliers]))/(size[0]*size[1]) if support >= 3 else 0
    matrix = np.diag([1/scale, 1/scale, 1])@fit@np.diag([scale, scale, 1])
    matrix /= matrix[2, 2]
    if support < 12 or support/len(src) < .6 or coverage < .03 or abs(np.linalg.det(matrix)) < 1e-8:
        return {**unavailable, 'reason': 'WEAK_OR_DEGENERATE_FIELD_SUPPORT', 'inliers': support}
    prediction = cv2.perspectiveTransform(src[inliers, None, :], fit).reshape(-1, 2)
    error = np.linalg.norm(prediction-dst[inliers], axis=1)/scale
    prior, _ = camera_motion(previous_camera, camera)
    original_error = None
    if prior is not None:
        projected = cv2.perspectiveTransform((src[inliers]/scale)[:, None, :], np.asarray(prior, float)).reshape(-1, 2)
        original_error = float(np.median(np.linalg.norm(projected-dst[inliers]/scale, axis=1)))
    result = {**unavailable, 'accepted': True, 'homography': matrix.tolist(), 'interval_frames': 1,
            'reason': 'RELATIVE_FIELD_RANSAC_ONLY; ABSOLUTE_PITCH_REFERENCE_REQUIRED',
            'inliers': support, 'feature_support': len(src), 'field_support_area_fraction': float(coverage),
            'internal_median_residual_px': float(np.median(error)), 'internal_p90_residual_px': float(np.quantile(error, .9)),
            'm8_internal_median_residual_px': original_error}
    if include_support:
        result.update(support_source_pixels=(src[inliers]/scale).tolist(),support_target_pixels=(dst[inliers]/scale).tolist())
    return result


def smooth_relative(current, previous, width, height, fps, time_constant=.08):
    """Research-only causal smoothing of relative image motion, never absolute calibration."""
    current=np.asarray(current,float)
    if previous is None:return current
    x,y=np.meshgrid(np.linspace(0,width-1,5),np.linspace(0,height-1,5));grid=np.c_[x.ravel(),y.ravel()].astype(np.float32)
    a=cv2.perspectiveTransform(grid[:,None,:],current).reshape(-1,2)
    b=cv2.perspectiveTransform(grid[:,None,:],np.asarray(previous,float)).reshape(-1,2)
    alpha=1-np.exp(-1/(fps*time_constant));target=alpha*a+(1-alpha)*b
    matrix,_=cv2.findHomography(grid,target,0)
    if matrix is None or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix))<1e-8:return current
    return matrix/matrix[2,2]
