"""Camera-aware diagnostics around the existing M21 candidate DAG."""
import math
import numpy as np
from src.vision.ball.geometry import camera_motion, warp_point
from src.vision.advanced21.offline import point, segments


def relative_camera(cameras, start, end):
    if start > end:
        matrix = relative_camera(cameras, end, start)
        return np.linalg.inv(matrix) if matrix is not None else None
    result = np.eye(3)
    for i in range(start+1, end+1):
        step, _ = camera_motion(cameras[i-1], cameras[i])
        if step is None:
            return None
        result = step@result
        if not np.isfinite(result).all() or abs(np.linalg.det(result)) < 1e-10:
            return None
    return result


def reconstruct_camera(selected, cameras, fps, width, height, maximum_gap_seconds=.24):
    inferred = [None]*len(selected)
    for start, end in segments(cameras):
        observed = [i for i in range(start, end) if selected[i] is not None]
        for a, b in zip(observed, observed[1:]):
            if b-a <= 1 or (b-a-1)/fps > maximum_gap_seconds:
                continue
            for i in range(a+1, b):
                left = relative_camera(cameras, a, i)
                right = relative_camera(cameras, b, i)
                if left is None or right is None:
                    continue
                x, y = warp_point(left, point(selected[a])), warp_point(right, point(selected[b]))
                if x is None or y is None:
                    continue
                t = (i-a)/(b-a)
                xy = (1-t)*x+t*y
                if not (0 <= xy[0] < width and 0 <= xy[1] < height):
                    continue
                inferred[i] = {'image': xy.tolist(), 'directly_observed': False, 'support_frames': [a, b],
                    'uses_future': True, 'reason': 'M23_CAMERA_COMPENSATED_TWO_SIDED_INTERPOLATION; NOT_CURRENT_OBSERVATION'}
    return inferred


def history(selected, cameras, frame, fps):
    result = []
    for i in range(max(0, frame-round(fps)), frame):
        if selected[i] is None:
            continue
        h = relative_camera(cameras, i, frame)
        xy = warp_point(h, point(selected[i])) if h is not None else None
        if xy is not None:
            result.append({'frame': i, 'source_image': point(selected[i]).tolist(), 'current_camera_image': xy.tolist(),
                'directly_observed_at_source_frame': True, 'current_observation': False})
    return result


def appearance_edge(descriptors, a, b, dt):
    x, y = descriptors.get(a['candidate_id']), descriptors.get(b['candidate_id'])
    if x is None or y is None:
        return 0.
    score = float(np.clip(np.sqrt(x*y).sum(), 0, 1))
    return .1*(1-score)


def anchored_complements(primary, candidates, cameras, fps, diagonal, descriptors, maximum_gap_seconds=.35):
    """M22 cannot displace the primary path or start a self-reinforcing trajectory.

    Only actual, already verified neural candidates can fill a short gap between
    primary observations in the same camera epoch. Camera-relative endpoints,
    scale, appearance and the existing DAG constrain the connector. The corridor
    is an uncertainty bound, never a synthetic visual measurement.
    """
    from src.vision.advanced21.offline import dag_link
    selected=list(primary)
    for start,end in segments(cameras):
        anchors=[i for i in range(start,end) if primary[i] is not None]
        for a,b in zip(anchors,anchors[1:]):
            if b-a<=1 or (b-a-1)/fps>maximum_gap_seconds:continue
            # Include the anchors' original adjacent visual support. Cropping
            # exactly at an anchor can falsely remove it in the unchanged DAG's
            # repeated-visual-support check when the next gap frame is missing.
            lo,hi=max(start,a-1),min(end,b+2)
            options=[([primary[i]] if primary[i] is not None else []) for i in range(lo,hi)]
            for i in range(a+1,b):
                left,right=relative_camera(cameras,a,i),relative_camera(cameras,b,i)
                if left is None or right is None:continue
                x,y=warp_point(left,point(primary[a])),warp_point(right,point(primary[b]))
                if x is None or y is None:continue
                t=(i-a)/(b-a);mean=(1-t)*x+t*y
                radius=max(6.,2*max(*primary[a]['bbox'][2:],*primary[b]['bbox'][2:]),.12*diagonal*(b-a)/fps)
                for c in candidates[i]:
                    if c.get('origin')!='complement' or not c.get('observation_eligible'):continue
                    if float(np.linalg.norm(point(c)-mean))>radius:continue
                    scale=math.sqrt(c['bbox'][2]*c['bbox'][3]/max(primary[a]['bbox'][2]*primary[a]['bbox'][3],1))
                    if not .25<=scale<=4:continue
                    appearance=descriptors.get((i,c['candidate_id']))
                    supports=[descriptors.get((j,primary[j]['candidate_id'])) for j in (a,b)]
                    if appearance is None or any(s is None or float(np.sqrt(s*appearance).sum())<.25 for s in supports):continue
                    options[i-lo].append(c)
            linked=dag_link(options,cameras[lo:hi],fps,diagonal,maximum_gap_seconds=maximum_gap_seconds)
            # The primary endpoints stay authoritative even if the gap abstains.
            if linked[a-lo] is None or linked[b-lo] is None:continue
            for i in range(a+1,b):
                if linked[i-lo] is not None:selected[i]=linked[i-lo]
    return selected
