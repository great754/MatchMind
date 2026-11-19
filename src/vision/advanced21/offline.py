"""Offline-only candidate linking: frame Viterbi and a skip-edge candidate DAG.

Current observations always retain their original pixels, boxes and confidence.
Two-sided reconstructions are diagnostics and never observations.
"""
import math
import numpy as np
from src.vision.ball.geometry import camera_motion, warp_point


def point(candidate):
    x, y, w, h = candidate['bbox']
    return np.array([x+w/2, y+h/2])


def segments(cameras):
    start = 0
    for i in range(1, len(cameras)):
        if cameras[i]['epoch'] != cameras[i-1]['epoch'] or cameras[i].get('continuity') == 'cut':
            yield start, i
            start = i
    if cameras:
        yield start, len(cameras)


def transition(a, b, dt, diagonal, transform=None):
    old = point(a)
    if transform is not None:
        moved = warp_point(transform, old)
        if moved is not None:
            old = moved
    distance = np.linalg.norm(point(b)-old)
    if distance > diagonal*2*dt+12:
        return math.inf
    radius = max(6, 2*max(a['bbox'][2:]), 2*max(b['bbox'][2:]))
    motion = min(8., distance/max(radius, .12*diagonal*dt))
    scale = abs(math.log(max(b['bbox'][2]*b['bbox'][3], 1)/max(a['bbox'][2]*a['bbox'][3], 1)))
    return float(.15*motion+.1*min(scale, 4))


def viterbi(candidates, cameras, fps, diagonal, threshold=.35):
    """One state per frame, including explicit missing; future evidence may change earlier choices."""
    selected = [None]*len(candidates)
    for start, end in segments(cameras):
        layers = [[None]+[c for c in row if c['confidence'] >= threshold] for row in candidates[start:end]]
        costs = None
        back = []
        for offset, layer in enumerate(layers):
            emit = np.array([.8 if c is None else -math.log(max(c['confidence'], 1e-9)) for c in layer])
            if costs is None:
                costs = emit
                back.append(None)
                continue
            motion, _ = camera_motion(cameras[start+offset-1], cameras[start+offset])
            edge = np.zeros((len(layers[offset-1]), len(layer)))
            for i, a in enumerate(layers[offset-1]):
                for j, b in enumerate(layer):
                    edge[i, j] = .25 if (a is None) != (b is None) else 0 if a is None else transition(a, b, 1/fps, diagonal, motion)
            total = costs[:, None]+edge
            parents = np.argmin(total, axis=0)
            costs = emit+total[parents, np.arange(len(layer))]
            back.append(parents)
        index = int(np.argmin(costs))
        for offset in range(len(layers)-1, -1, -1):
            selected[start+offset] = layers[offset][index]
            if offset:
                index = int(back[offset][index])
    return repeated_visual_support(selected, cameras, fps)


def dag_link(candidates, cameras, fps, diagonal, threshold=.35, maximum_gap_seconds=.35):
    """Maximum-score path through a time-ordered DAG; skip edges model bounded occlusions.

    Each candidate is visited once in time order. Positive contiguous paths can
    restart; no edges cross an epoch, and unobserved nodes are never invented.
    """
    selected = [None]*len(candidates)
    maximum_gap = max(1, math.floor(maximum_gap_seconds*fps))
    for start, end in segments(cameras):
        nodes = []
        for frame in range(start, end):
            for candidate in candidates[frame]:
                if candidate['confidence'] < threshold:
                    continue
                reward = math.log(max(candidate['confidence'], 1e-6)/max(1-candidate['confidence'], 1e-6))+.6
                best, parent = reward-.4, None
                for j in range(len(nodes)-1, -1, -1):
                    prior = nodes[j]
                    gap = frame-prior['frame']
                    if gap <= 0:
                        continue
                    if gap > maximum_gap:
                        break
                    transform = np.eye(3)
                    for k in range(prior['frame']+1, frame+1):
                        step, _ = camera_motion(cameras[k-1], cameras[k])
                        if step is None:
                            transform = None
                            break
                        transform = step@transform
                    cost = transition(prior['candidate'], candidate, gap/fps, diagonal, transform)+.12*(gap-1)
                    score = prior['score']+reward-cost
                    if score > best:
                        best, parent = score, j
                nodes.append({'frame': frame, 'candidate': candidate, 'score': best, 'parent': parent})
        # Recover disjoint-in-time paths, strongest first. Never select two balls in one frame.
        occupied = set()
        for tip in sorted(range(len(nodes)), key=lambda j: (-nodes[j]['score'], j)):
            if nodes[tip]['score'] <= 0:
                continue
            path = []
            cursor = tip
            while cursor is not None:
                path.append(cursor)
                cursor = nodes[cursor]['parent']
            if len(path) < 2:
                continue
            lo, hi = nodes[path[-1]]['frame'], nodes[path[0]]['frame']
            if any(frame in occupied for frame in range(lo, hi+1)):
                continue
            for j in path:
                selected[nodes[j]['frame']] = nodes[j]['candidate']
            occupied.update(range(lo, hi+1))
    return repeated_visual_support(selected, cameras, fps)


def repeated_visual_support(selected, cameras, fps):
    """Retain the unchanged repeated-visual-birth principle, even offline."""
    out = list(selected)
    for start, end in segments(cameras):
        for i in range(start, end):
            if selected[i] is None:
                continue
            neighbors = [j for j in range(max(start, i-1), min(end, i+2)) if j != i and selected[j] is not None]
            if not neighbors:
                out[i] = None
    return out


def reconstruct(selected, cameras, fps, maximum_gap_seconds=.24):
    """Two-sided image interpolation is explicitly inferred and has no ground-plane position."""
    inferred = [None]*len(selected)
    for start, end in segments(cameras):
        observed = [i for i in range(start, end) if selected[i] is not None]
        for a, b in zip(observed, observed[1:]):
            if b-a <= 1 or (b-a-1)/fps > maximum_gap_seconds:
                continue
            for frame in range(a+1, b):
                fraction = (frame-a)/(b-a)
                xy = point(selected[a])*(1-fraction)+point(selected[b])*fraction
                inferred[frame] = {'image': xy.tolist(), 'directly_observed': False,
                                   'support_frames': [a, b], 'uses_future': True,
                                   'reason': 'OFFLINE_TWO_SIDED_IMAGE_INTERPOLATION; NOT_BALL_HEIGHT'}
    return inferred
