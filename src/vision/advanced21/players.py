"""Two-stage tracking by detection with camera motion and existing M9 appearance.

Alternative identities are isolated, anonymous experiments. They never inherit
manual names, merge existing M9 segments or replace pass-participant references.
"""
import math
import numpy as np
from scipy.optimize import linear_sum_assignment
from src.evaluation.cv_m19 import iou
from src.vision.ball.geometry import camera_motion, warp_point


def cosine(a, b):
    if a is None or b is None:
        return None
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or min(np.linalg.norm(a), np.linalg.norm(b)) <= 1e-9:
        return None
    return float(np.clip(a@b/(np.linalg.norm(a)*np.linalg.norm(b)), -1, 1))


def observations(world):
    people = {p['current_track_id']: p for p in world['player_world']['players']
              if p['visibility_state'] == 'VISIBLE' and p.get('current_track_id')}
    rows = [{'track_id': p['track_id'], 'bbox': p['image_bbox'],
             'confidence': p.get('detection_confidence'), 'appearance': p.get('appearance')}
            for p in world['entities'] if p['visibility'] == 'visible' and p.get('image_bbox')]
    rows += [{'track_id': u['observation']['track_id'], 'bbox': u['observation']['bbox'],
              'confidence': u['observation'].get('detection_confidence'),
              'appearance': u['observation'].get('appearance')} for u in world['unassigned']]
    for row in rows:
        person = people.get(row['track_id'], {})
        row['team'] = person.get('team', {}).get('label', 'UNKNOWN')
        row['team_confidence'] = person.get('team', {}).get('confidence', 0)
        row['m9_entity'] = person.get('entity_id')
        row['m9_segment'] = person.get('current_segment_id')
        current = person.get('current_appearance') or {}
        row['histogram'] = current.get('histogram') or row['appearance']
        row['identity_status'] = person.get('identity_status', 'UNKNOWN')
    return rows


class PlayerTracker:
    def __init__(self, fps, config, compensated=True, appearance=True, deep=False):
        self.fps, self.config = fps, config
        self.compensated, self.appearance = compensated, appearance
        self.deep = deep
        self.tracks = {}
        self.previous_camera = None
        self.next_id = 1

    def step(self, frame, camera, detections):
        cut = self.previous_camera is not None and (camera['epoch'] != self.previous_camera['epoch'] or camera.get('continuity') == 'cut')
        motion, source = camera_motion(self.previous_camera, camera)
        self.previous_camera = camera
        if cut:
            self.tracks.clear()
        active = {k: t for k, t in self.tracks.items()
                  if (frame-t['last'])/self.fps <= self.config['maximum_missing_seconds']}
        for track in active.values():
            box = np.array(track['bbox'], float)
            if self.compensated and motion is not None:
                a, b = warp_point(motion, box[:2]), warp_point(motion, box[:2]+box[2:])
                if a is not None and b is not None and np.all(b > a):
                    box = np.r_[a, b-a]
            box[:2] += np.asarray(track['velocity'])/self.fps
            track['predicted_bbox'] = box.tolist()
        assigned, used = {}, set()
        # Missing source scores are not fabricated as high-confidence detections.
        high = [i for i, d in enumerate(detections) if d['confidence'] is None or d['confidence'] >= self.config['high_confidence']]
        low = [i for i, d in enumerate(detections) if d['confidence'] is not None and self.config['low_confidence'] <= d['confidence'] < self.config['high_confidence']]
        for indices, second in [(high, False), (low, True)]:
            tracks = [t for k, t in active.items() if k not in assigned and (not second or t['last'] == frame-1)]
            indices = [i for i in indices if i not in used]
            costs = np.full((len(tracks), len(indices)), 1e6)
            for a, track in enumerate(tracks):
                for b, index in enumerate(indices):
                    d = detections[index]
                    overlap = iou(track['predicted_bbox'], d['bbox'])
                    conflict = (track['team'] != d['team'] and track['team'] in ['TEAM_A', 'TEAM_B', 'OFFICIAL']
                                and d['team'] in ['TEAM_A', 'TEAM_B', 'OFFICIAL']
                                and min(track['team_confidence'], d['team_confidence']) >= .8)
                    deep = cosine(track.get('embedding'),d.get('embedding')) if self.deep else None
                    bank = track.get('embedding_samples',[])
                    oldbox,newbox=track['predicted_bbox'],d['bbox']
                    displacement=np.linalg.norm(np.asarray(oldbox[:2])-newbox[:2])/max(oldbox[3],newbox[3],1)
                    deep_continuation=not second and deep is not None and deep>=.9 and len(bank)>=2 and displacement<=1.5
                    if overlap < self.config['minimum_iou'] and not deep_continuation or conflict:
                        continue
                    app = cosine(track['histogram'], d['histogram']) if self.appearance and not second else None
                    if deep is not None and not second:app=deep if app is None else .5*(app+deep)
                    costs[a, b] = 1-overlap+(self.config['appearance_weight']*(1-app) if app is not None else 0)
            if costs.size:
                for a, b in zip(*linear_sum_assignment(costs)):
                    if costs[a, b] >= 1e6:
                        continue
                    track, index = tracks[a], indices[b]
                    assigned[track['entity']] = index
                    used.add(index)
        for index in high:
            if index not in used:
                key = f"M21:{camera['epoch']}:{self.next_id}"
                self.next_id += 1
                active[key] = dict(entity=key, last=frame, bbox=detections[index]['bbox'],
                                   velocity=[0, 0], histogram=None, team='UNKNOWN', team_confidence=0)
                assigned[key] = index
        out = []
        for key, index in assigned.items():
            d, track = detections[index], active[key]
            dt = max(1/self.fps, (frame-track['last'])/self.fps)
            old = np.asarray(track.get('predicted_bbox', track['bbox']))[:2]
            residual = (np.asarray(d['bbox'])[:2]-old)/dt
            velocity = .5*np.asarray(track['velocity'])+.5*residual
            track.update(last=frame, bbox=d['bbox'], velocity=velocity.tolist(), histogram=d['histogram'],
                         team=d['team'], team_confidence=d['team_confidence'])
            if self.deep and d.get('embedding') is not None:
                bank=(track.get('embedding_samples',[])+[d['embedding']])[-5:]
                track['embedding_samples']=bank;track['embedding']=np.median(bank,axis=0).tolist()
            out.append({**{k:v for k,v in d.items() if k!='embedding'}, 'entity': key, 'observed': True,
                        'embedding_observations':len(track.get('embedding_samples',[])),
                        'camera_motion': source if self.compensated else 'UNCOMPENSATED',
                        'manual_identity_propagated': False, 'experimental_identity': True})
        self.tracks = active
        return out
