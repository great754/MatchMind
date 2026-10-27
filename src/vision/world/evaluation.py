"""Optional hash/clock-bound reference diagnostics; unavailable never becomes zero error."""
import hashlib,json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from .tracking import iou

class ReferenceEvaluator:
    def __init__(self,video,fps,reference=None):
        self.rows={};self.frames=0;self.detections=0;self.references=0;self.matches=0;self.errors=[];self.previous={};self.switches=0;self.identity_samples=0
        self.status='unavailable: no matching reference supplied'
        if reference:
            data=json.loads(Path(reference).read_text())
            if data['video_sha256']!=hashlib.sha256(Path(video).read_bytes()).hexdigest() or abs(data['fps']-fps)>1e-6:raise ValueError('Reference video hash/clock mismatch')
            if len({r['frame'] for r in data['frames']})!=len(data['frames']):raise ValueError('Duplicate reference frame')
            self.rows={r['frame']:r['players'] for r in data['frames']};self.status='reference-supported; supplied mapping, not independent ground truth'
    def update(self,frame,tracks):
        if frame not in self.rows:return
        refs=self.rows[frame];self.frames+=1;self.detections+=len(tracks);self.references+=len(refs)
        cost=np.ones((len(tracks),len(refs)))
        for i,t in enumerate(tracks):
            for j,r in enumerate(refs):cost[i,j]=1-iou(t.bbox,r['bbox'])
        if not len(tracks) or not len(refs):return
        for i,j in zip(*linear_sum_assignment(cost)):
            if cost[i,j]>.5:continue
            self.matches+=1;r=refs[j];t=tracks[i]
            if r.get('actor_id') is not None:
                old=self.previous.get(str(r['actor_id']))
                if old and old[0]==frame-1 and old[1]!=t.track_id:self.switches+=1
                self.previous[str(r['actor_id'])]=(frame,t.track_id);self.identity_samples+=1
            if t.pitch is not None and r.get('pitch') is not None:self.errors.append(float(np.linalg.norm(np.array(t.pitch)-r['pitch'])))
    def report(self):
        available=self.frames>0
        return dict(status=self.status,reference_frames=self.frames,
            matched_person_boxes=self.matches if available else None,missed_reference_boxes=self.references-self.matches if available else None,
            extra_person_boxes=self.detections-self.matches if available else None,reference_box_coverage=self.matches/self.references if self.references else None,
            consecutive_reference_track_switches=self.switches if self.identity_samples else None,
            localization_samples=len(self.errors),median_pitch_error_m=float(np.median(self.errors)) if self.errors else None,
            p95_pitch_error_m=float(np.percentile(self.errors,95)) if self.errors else None,
            independent_accuracy=False,limitation='IoU >= .5 one-to-one matches on explicit reference frames. Person boxes can include officials/spectators. No ID metrics without reference identities; no metric errors without matched calibrated positions.')
