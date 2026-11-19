"""Strict M21 sidecars, with optional variants validated lazily and independently."""
import copy
import json
from pathlib import Path
import threading
from fastapi import HTTPException
from src.serving.cv import digest
from src.vision.advanced21.schema import AdvancedFrame
from src.vision.events.passes import signature


class AdvancedReview:
    def __init__(self, ball):
        self.ball = ball
        self.manifest = ball.manifest
        self.path = ball.directory/'advanced_frames.jsonl'
        self.offsets = self.manifest['advanced_frame_offsets']
        if signature(self.manifest['config']) != self.manifest['config_sha256'] or digest(self.path) != self.manifest['advanced_sha256']:
            raise ValueError('M21 diagnostics/config hash mismatch')
        if len(self.offsets) != self.manifest['frames']:
            raise ValueError('M21 diagnostic frame count mismatch')
        with self.path.open('rb') as source:
            for i, offset in enumerate(self.offsets):
                if offset != source.tell():
                    raise ValueError('M21 diagnostic byte index mismatch')
                row = AdvancedFrame.model_validate_json(source.readline())
                state = ball.frame(i)
                if row.frame != i or abs(row.timestamp-state.timestamp) > 1e-8 or row.camera_epoch != state.camera_epoch or row.variant != self.manifest['config']['variant']:
                    raise ValueError('M21 diagnostic clock/version mismatch')
                if row.directly_observed != state.directly_observed:
                    raise ValueError('M21 observed/inferred mismatch')
                if state.directly_observed:
                    selected = next(c for c in row.candidates if c.selected)
                    if selected.bbox != state.image_bbox or selected.visual_confidence != state.detection_confidence or selected.visual_confidence < .35:
                        raise ValueError('M21 selected pixels/confidence differ from visual evidence')
                    if 'PROPOSAL_NOT_OBSERVED' in selected.inference_view:
                        raise ValueError('Fusion-only proposal cannot become observed')
                for c in row.candidates:
                    x,y,w,h = c.bbox
                    if min(x,y) < 0 or min(w,h) <= 0 or x+w > self.manifest['width']+1e-6 or y+h > self.manifest['height']+1e-6:
                        raise ValueError('M21 candidate outside source image')
                if row.reconstruction is not None:
                    a,b = row.reconstruction.support_frames
                    if not 0 <= a < i < b < self.manifest['frames'] or state.image_position is not None:
                        raise ValueError('Invalid future reconstruction provenance')
            if source.read(1):
                raise ValueError('Unindexed M21 diagnostics')
        self.stamp = self.path.stat().st_mtime_ns

    def check(self):
        if self.path.stat().st_mtime_ns != self.stamp:
            raise HTTPException(503,'M21 evidence changed; revalidate before use')

    def frame(self, index):
        self.check(); self.ball.frame(index)
        with self.path.open('rb') as source:
            source.seek(self.offsets[index])
            return AdvancedFrame.model_validate_json(source.readline())


class AdvancedRegistry:
    def __init__(self, bases, path='configs/serving/cv_m21.json'):
        self.bases = bases
        self.entries = {}
        self.cache = {}
        self.lock = threading.RLock()
        try:
            document = json.loads(Path(path).read_text())
            if isinstance(document,dict): self.entries = document
        except (OSError,ValueError):
            pass

    def load(self, recording, experiment=None):
        entry = self.entries.get(recording)
        if recording not in self.bases or not isinstance(entry,dict):
            raise HTTPException(503,'M21 experiment unavailable')
        try:
            if not all(isinstance(entry.get(key),str) for key in ['root','report_sha256','default_experiment']):
                raise ValueError('Invalid M21 registry entry')
            root = Path(entry['root'])
            if digest(root/'report.json') != entry['report_sha256']:
                raise ValueError('M21 experiment report changed')
            report = json.loads((root/'report.json').read_text())
            variant = experiment or entry['default_experiment']
            if variant not in report['variants']:
                raise HTTPException(422,'Choose a registered M21 experiment')
            key = (recording,variant)
            with self.lock:
                if key not in self.cache:
                    from src.serving.cv_events import CVEventReview
                    review = copy.copy(self.bases[recording])
                    review.attach_ball(root/recording/variant)
                    if review.ball_review is None or review.ball_review.manifest.get('version') != 'm21':
                        raise ValueError('Invalid M21 observed/diagnostic artifact')
                    review.event_review = CVEventReview(root/recording/variant/'passes',review)
                    review.m21_experiment = variant
                    self.cache[key] = review
                result = self.cache[key]
                result.ball_review.advanced.check()
                result.event_review.check()
                return result
        except HTTPException:
            raise
        except (OSError,ValueError,KeyError,TypeError):
            raise HTTPException(503,'M21 experiment unavailable or invalid; frozen modes remain available') from None

    def metadata(self, recording, experiment=None):
        review = self.load(recording,experiment)
        root = Path(self.entries[recording]['root'])
        report = json.loads((root/'report.json').read_text())
        return {'recording_id':recording,'selected_experiment':review.m21_experiment,
                'default_experiment':self.entries[recording]['default_experiment'],
                'reference_role':'development_only','final_test_status':report['final_test_status'],
                'experiments':[{'id':key,'metrics':value['per_recording'][recording],
                                'aggregate':value['aggregate']} for key,value in report['variants'].items()],
                'camera':report['clips'][recording]['camera'],'physics':report['clips'][recording]['physics'],
                'selection':report['selection']}
