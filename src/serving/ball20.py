"""Validate M20 debugging separately from current observed ball coordinates."""
import json
from fastapi import HTTPException
from src.vision.ball20.schema import TrajectoryFrame
from src.vision.events.passes import signature
from .cv import digest


class TrajectoryReview:
    def __init__(self, ball_review):
        self.ball_review = ball_review
        self.manifest = ball_review.manifest
        self.path = ball_review.directory.parent/'trajectory_frames.jsonl'
        self.offsets = self.manifest['trajectory_frame_offsets']
        if signature(self.manifest['config']) != self.manifest['config_sha256'] or digest(self.path) != self.manifest['trajectory_sha256']:
            raise ValueError('M20 trajectory/config hash mismatch')
        if len(self.offsets) != self.manifest['frames']:
            raise ValueError('M20 trajectory frame count mismatch')
        with self.path.open('rb') as source:
            for i, offset in enumerate(self.offsets):
                if source.tell() != offset: raise ValueError('M20 trajectory index mismatch')
                row = TrajectoryFrame.model_validate_json(source.readline())
                ball = ball_review.frame(i)
                if row.frame != i or abs(row.timestamp-ball.timestamp) > 1e-8 or row.camera_epoch != ball.camera_epoch:
                    raise ValueError('M20 trajectory/ball clock mismatch')
                expected = 'OBSERVED' if ball.directly_observed else 'AMBIGUOUS' if ball.tracking_status == 'AMBIGUOUS' else 'PREDICTED' if ball.predicted_image_position is not None else 'UNKNOWN'
                if row.state != expected: raise ValueError('M20 observed/predicted state mismatch')
                candidates = {c.candidate_id: c for c in ball.candidates}
                if set(candidates) != {c.candidate_id for c in row.candidates} or len(row.candidates) != len(candidates):
                    raise ValueError('M20 decisions must refer to actual visual candidates')
                for candidate in row.candidates:
                    if candidate.visual_confidence != candidates[candidate.candidate_id].confidence:
                        raise ValueError('M20 visual confidence changed by trajectory scoring')
                if ball.directly_observed:
                    selected = candidates.get(row.selected_candidate_id)
                    if selected is None or selected.bbox != ball.image_bbox or selected.center != ball.image_position:
                        raise ValueError('M20 selected observation differs from visual evidence')
                for region in row.search_regions:
                    x, y, width, height = region.bbox
                    if x+width > self.manifest['width'] or y+height > self.manifest['height']:
                        raise ValueError('M20 crop is outside source bounds')
            if source.read(1): raise ValueError('Unindexed M20 trajectory data')
        self.stamp = self.path.stat().st_mtime_ns

    def frame(self, index):
        if self.path.stat().st_mtime_ns != self.stamp:
            raise HTTPException(503, 'M20 trajectory changed; restart validation')
        self.ball_review.frame(index)
        with self.path.open('rb') as source:
            source.seek(self.offsets[index]); return TrajectoryFrame.model_validate_json(source.readline())
