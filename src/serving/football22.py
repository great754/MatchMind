"""Lazy optional M22 loader. A corrupt experiment never removes a legacy recording."""
import copy
import json
from pathlib import Path
import threading
from fastapi import HTTPException
from src.serving.cv import digest
from src.vision.football22.schema import FootballFrame
from src.vision.events.passes import signature


class FootballReview:
    def __init__(self,ball):
        self.ball=ball;self.path=ball.directory/'football_frames.jsonl';m=ball.manifest;self.offsets=m['football_frame_offsets']
        if digest(self.path)!=m['football_sha256'] or signature(m['config'])!=m['config_sha256']:
            raise ValueError('M22 source diagnostics/config mismatch')
        if len(self.offsets)!=m['frames']:raise ValueError('M22 diagnostic count mismatch')
        if m['config']['observation_threshold']!=.35 or m['config']['current_only_verification_threshold']!=.35:
            raise ValueError('M22 current-pixel observation gates changed')
        checkpoint=m['trained_checkpoint']
        if digest(checkpoint)!=m['detector_sha256']:raise ValueError('M22 trained checkpoint changed')
        with self.path.open('rb') as source:
            for i,offset in enumerate(self.offsets):
                if source.tell()!=offset:raise ValueError('M22 diagnostic byte index mismatch')
                row=FootballFrame.model_validate_json(source.readline());state=ball.frame(i)
                if row.recording_id!=m['recording_id'] or row.frame!=i or abs(row.timestamp-state.timestamp)>1e-8 or row.camera_epoch!=state.camera_epoch or row.checkpoint_sha256!=m['detector_sha256']:
                    raise ValueError('M22 source/clock/checkpoint mismatch')
                if row.directly_observed!=state.directly_observed:raise ValueError('M22 observed/inferred mismatch')
                if row.directly_observed:
                    selected=next(c for c in row.candidates if c.selected)
                    if selected.bbox!=state.image_bbox or selected.confidence!=state.detection_confidence:raise ValueError('M22 selected visual evidence differs from observation')
                for c in row.candidates:
                    x,y,w,h=c.bbox
                    if x+w>m['width']+1e-6 or y+h>m['height']+1e-6:raise ValueError('M22 candidate outside source')
                if row.reconstruction is not None:
                    a,b=row.reconstruction['support_frames']
                    if not 0<=a<i<b<m['frames'] or state.image_position is not None:raise ValueError('M22 inferred reconstruction provenance invalid')
            if source.read(1):raise ValueError('Unindexed M22 diagnostics')
        self.stamp=self.path.stat().st_mtime_ns;self.checkpoint=Path(checkpoint);self.checkpoint_stamp=self.checkpoint.stat().st_mtime_ns

    def check(self):
        if self.path.stat().st_mtime_ns!=self.stamp or self.checkpoint.stat().st_mtime_ns!=self.checkpoint_stamp:raise ValueError('M22 evidence/checkpoint changed; revalidate before querying')

    def frame(self,index):
        self.check();self.ball.frame(index)
        with self.path.open('rb') as stream:stream.seek(self.offsets[index]);return FootballFrame.model_validate_json(stream.readline())


class FootballRegistry:
    def __init__(self,bases,path='configs/serving/cv_m22.json'):
        self.bases=bases;self.path=Path(path);self.cache={};self.lock=threading.RLock()

    def entries(self):
        try:
            data=json.loads(self.path.read_text());return data if isinstance(data,dict) else {}
        except (OSError,ValueError):return {}

    def configured(self,recording):return recording in self.entries()

    def available(self,recording):
        if not self.configured(recording):return False
        try:self.load(recording);return True
        except HTTPException:return False

    def load(self,recording):
        try:
            entry=self.entries().get(recording)
            if recording not in self.bases or not isinstance(entry,dict):raise ValueError('Unconfigured M22 source')
            root=Path(entry['root']);report=root/'report.json'
            if digest(report)!=entry['report_sha256']:raise ValueError('M22 report changed')
            key=(recording,entry['report_sha256'])
            with self.lock:
                if key not in self.cache:
                    from src.serving.cv_events import CVEventReview
                    review=copy.copy(self.bases[recording]);review.event_review=None;review.attach_ball(root/recording)
                    if review.ball_review is None or review.ball_review.manifest['version']!='m22':raise ValueError('Invalid M22 ball evidence')
                    review.event_review=CVEventReview(root/recording/'passes',review);self.cache[key]=review
                result=self.cache[key];result.ball_review.football.check();result.event_review.check()
                return result
        except (OSError,ValueError,KeyError,TypeError):raise HTTPException(503,'M22 temporal detector unavailable or invalid; existing modes remain available') from None

    def metadata(self,recording):
        review=self.load(recording);entry=self.entries()[recording];report=json.loads((Path(entry['root'])/'report.json').read_text())
        return {'recording_id':recording,'model':report['perception']['clips'][recording]['model'],
            'metrics':report['per_recording'][recording],'aggregate':report['aggregate'],'baselines':report['baselines'],
            'detector_only':report.get('detector_only_per_recording',{}).get(recording),
            'reference_role':'development_only','independent_test_status':'not_available','promotion':report['promotion'],
            'runtime':report['perception']['clips'][recording],'training':report['training'][recording],
            'checkpoint_sha256':review.ball_review.manifest['detector_sha256'],'review_tasks':50,
            'pass_trace':report['pass_traces'][recording]}
