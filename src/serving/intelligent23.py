"""Fail-closed, optional M23 evidence and an explicitly pinned M21 comparison."""
import copy
import json
import threading
from pathlib import Path
from fastapi import HTTPException
from src.serving.cv import digest
from src.vision.events.passes import signature
from src.vision.intelligent23.schema import IntelligentFrame


class IntelligentReview:
    def __init__(self, ball):
        self.ball = ball
        m = ball.manifest
        self.path = ball.directory/'intelligent_frames.jsonl'
        self.records = Path(m['perception_records_path'])
        self.offsets = m['intelligent_frame_offsets']
        if digest(self.path) != m['intelligent_sha256'] or digest(self.records) != m['perception_records_sha256']:
            raise ValueError('M23 visual provenance hash mismatch')
        if signature(m['config']) != m['config_sha256'] or m['config']['visual_threshold'] != .35:
            raise ValueError('M23 frozen visual/config gate changed')
        verification = m['config']['verification']
        if verification['m22_temporal_threshold'] != .35 or verification['m22_current_threshold'] != .35:
            raise ValueError('M22 complementary visual gates changed')
        self.models = [Path(c['path']) for c in m['model_components']]
        for c in m['model_components']:
            if digest(c['path']) != c['sha256']:
                raise ValueError('M23 component checkpoint changed')
        expected_model = signature({'components': m['model_components'], 'config': m['perception_config_sha256'], 'variant': m['config']['method']})
        if expected_model != m['detector_sha256'] or len(self.offsets) != m['frames']:
            raise ValueError('M23 model/index provenance mismatch')
        with self.path.open('rb') as stream, self.records.open() as records:
            for i, offset in enumerate(self.offsets):
                if stream.tell() != offset:
                    raise ValueError('M23 byte index mismatch')
                row = IntelligentFrame.model_validate_json(stream.readline())
                source = json.loads(records.readline())
                state = ball.frame(i)
                if row.recording_id != m['recording_id'] or row.frame != i or abs(row.timestamp-i/m['fps']) > 1e-6 or row.camera_epoch != state.camera_epoch:
                    raise ValueError('M23 diagnostic source clock mismatch')
                if row.model_set_sha256 != m['detector_sha256'] or row.config_sha256 != m['perception_config_sha256']:
                    raise ValueError('M23 diagnostic model/config mismatch')
                if source['frame'] != i or source['source_pixels_sha256'] != row.source_pixels_sha256:
                    raise ValueError('M23 source-pixel evidence mismatch')
                if row.directly_observed != state.directly_observed:
                    raise ValueError('M23 predicted/observed mismatch')
                raw = source['primary_candidates'] if row.variant == 'm21_temporal_offline' else source['methods'][row.variant]['candidates']
                lookup = {c['candidate_id']: c for c in raw}
                for c in row.candidates:
                    original = lookup[c.candidate_id]
                    if list(c.bbox) != original['bbox'] or c.confidence != original['confidence']:
                        raise ValueError('M23 candidate differs from actual inference')
                    if min(c.bbox[:2]) < 0 or c.bbox[0]+c.bbox[2] > m['width']+1e-6 or c.bbox[1]+c.bbox[3] > m['height']+1e-6:
                        raise ValueError('M23 candidate outside original pixels')
                    eligible = original.get('observation_eligible', original['confidence'] >= .35 and original['deterministic']['accepted'])
                    if c.observation_eligible != eligible:
                        raise ValueError('M23 verification decision changed')
                if row.directly_observed:
                    chosen = next(c for c in row.candidates if c.selected)
                    if chosen.bbox != state.image_bbox or chosen.confidence != state.detection_confidence:
                        raise ValueError('M23 observation differs from selected current pixels')
                for r in row.memory.search_regions if row.memory else []:
                    x, y, w, h = r.bbox
                    if min(x, y) < 0 or min(w, h) <= 0 or x+w > m['width'] or y+h > m['height']:
                        raise ValueError('M23 search region outside source')
            if stream.read(1) or records.read(1):
                raise ValueError('Unindexed M23 source evidence')
        self.paths = [self.path, self.records, *self.models]
        self.stamps = [p.stat().st_mtime_ns for p in self.paths]

    def check(self):
        if [p.stat().st_mtime_ns for p in self.paths] != self.stamps:
            raise ValueError('M23 source/model changed; restart and revalidate')

    def frame(self, index):
        self.check()
        self.ball.frame(index)
        with self.path.open('rb') as stream:
            stream.seek(self.offsets[index])
            return IntelligentFrame.model_validate_json(stream.readline())


class IntelligentRegistry:
    def __init__(self, bases, path='configs/serving/cv_m23.json'):
        self.bases = bases
        self.path = Path(path)
        self.cache = {}
        self.references = {}
        self.lock = threading.RLock()

    def entries(self):
        try:
            data = json.loads(self.path.read_text())
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def available(self, recording):
        if recording not in self.entries():return False
        try:
            self.load(recording)
            return True
        except HTTPException:
            return False

    def configured(self, recording):
        """Listing an optional mode must not validate its large sidecars eagerly.

        Selection always calls load(); configuration is never artifact readiness.
        """
        return recording in self.entries()

    def validation_state(self, recording):
        entry=self.entries().get(recording)
        if not isinstance(entry,dict):return 'not_configured'
        result=self.cache.get((recording,entry.get('report_sha256')))
        if result is None:return 'pending_validation'
        try:
            result.ball_review.intelligent.check()
            result.event_review.check()
        except (OSError,ValueError):return 'error'
        return 'validated'

    def load(self, recording):
        try:
            entry = self.entries()[recording]
            root = Path(entry['root'])
            if recording not in self.bases or digest(root/'report.json') != entry['report_sha256']:
                raise ValueError('M23 report/base mismatch')
            key = recording, entry['report_sha256']
            with self.lock:
                if key not in self.cache:
                    from .cv_events import CVEventReview
                    review = copy.copy(self.bases[recording])
                    review.event_review = None
                    review.attach_ball(root/recording/'full_m23')
                    if review.ball_review is None or review.ball_review.intelligent is None:
                        raise ValueError('Invalid M23 source evidence')
                    review.event_review = CVEventReview(root/recording/'full_m23/passes', review)
                    self.cache[key] = review
                result = self.cache[key]
                result.ball_review.intelligent.check()
                result.event_review.check()
                return result
        except (OSError, ValueError, KeyError, TypeError):
            raise HTTPException(503, 'M23 intelligent tracking unavailable or invalid; existing modes remain available') from None

    def reference(self, recording, index):
        try:
            review = self.load(recording)
            entry = self.entries()[recording]
            key = recording, entry['report_sha256']
            with self.lock:
                if key not in self.references:
                    from .ball import BallReview
                    from src.vision.intelligent23.inputs import Inputs
                    report = json.loads((Path(entry['root'])/'report.json').read_text())
                    if report.get('schema_version') == 'cv-m23-unlabelled-source-v1':
                        # A new source has its own fresh control, never an old
                        # recording's frozen reference or invented annotations.
                        path = Path(report['reference']['path'])
                        expected = Path(entry['root'])/recording/'m21_temporal_offline'
                        if path.resolve() != expected.resolve() or digest(path/'manifest.json') != report['reference']['manifest_sha256']:
                            raise ValueError('New-source M21 comparison changed')
                    else:
                        inputs = Inputs(report['inputs_root'])
                        if inputs.sha256 != report['input_manifest_sha256']:
                            raise HTTPException(503, 'Frozen M21 comparison snapshot changed')
                        path = inputs.path(Path('outputs/m21/analysis_final_v1')/recording/'temporal_offline/manifest.json').parent
                    self.references[key] = BallReview(path, review.manifest)
                return self.references[key].frame(index)
        except (OSError, ValueError, KeyError, TypeError, IndexError):
            raise HTTPException(503, 'Frozen M21 comparison unavailable or invalid') from None

    def metadata(self, recording):
        review = self.load(recording)
        entry = self.entries()[recording]
        report = json.loads((Path(entry['root'])/'report.json').read_text())
        if report.get('schema_version') == 'cv-m23-unlabelled-source-v1':
            if report['recording_id'] != recording or any(report['source'][k] != review.manifest[k] for k in ['video_sha256','frames_sha256','frames','fps','width','height']):
                raise HTTPException(503, 'New-source report does not match this recording')
            return {'recording_id':recording,'reference_role':'unlabelled_inference',
                'independent_final_test':report['independent_final_test'],'promotion':report['promotion'],
                'metrics':report['metrics'],'comparisons':[],'frozen_baselines':[],
                'aggregate':None,'accuracy':None,'runtime':report['runtime'],'research_targets':[],
                'model_set_sha256':review.ball_review.manifest['detector_sha256'],
                'ball_revision':review.ball_review.manifest['ball_frames_sha256'],
                'model_components':review.ball_review.manifest['model_components']}
        current = report['variants']['full_m23']
        return {'recording_id': recording, 'reference_role': 'development_only',
            'independent_final_test': report['independent_final_test'], 'promotion': report['promotion'],
            'metrics': current['per_recording'][recording], 'aggregate': {'new_150': current['new_150'], 'original_references': current['original_references']},
            'comparisons': [{'id': k, 'new_150': v['per_recording'][recording]['new_150']['ball'],
                'original': v['per_recording'][recording]['original_references']['ball'],
                'passes': v['per_recording'][recording]['original_references']['passes']} for k, v in report['variants'].items()],
            'frozen_baselines': [{'id': k, 'new_150': v['per_recording'][recording]['new_150']['ball']} for k, v in report['baselines'].items()],
            'runtime': report['perception']['clips'][recording], 'research_targets': report['research_targets'],
            'model_set_sha256': review.ball_review.manifest['detector_sha256'],
            'ball_revision': review.ball_review.manifest['ball_frames_sha256'],
            'model_components': review.ball_review.manifest['model_components']}
