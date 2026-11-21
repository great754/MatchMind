import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from scripts.prepare_cv_m23_independent import prepare


class IndependentSourceWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.video = self.root/'approved.mp4'
        self.video.write_bytes(b'not decoded in freeze stage')
        self.sha = hashlib.sha256(self.video.read_bytes()).hexdigest()
        self.approval = self.root/'approval.json'
        self.approval.write_text(json.dumps({'explicit_user_approval': True,
            'video_sha256': self.sha, 'original_match_id': 'new-independent-match',
            'approval_reference': 'Explicit user request identifying this source'}))
        self.analysis = self.root/'report.json'
        self.analysis.write_text(json.dumps({'perception': {'clips': {'old': {'source_video_sha256': '0'*64}},
            'config_sha256': 'a'*64, 'source_sha256': {'detector.py': 'b'*64}}}))
        self.review = self.root/'review.json'
        self.audit = {'video_sha256': self.sha, 'original_match_id': 'new-independent-match',
            'football_content_reviewed': True, 'football_players_and_ball_present': True,
            'original_match_overlap_audited': True,
            'original_match_overlaps_training_or_development': False,
            'reviewer': 'test reviewer', 'overlap_audit_evidence': 'test inventory comparison',
            'training_source_inventory_sha256': 'c'*64}
        self.review.write_text(json.dumps(self.audit))

    def run_freeze(self):
        return prepare(self.video, self.approval, self.root/'freeze', self.analysis, self.review)

    def test_approval_must_identify_exact_source(self):
        approval=json.loads(self.approval.read_text());approval['video_sha256']='d'*64
        self.approval.write_text(json.dumps(approval))
        with self.assertRaisesRegex(ValueError,'exact source'):self.run_freeze()

    def test_development_source_cannot_be_independent(self):
        report=json.loads(self.analysis.read_text());report['perception']['clips']['old']['source_video_sha256']=self.sha
        self.analysis.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError,'development source'):self.run_freeze()

    def test_unknown_match_overlap_blocks_freeze(self):
        self.audit['original_match_overlaps_training_or_development']=None
        self.review.write_text(json.dumps(self.audit))
        with self.assertRaisesRegex(ValueError,'overlap'):self.run_freeze()

    def test_preview_and_inventory_review_are_required(self):
        for key in ['football_content_reviewed','football_players_and_ball_present','original_match_overlap_audited','training_source_inventory_sha256']:
            with self.subTest(key=key):
                audit=self.audit|{key:False};self.review.write_text(json.dumps(audit))
                with self.assertRaises(ValueError):self.run_freeze()

    def test_protocol_freeze_does_not_claim_model_or_labels_are_ready(self):
        result=self.run_freeze()
        self.assertFalse(result['labels_reviewed'])
        self.assertFalse(result['promotion_allowed'])
        self.assertEqual(result['matching_iou'],.25)
        self.assertEqual(result['status'],'source_and_protocol_frozen_model_selection_required')
        self.assertEqual(result['analysis_report_sha256'],hashlib.sha256(self.analysis.read_bytes()).hexdigest())

    def test_freeze_is_immutable(self):
        self.run_freeze()
        with self.assertRaisesRegex(ValueError,'immutable'):self.run_freeze()


if __name__ == '__main__':unittest.main()
