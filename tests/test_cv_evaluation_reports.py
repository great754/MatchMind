"""Report/output regressions with isolated synthetic inputs, not accuracy evidence."""
import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts import evaluate_cv_m19, evaluate_cv_m20


class EvaluationReportTests(unittest.TestCase):
    def test_explicit_existing_output_is_preserved_before_loading_artifacts(self):
        for module in [evaluate_cv_m19, evaluate_cv_m20]:
            with self.subTest(module=module.__name__), TemporaryDirectory() as directory:
                target = Path(directory) / 'results.json'
                target.write_bytes(b'previous measurement\n')
                with patch.object(module, 'HumanReviewStore') as store:
                    with self.assertRaisesRegex(ValueError, 'fresh evaluation output'):
                        module.run('unused', output=target)
                    store.assert_not_called()
                self.assertEqual(target.read_bytes(), b'previous measurement\n')

    def report(self, directory, labelled):
        task = {'task_id': 'ball', 'recording_id': 'sample', 'kind': 'ball', 'frame': 0}
        source = {'video_sha256': 'synthetic-source', 'fps': 25}
        dataset = {'fingerprint': 'synthetic-dataset', 'tasks': [task],
                   'sources': {'sample': source},
                   'protocol': {'ball_box_match_iou': .25, 'pass_endpoint_tolerance_seconds': .25}}
        row = {'task_id': 'ball', 'reviewer_id': 'test-reviewer', 'origin': 'human',
               'reviewed_from_source': True, 'model_assisted': False, 'state': 'visible',
               'bbox': [0, 0, 4, 4], 'source_video_sha256': 'synthetic-source'}
        reference = Path(directory) / 'private.json'
        reference.write_text(json.dumps({'dataset_fingerprint': dataset['fingerprint'], 'annotations': [row]}))
        original = reference.read_bytes()
        target = Path(directory) / 'nested' / 'results.json'
        store = Mock(dataset=dataset, tasks={'ball': task})
        store.labels.return_value = [row]
        frame = SimpleNamespace(model_dump=lambda: {'directly_observed': True,
                                'image_bbox': [0, 0, 4, 4], 'image_position': [2, 2]})
        ball = Mock()
        ball.frame.return_value = frame
        with patch.object(evaluate_cv_m20, 'HumanReviewStore', return_value=store), \
             patch.object(evaluate_cv_m20, 'CVReview', return_value=SimpleNamespace(manifest={'frames': 1}, data='unused')), \
             patch.object(evaluate_cv_m20, 'BallReview', return_value=ball), \
             patch.object(evaluate_cv_m20, 'CVEventReview', return_value=SimpleNamespace(events=[])), \
             patch.object(evaluate_cv_m20, 'compact_worlds', return_value=[]), \
             contextlib.redirect_stdout(io.StringIO()):
            report = evaluate_cv_m20.run('unused', reference if labelled else None, output=target)
        self.assertEqual(json.loads(target.read_text()), report)
        self.assertEqual(reference.read_bytes(), original)
        store.save.assert_not_called()
        return report

    def test_completed_review_reports_sample_metrics_without_claiming_acceptance(self):
        with TemporaryDirectory() as directory:
            report = self.report(directory, labelled=True)
        self.assertEqual(report['independent_evaluation_status'], 'human_sample_evaluation_available')
        self.assertEqual(report['independent_accuracy_acceptance'], 'not_established_from_this_sample')
        self.assertTrue(report['reference_review']['all_tasks_reviewed'])
        for result in report['variants'].values():
            self.assertEqual(result['per_recording']['sample']['ball'], result['ball'])
            self.assertEqual(result['ball']['tp'], 1)
            self.assertIsNone(result['independent_reacquisition_success'])

    def test_no_labels_still_reports_pending_and_undefined_metrics(self):
        with TemporaryDirectory() as directory:
            report = self.report(directory, labelled=False)
        self.assertEqual(report['independent_accuracy_acceptance'], 'pending_human_review')
        self.assertFalse(report['reference_review']['all_tasks_reviewed'])
        self.assertEqual(report['annotations_used'], 0)
        self.assertIsNone(report['variants']['m20']['ball']['precision'])


if __name__ == '__main__':
    unittest.main()
