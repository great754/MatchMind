import unittest
from src.evaluation.artifact_quality import guard_intervals
from src.evaluation.aggregate import aggregate_events
from src.evaluation.event_evaluation import evaluate_events

class ArtifactQualityTests(unittest.TestCase):
    def test_overlap_and_duplicates(self):
        a=dict(recording_id='a',match_id='m',half=1,source_offset_frames=0,frames=100)
        with self.assertRaises(ValueError):guard_intervals([a,dict(a,recording_id='b',source_offset_frames=99)])
        with self.assertRaises(ValueError):guard_intervals([a,a])
        guard_intervals([a,dict(a,recording_id='b',source_offset_frames=100)])
        guard_intervals([a,dict(a,recording_id='c',half=2)])
    def test_micro_counts_and_zero_reference(self):
        good=evaluate_events([{'start_frame':10}],[{'annotation_id':1,'frame':10,'label':'SHOT','player_id':None}],25)
        missing=evaluate_events([{'start_frame':10}],[],25)
        result=aggregate_events([good,missing])
        self.assertEqual(result['precision'],1);self.assertEqual(result['no_reference_recordings'],1)
        self.assertIsNone(missing['precision']);self.assertFalse(result['independent_validation'])

    def test_frozen_export_drift_is_unavailable(self):
        import tempfile
        from pathlib import Path
        from src.evaluation.artifact_quality import frozen_exports
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);bundle=root/'bundle';exports=root/'exports';bundle.mkdir();exports.mkdir()
            for name in ('summary.json','events.json'):
                (bundle/name).write_text('{}');(exports/name).write_text('{}')
            self.assertEqual(frozen_exports(bundle,exports),exports)
            (exports/'events.json').write_text('{"changed":true}')
            self.assertIsNone(frozen_exports(bundle,exports))
    def test_shot_context_is_hypothesis_not_detector_change(self):
        import tempfile,csv
        from pathlib import Path
        from src.evaluation.shot_diagnostics import diagnose_exported_shots
        with tempfile.TemporaryDirectory() as root:
            root=Path(root)
            (root/'shots.csv').write_text('start_frame,end_frame,shooter_id,team\n10,20,,\n')
            (root/'bas_shot_windows.csv').write_text('annotation_id,window_coverage_fraction\n2,0\n')
            (root/'ball_frames.csv').write_text('frame,position_source\n10,event_interpolated\n')
            refs=[dict(annotation_id=1,frame=10,label='CROSS'),dict(annotation_id=2,frame=100,label='SHOT')]
            evaluation={'matching':[dict(match_status='false_positive',prediction_index=0,prediction_frame=10,annotation_id=None),dict(match_status='false_negative',prediction_index=None,annotation_id=2,annotation_frame=100)]}
            rows=diagnose_exported_shots(root,evaluation,refs,25)
            self.assertIn('CROSS_CONTEXT',rows[0]['hypotheses'])
            self.assertIn('MISSING_USABLE_BALL_WINDOW',rows[1]['hypotheses'])
            self.assertTrue(all('No production thresholds' in r['limitation'] for r in rows))
