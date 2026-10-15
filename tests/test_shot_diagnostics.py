import inspect
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from dataclasses import replace
from src.analytics.shot_evidence import trace_shot_frames,ShotOptions
from src.analytics.shots import detect_shots
from src.analytics.validation import validate_events
from src.evaluation.shot_diagnostics import analyze_shot_errors,write_shot_diagnostics,diagnostic_svg,report_section
from src.evaluation.shot_experiments import load_experiment
from tests.helpers import shot_fixture

class ShotDiagnosticTests(unittest.TestCase):
    def test_true_positive(self):
        rec,result,c=shot_fixture();rows=analyze_shot_errors(rec,result,c,'synthetic')
        self.assertEqual(rows[0]['failure_reason'],'MATCHED');self.assertEqual(rows[0]['BAS_event_id'],1)
    def test_false_positive_context_not_intent(self):
        rec,result,c=shot_fixture();rec.events[0].update(frame=79,seconds=79/25)
        _,result['shot_matches']=validate_events(result['shots'],rec.events,25,.1)
        rows=analyze_shot_errors(rec,result,replace(c,validation_tolerance_seconds=.1),'synthetic')
        self.assertEqual(rows[0]['match_status'],'false_positive');self.assertEqual(rows[0]['diagnostic_hypothesis'],'DIRECTION_UNCONFIRMED')
    def test_false_negative_speed_gate(self):
        rec,result,c=shot_fixture(speed=2.)
        row=analyze_shot_errors(rec,result,c,'synthetic')[0]
        self.assertEqual(row['failure_reason'],'SPEED_TOO_LOW');self.assertIn('SPEED_TOO_LOW',row['reference_frame_gates'])
    def test_missing_ball_gate_and_json(self):
        rec,result,c=shot_fixture(missing=True)
        row=analyze_shot_errors(rec,result,c,'synthetic')[0]
        self.assertEqual(row['failure_reason'],'BALL_DATA_MISSING')
        json.dumps(row,allow_nan=False)
    def test_unknown_direction_is_disclosed(self):
        rec,result,c=shot_fixture();trace=trace_shot_frames(result['possession'],result['ball'],c,ShotOptions(require_recent_possession=True))
        self.assertIsNone(trace[0]['attacking_direction']);self.assertIn('POSSESSION_DIRECTION_UNKNOWN',trace[0]['failures'])
    def test_serialization_and_inspection(self):
        rec,result,c=shot_fixture();rows=analyze_shot_errors(rec,result,c,'synthetic')
        with TemporaryDirectory() as d:
            payload=write_shot_diagnostics(rows,d)
            self.assertEqual(json.loads((Path(d)/'shot_error_analysis.json').read_text())['counts'],{'matched':1})
        self.assertIn('<svg',diagnostic_svg(rows[0]));self.assertIn('report.html?t=',report_section(payload))
    def test_baseline_configuration(self):
        spec=load_experiment('configs/shots/baseline.json');self.assertEqual(spec.analytics_overrides,{})
        rec,result,c=shot_fixture();self.assertEqual(detect_shots(result['possession'],result['ball'],c).to_dict('records'),result['shots'].to_dict('records'))
    def test_all_experiments_load(self):
        self.assertEqual(len([load_experiment(p) for p in Path('configs/shots').glob('*.json')]),11)
    def test_invalid_experiment_rejects_non_shot_override(self):
        with TemporaryDirectory() as d:
            p=Path(d)/'bad.json';p.write_text(json.dumps(dict(name='bad',description='bad',analytics_overrides={'fps':30})))
            with self.assertRaises(ValueError):load_experiment(p)
    def test_deterministic_prediction_and_no_bas_input(self):
        rec,result,c=shot_fixture();before=detect_shots(result['possession'],result['ball'],c).to_json()
        rec.events=[]
        self.assertEqual(before,detect_shots(result['possession'],result['ball'],c).to_json())
        self.assertEqual(list(inspect.signature(detect_shots).parameters),['possession','ball_motion','config','options'])
