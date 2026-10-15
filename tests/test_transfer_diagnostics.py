import unittest
from tempfile import TemporaryDirectory
import pandas as pd
import numpy as np
from src.analytics.motion import calculate_motion
from src.analytics.passes import detect_passes
from src.analytics.validation import validate_events
from src.evaluation.transfer_diagnostics import analyze_transfer_errors,write_transfer_diagnostics
from src.evaluation.event_evaluation import evaluate_transfers
from tests.helpers import tactical_fixture
from src.analytics.config import AnalyticsConfig

class TransferDiagnosticTests(unittest.TestCase):
    def fixture(self):
        rec,result=tactical_fixture(seconds=2);c=AnalyticsConfig()
        result['ball']=calculate_motion(np.tile([20.,34.],(50,1)),25,3,3,50)
        result['passes']=detect_passes(result['possession'],result['ball'],c)
        rec.events=[dict(annotation_id=1,frame=25,seconds=1.,label='THROW IN',player_id=999)]
        _,result['pass_matches']=validate_events(result['passes'],rec.events,25,actor_field='sender_id')
        return rec,result,c
    def test_missed_reference_classification(self):
        rec,result,c=self.fixture();rows,metrics=analyze_transfer_errors(rec,result,c,'synthetic')
        self.assertEqual(rows[0]['failure_reason'],'POSSESSION_NEVER_ACQUIRED')
        self.assertEqual(rows[0]['BAS_class'],'THROW IN')
    def test_per_class_breakdown_serialization(self):
        rec,result,c=self.fixture();rows,metrics=analyze_transfer_errors(rec,result,c,'synthetic')
        with TemporaryDirectory() as d:
            p=write_transfer_diagnostics(rows,[metrics],d)
        self.assertEqual(p['by_bas_class']['THROW IN']['false_negatives'],1)
        self.assertIsNone(p['by_bas_class']['PASS']['precision'])
    def test_unmatched_candidate_classification(self):
        rec,result,c=self.fixture()
        predicted=pd.DataFrame([dict(transfer_id=1,start_frame=1,sender_id=101,outcome='unresolved')])
        rec.events[0].update(frame=49,seconds=49/25)
        result['passes']=predicted
        _,result['pass_matches']=validate_events(predicted,rec.events,25,.1,actor_field='sender_id')
        from unittest.mock import patch
        from dataclasses import replace
        with patch('src.evaluation.transfer_diagnostics.detect_passes',return_value=predicted):
            rows,_=analyze_transfer_errors(rec,result,replace(c,validation_tolerance_seconds=.1),'synthetic')
        self.assertEqual(rows[0]['failure_reason'],'ANNOTATION_TIMING_OR_SEMANTICS_UNSUPPORTED')
    def test_audit_does_not_change_detections(self):
        rec,result,c=self.fixture();audit=[]
        self.assertEqual(detect_passes(result['possession'],result['ball'],c,audit).to_json(),result['passes'].to_json())
