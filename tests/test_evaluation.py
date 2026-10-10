import unittest
import json
from pathlib import Path
import pandas as pd
from src.evaluation.event_evaluation import evaluate_events,evaluate_transfers
from src.evaluation.aggregate import aggregate_events
from src.evaluation.possession_evaluation import evaluate_possession
from src.vision.data_loading import DatasetPaths
from src.analytics.data import load_recording
from src.evaluation.__main__ import run
from unittest.mock import patch
from types import SimpleNamespace
from tempfile import TemporaryDirectory


class EvaluationTests(unittest.TestCase):
    def test_known_assignment(self):
        p=[dict(start_frame=26,sender_id=7),dict(start_frame=200,sender_id=8)]
        refs=[dict(frame=25,player_id=7,annotation_id=1,label='PASS'),dict(frame=300,player_id=8,annotation_id=2,label='CROSS')]
        r=evaluate_transfers(p,refs,25)
        self.assertEqual((r['matched'],r['false_positives'],r['false_negatives']),(1,1,1))
        self.assertEqual((r['precision'],r['recall'],r['f1']),(.5,.5,.5))
        self.assertAlmostEqual(r['timing_error_seconds']['p95'],.04)
        self.assertEqual(r['by_bas_class']['PASS']['matched'],1)
        self.assertEqual(r['by_bas_class']['CROSS']['recall'],0)
        self.assertIsNone(r['by_bas_class']['PASS']['precision'])

    def test_sender_must_agree(self):
        r=evaluate_transfers([dict(start_frame=25,sender_id=8)],[dict(frame=25,player_id=7,annotation_id=1,label='PASS')],25)
        self.assertEqual(r['matched'],0)

    def test_no_reference(self):
        r=evaluate_events([dict(start_frame=1)],[],25)
        self.assertIsNone(r['precision']);self.assertIsNone(r['recall']);self.assertIsNone(r['false_positives'])
        self.assertIsNone(aggregate_events([r])['f1'])

    def test_empty_predictions(self):
        r=evaluate_events([],[dict(frame=25,player_id=None,annotation_id=1,label='SHOT')],25)
        self.assertEqual(r['false_negatives'],1);self.assertEqual(r['f1'],0)

    def test_aggregate(self):
        r=evaluate_events([dict(start_frame=25)],[dict(frame=25,player_id=None,annotation_id=1,label='SHOT')],25)
        a=aggregate_events([r,evaluate_events([],[],25)])
        self.assertEqual(a['f1'],1);self.assertEqual(a['no_reference_recordings'],1)

    def test_possession_sanity(self):
        table=pd.DataFrame(dict(possessing_player_id=[1,1,None,2],state=['controlled','controlled','unknown','controlled']))
        r=evaluate_possession(table,[dict(frame=2,annotation_id=1,label='PASS',player_id=1)],25)
        self.assertEqual(r['state_percent']['controlled'],75)
        self.assertEqual(r['possession_changes_including_gaps'],1)
        self.assertEqual(r['bas_pre_event_checks'][0]['sample_frame'],1)
        self.assertTrue(r['bas_pre_event_checks'][0]['sender_agreement'])
        self.assertIsNone(r['precision'])

    def test_paths(self):
        for match in ('118575','999'):
            p=DatasetPaths(match,Path('/tmp/data'))
            self.assertEqual(p.bas,Path(f'/tmp/data/bas/{match}/{match}_12_class_events.json'))
            self.assertEqual(p.sync,Path(f'/tmp/data/mot/{match}_sync.json'))
            self.assertIn(match,str(p.gsr_compact(2)));self.assertIn('_2nd_',str(p.ball(2)))
        with self.assertRaises(ValueError):p.ball(3)

    def test_reject_duplicate_and_overlap(self):
        fake=SimpleNamespace(half=1,source_offset=0,ball=[0]*10)
        with TemporaryDirectory() as dest,patch('src.evaluation.__main__.load_recording',return_value=fake):
            with self.assertRaises(ValueError):run({'recordings':[dict(match_id='1'),dict(match_id='1')]},dest)
            with self.assertRaises(ValueError):run({'recordings':[dict(match_id='1',scope='clip'),dict(match_id='1',scope='half')]},dest)

    def test_clip_half_mismatch(self):
        with patch('pathlib.Path.read_text',side_effect=[json.dumps(dict(match_id='118575',half=1,fps=25)),json.dumps(dict(fps=25,clock='first frame'))]):
            with self.assertRaisesRegex(ValueError,'does not contain'):
                load_recording(DatasetPaths(),half=2)

    def test_sensitivity_preserves_baseline(self):
        from src.evaluation.tactical_evaluation import sensitivity
        from src.analytics.config import AnalyticsConfig
        from src.tactics.config import TacticsConfig
        from test_tactics import tactical_fixture
        recording,result=tactical_fixture(seconds=1)
        with patch('src.intelligence.events.build_events',return_value=[]),patch('src.evaluation.tactical_evaluation.VARIANTS',{'pressure_local_radius_m':(4.,6.)}):
            measured=sensitivity(recording,result,AnalyticsConfig(),TacticsConfig())
        self.assertEqual(measured['baseline_config']['pressure_local_radius_m'],5.)
        self.assertEqual([v['value'] for v in measured['variants']],[4.,6.])
        self.assertTrue(all(v['measurements']['samples']==5 for v in measured['variants']))

    def test_multi_match_output_isolation(self):
        records=[SimpleNamespace(half=1,source_offset=0,ball=[0]*10,fps=25.,scope='half',sync={'match_id':match},events=[]) for match in ('a','b')]
        result=dict(passes=[],shots=[],possession=pd.DataFrame(dict(possessing_player_id=[None]*10,state=['unknown']*10)))
        destinations=[]
        def write(recording,result,config,dest,*args,**kwargs):
            destinations.append(dest);dest.mkdir(parents=True)
            return {'source_files':[],'limitations':[]}
        with TemporaryDirectory() as dest,patch('src.evaluation.__main__.load_recording',side_effect=records) as loader,patch('src.evaluation.__main__.analyze_recording',return_value=result),patch('src.evaluation.__main__.write_outputs',side_effect=write),patch('src.evaluation.__main__.sensitivity',return_value={}):
            report=run({'recordings':[dict(match_id='a',scope='half'),dict(match_id='b',scope='half')]},dest)
            self.assertEqual([call.args[0].match_id for call in loader.call_args_list],['a','b'])
            self.assertNotEqual(destinations[0],destinations[1])
            self.assertTrue(all((p/'evaluation.json').exists() for p in destinations))
            self.assertEqual(report['aggregate']['shots']['recordings'],2)
            self.assertIsNone(report['aggregate']['shots']['precision'])
