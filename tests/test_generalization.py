import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch
from types import SimpleNamespace
from src.evaluation.__main__ import run

class GeneralizationTests(unittest.TestCase):
    def test_known_development_match_cannot_be_held_out(self):
        with TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'Known development'):
                run({'recordings':[dict(match_id='118575',split='evaluation')]},d)
    def test_match_cannot_cross_splits(self):
        fake=SimpleNamespace(half=1,source_offset=0,ball=[0]*10)
        with TemporaryDirectory() as d,patch('src.evaluation.__main__.load_recording',return_value=fake):
            with self.assertRaisesRegex(ValueError,'both development'):
                run({'recordings':[dict(match_id='other',split='development'),dict(match_id='other',half=2,split='evaluation')]},d)
    def test_development_excluded_from_held_out_aggregate(self):
        import pandas as pd
        recordings=[SimpleNamespace(half=1,source_offset=0,ball=[0]*10,fps=25.,scope='half',sync={'match_id':match},events=[]) for match in ('dev','eval')]
        result=dict(passes=[],shots=[],possession=pd.DataFrame(dict(possessing_player_id=[None]*10,state=['unknown']*10)))
        def write(recording,result,config,path,*args,**kwargs):
            path.mkdir(parents=True);return dict(source_files=[],limitations=[])
        manifest={'recordings':[dict(match_id='dev',scope='half',split='development'),dict(match_id='eval',scope='half',split='evaluation')]}
        with TemporaryDirectory() as d,patch('src.evaluation.__main__.load_recording',side_effect=recordings),patch('src.evaluation.__main__.analyze_recording',return_value=result),patch('src.evaluation.__main__.write_outputs',side_effect=write),patch('src.evaluation.__main__.sensitivity',return_value={}):
            payload=run(manifest,d)
        self.assertEqual(payload['aggregate']['shots']['recordings'],1)
        self.assertEqual(payload['development_aggregate']['shots']['recordings'],1)
        self.assertEqual(payload['held_out_match_ids'],['eval'])
