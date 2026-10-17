import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from src.serving.app import create_app
from src.serving.review import Replay,load_replay
from tests.helpers import write_query_artifacts

class FrontendReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.path=Path(self.temp.name);self.event=write_query_artifacts(self.path)
        self.replay=dict(fps=25,frames=10,stride=5,ids=[101],teams=['left'],side_colors={'0':1,'1':0},positions=[[[1,2]],[[3,4]]],ball=[[1,2],[None,None]],owner=[101,None],states=['controlled','unknown'])
    def client(self):return TestClient(create_app({'demo':self.path}))
    def test_missing_replay_is_explicitly_unknown(self):
        data=self.client().get('/matches/demo/view').json();self.assertEqual(data['status'],'unknown');self.assertIsNone(data['replay']);self.assertIsNone(data['media_url'])
    def test_precomputed_json_projection(self):
        (self.path/'review_frames.json').write_text(json.dumps(self.replay))
        data=self.client().get('/matches/demo/view').json();self.assertEqual(data['status'],'ok');self.assertEqual(data['replay']['positions'],self.replay['positions']);self.assertIn('review_frames.json',data['sources'])
    def test_legacy_report_literal_parsed_without_script_execution(self):
        raw={**self.replay,'sideColors':self.replay['side_colors']}
        (self.path/'report.html').write_text('<script>const data='+json.dumps(raw)+';throw Error("must not run");</script>')
        replay,source=load_replay(self.path);self.assertEqual(replay.frames,10);self.assertEqual(source,'report.html#precomputed_review')
    def test_malformed_legacy_report_is_unavailable(self):
        (self.path/'report.html').write_text('<script>const data={broken};</script>');self.assertIsNone(load_replay(self.path)[0])
    def test_lengths_and_actor_mapping_validated(self):
        for changes in [{'ball':[]},{'positions':[[[1,2]]]}, {'ids':[101,101]}]:
            with self.assertRaises(ValidationError):Replay.model_validate({**self.replay,**changes})
    def test_nonfinite_positions_rejected(self):
        with self.assertRaises(ValidationError):Replay.model_validate({**self.replay,'ball':[[float('nan'),1],[1,1]]})
    def test_unknown_recording_and_traversal(self):
        self.assertEqual(self.client().get('/matches/missing/view').status_code,404)
        self.assertEqual(self.client().get('/matches/%2E%2E%2Fsecret/view').status_code,404)
    def test_view_uses_startup_loaded_samples(self):
        (self.path/'review_frames.json').write_text(json.dumps(self.replay));client=self.client()
        with patch('src.serving.app.load_replay',side_effect=AssertionError('No per-request reload')),patch('src.analytics.pipeline.analyze_recording',side_effect=AssertionError('No analytics')):
            self.assertEqual(client.get('/matches/demo/view').status_code,200);self.assertEqual(client.get('/matches/demo/view').status_code,200)
    def test_response_is_closed_and_secret_free(self):
        data=self.client().get('/matches/demo/view').json();self.assertNotIn('API_KEY',json.dumps(data))
        schema=self.client().get('/openapi.json').json()['components']['schemas']['Review'];self.assertFalse(schema['additionalProperties'])
    def test_explanation_annotation_reuses_backend_categories(self):
        response=self.client().get(f'/matches/demo/events/{self.event.event_id}/explain?with_evidence=true&mode=analyst')
        self.assertEqual(response.status_code,200);data=response.json();self.assertEqual(data['answer']['mode'],'analyst');self.assertTrue(data['evidence']);self.assertTrue(all(e['category'] in ('observed','inferred','validated','heuristic','unknown') for e in data['evidence']))
    def test_default_explain_response_is_preserved(self):
        data=self.client().get(f'/matches/demo/events/{self.event.event_id}/explain').json();self.assertIn('statements',data);self.assertNotIn('answer',data)
    def test_unconfirmed_denominator_and_provenance(self):
        p=self.path/'summary.json';s=json.loads(p.read_text());s['frames']=10;s['possession_state_frames']={'controlled':4};p.write_text(json.dumps(s))
        data=self.client().get('/matches/demo/view').json();self.assertEqual(data['unconfirmed_control_fraction'],.6);self.assertIn('summary.json',data['sources'])
