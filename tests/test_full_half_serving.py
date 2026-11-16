import json,tempfile,unittest
from pathlib import Path
from unittest.mock import Mock
from types import SimpleNamespace
from fastapi.testclient import TestClient
from tests.helpers import write_query_artifacts
from src.serving.app import create_app
from src.serving.media import AzureClipMedia
class FullHalfServingTests(unittest.TestCase):
 def test_full_half_is_selectable_and_streams_registered_panorama(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);write_query_artifacts(p);s=json.loads((p/'summary.json').read_text());s.update(scope='half',half=2);(p/'summary.json').write_text(json.dumps(s));blob=Mock();blob.get_blob_properties.return_value=SimpleNamespace(size=4,etag='half');blob.download_blob.return_value=SimpleNamespace(chunks=lambda:iter([b'half']));container=Mock();container.get_blob_client.return_value=blob
   client=TestClient(create_app({'match_h2':p},media_store=AzureClipMedia(container)));row=client.get('/recordings').json()['recordings'][0];self.assertTrue(row['media_available']);self.assertEqual(client.get('/matches/match_h2/view').json()['media_url'],'/matches/match_h2/video.mp4');self.assertEqual(client.get('/matches/match_h2/video.mp4',headers={'Range':'bytes=0-3'}).content,b'half');container.get_blob_client.assert_called_with('videos/synthetic/synthetic_panorama_2nd_half.mp4')

class CVTimeoutStartupTests(unittest.TestCase):
 def make(self,side_effect):
  from unittest.mock import patch
  with patch('src.serving.cv.CVReview',side_effect=side_effect) as constructor:
   client=TestClient(create_app({},cv_registry={'cv_test':'unused'}));return client,constructor.call_count
 def test_transient_read_timeout_retries_full_validation(self):
  review=SimpleNamespace(manifest={'recording_id':'cv_test','pitch_calibration_available':False})
  client,calls=self.make([TimeoutError('hydrating'),review]);self.assertEqual(calls,2);self.assertEqual(client.get('/ready').status_code,200)
 def test_corrupt_source_is_not_retried_or_accepted(self):
  client,calls=self.make([ValueError('hash mismatch')]);self.assertEqual(calls,1);self.assertEqual(client.get('/ready').status_code,503)
 def test_persistent_timeout_stays_unavailable(self):
  client,calls=self.make([TimeoutError(),TimeoutError()]);self.assertEqual(calls,2);self.assertEqual(client.get('/ready').status_code,503)

class CompactWorkspaceTests(unittest.TestCase):
 def test_compact_view_preserves_replay_events_and_facts(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);write_query_artifacts(p);client=TestClient(create_app({'r':p}));full=client.get('/matches/r/view').json();compact=client.get('/matches/r/view',headers={'X-TactiqAI-View':'workspace'}).json()
   for field in ['replay','events','summary','pass_share','formation_windows','media_url']:self.assertEqual(full[field],compact[field])
   self.assertEqual(compact['tactical_states'],[]);self.assertEqual(compact['pressure_context'],[]);self.assertEqual(compact['ownership'],full['ownership'][::(full['replay']['stride'] if full['replay'] else 1)])
