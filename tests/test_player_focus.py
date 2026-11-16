import tempfile,unittest,uuid,json,threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch,Mock
import numpy as np
from src.analytics.motion import calculate_motion
from src.assistant.sprints import fastest_measured_run,qualifying_runs
from src.assistant.catalog import PlayerCatalog
from src.assistant.schema import AssistantRequest
from tests import test_assistant_m18 as fixtures

class FocusTests(unittest.TestCase):
 setUp=fixtures.ServiceTests.setUp
 ask=fixtures.ServiceTests.ask
 def test_bare_question_stays_on_the_current_player(self):
  r=self.ask('How many goals?',selected_player='actor:2');self.assertEqual(r['events'],[]);self.assertEqual(r['selected_player'],'actor:2');self.assertIn('Player B scored 0 goals',r['text'])
 def test_changing_player_drops_old_event_references(self):
  first=self.ask('Show goals',selected_player='actor:1')
  second=self.ask('Who was that?',selected_player='actor:2',previous_turn=first['turn_id']);self.assertEqual(second['events'],[]);self.assertNotIn('That was Player A',second['text'])
 def test_old_selected_response_does_not_override_new_focus(self):
  first=self.ask('How many goals?',selected_player='actor:1');second=self.ask('How many goals?',selected_player='actor:2',previous_turn=first['turn_id']);self.assertEqual(second['selected_player'],'actor:2');self.assertEqual(second['metrics'][0]['value'],0)
 def test_explicit_other_player_does_not_silently_change_focus(self):
  r=self.ask('Show Player A goals',selected_player='actor:2');self.assertIn('focusing on Player B',r['text']);self.assertFalse(r['events']);self.assertIsNone(r['highlight'])
 def test_player_videos_automatically_request_a_marker(self):
  cat=self.a.catalog('r');cat.boxes={'actor:1':{0:[10,10,20,40]}}
  with patch.object(self.h,'submit',return_value={'job_id':'test'}) as submit:
   self.ask('Show goals',selected_player='actor:1');overlay=submit.call_args.kwargs['selection'][3];self.assertEqual(overlay['player_ref'],'actor:1');self.assertEqual(overlay['version'],'player-focus-overlay-v3')
 def test_fallback_run_is_honestly_labelled_and_can_render(self):
  run={'event_id':'run_example','player_ref':'actor:1','actor_id':1,'player':10,'team':'left','start':2.,'end':3.,'speed_mps':5.2,'samples':25,'definition':'One-second observed run.','qualifies_as_sprint':False}
  with patch.object(self.a,'sprint_service',return_value=SimpleNamespace(intervals=[],runs=[run],diagnostic=None)),patch.object(self.h,'submit',return_value={'job_id':'test'}):
   r=self.ask("Show this player's fastest sprint",selected_player='actor:1');self.assertIn('fastest measured run',r['text']);self.assertIn('doesn’t meet',r['text']);self.assertEqual(r['metrics'][0]['name'],'Fastest measured run speed');self.assertIsNotNone(r['highlight'])
 def test_action_summary_describes_statistics_in_plain_language(self):
  r=self.ask('Show their actions',selected_player='actor:1');self.assertIn('Player A took 0 shots, scored 1 goal',r['text']);self.assertIn('completed 0 of 0 passes',r['text']);self.assertNotIn('recorded or supported actions',r['text'])

class MeasuredRunTests(unittest.TestCase):
 def setUp(self):self.cfg=json.loads(Path('configs/assistant/m18.json').read_text())
 def test_subthreshold_run_is_not_reclassified_as_a_sprint(self):
  p=np.column_stack([np.arange(150)*.2+5,np.full(150,30.)]);m=calculate_motion(p,25);self.assertEqual(list(qualifying_runs(m,25,self.cfg)),[]);r=fastest_measured_run(m,25,self.cfg);self.assertIsNotNone(r);self.assertAlmostEqual(r[2],5,places=4)
 def test_gap_and_spike_are_not_used_as_peak_run(self):
  p=np.column_stack([np.arange(150)*.2+5,np.full(150,30.)]);p[55:80]=np.nan;p[110]=[900,900];r=fastest_measured_run(calculate_motion(p,25),25,self.cfg);self.assertTrue(r);self.assertFalse(r[0]<70<r[1]);self.assertLess(r[2],14)
 def test_measured_run_indices_are_json_serializable(self):
  p=np.column_stack([np.arange(150)*.2+5,np.full(150,30.)]);r=fastest_measured_run(calculate_motion(p,25),25,self.cfg);self.assertIsInstance(r[0],int);self.assertIsInstance(r[1],int);json.dumps(r,allow_nan=False)
 def test_insufficient_or_stationary_motion_stays_unavailable(self):
  for p in [np.column_stack([np.arange(20)*.2,np.full(20,30.)]),np.full((100,2),30.),np.full((100,2),np.nan)]:self.assertIsNone(fastest_measured_run(calculate_motion(p,25),25,self.cfg))

class ImagePointTests(unittest.TestCase):
 def make(self):
  c=PlayerCatalog.__new__(PlayerCatalog);c.point_coords=np.array([[[100.,200.],[300.,400.]]]);c.point_ids=np.array([[7,8]]);c.frames=1;c.by_ref={'actor:7':{'actor_id':7},'actor:8':{'actor_id':8}};c.boxes={};return c
 def test_exact_actor_image_point_no_nearest_player_substitution(self):
  c=self.make();np.testing.assert_array_equal(c.image_point('actor:7',0),[100,200]);self.assertIsNone(c.image_point('actor:missing',0));self.assertIsNone(c.image_point('actor:7',1))
 def test_duplicate_or_missing_point_stays_hidden(self):
  c=self.make();c.point_ids[0,1]=7;self.assertIsNone(c.image_point('actor:7',0));c.point_ids[0,1]=8;c.point_coords[0,0]=np.nan;self.assertIsNone(c.image_point('actor:7',0))
 def test_source_hash_failure_does_not_load_image_points(self):
  c=self.make();c.frames=2
  with tempfile.TemporaryDirectory() as tmp:
   with self.assertRaises(ValueError):c.load_image_points({'half':1,'match_id':'118575','source_offset_frames':0},Path(tmp),{})

class SprintBuildTests(unittest.TestCase):
 def test_build_serializes_fallback_events_for_real_numpy_frame_indices(self):
  from src.assistant.sprints import SprintService
  from tests.test_assistant_m18 import catalog
  cat=catalog();cat.frames=150;cat.tools=SimpleNamespace(summary={'player_coordinate_source':'gsr','half':1,'match_id':'test','source_offset_frames':0,'source_files':[{'path':'1st_compact.npz','sha256':'verified'}]})
  positions=np.column_stack([np.arange(150)*.2+5,np.full(150,30.)])[:,None,:]
  with patch('src.assistant.sprints.file_hash',return_value='verified'),patch('src.assistant.sprints.np.load') as data:
   data.return_value.__enter__.return_value={'pitch':positions,'ids':np.ones((150,1),int)};service=SprintService(cat)
  self.assertIsNone(service.diagnostic);self.assertTrue(service.runs);json.dumps(service.runs,allow_nan=False);self.assertEqual(service.intervals,[])
