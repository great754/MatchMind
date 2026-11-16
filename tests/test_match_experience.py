import json,tempfile,threading,unittest,uuid,hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch,Mock
import requests,numpy as np
from pydantic import ValidationError
from src.assistant.profiles import MatchPresentation
from src.assistant.planner import IntentAndQueryPlanner
from src.assistant.schema import AssistantPlan
from src.highlights.azure_stream import azure_video_source
from src.highlights.render import HighlightError
from src.highlights.boundaries import ClipBoundaryEngine
from src.highlights.schema import HighlightQueryPlan
from tests.test_highlights_m17 import event,tools
from tests.test_assistant_m18 import catalog

class ScoreTests(unittest.TestCase):
 def test_full_halves_override_duplicate_clip_goal_and_preserve_teams(self):
  def t(half,scope,events):return SimpleNamespace(summary={'match_id':'118575','half':half,'scope':scope},events=events)
  first=event(team='right',actor_id=1);second=event(frame=200,team='left',actor_id=2)
  a=SimpleNamespace(loaded={'h1':t(1,'half',[first]),'h2':t(2,'half',[second]),'clip':t(1,'clip',[first])});score=MatchPresentation(a).score('clip');self.assertEqual((score['left'],score['right']),(1,1));self.assertEqual(score['scope'],'full_match');self.assertEqual(len(score['goals']),2)
 def test_clip_only_score_is_not_claimed_full_match(self):
  a=SimpleNamespace(loaded={'c':SimpleNamespace(summary={'match_id':'118575','half':1,'scope':'clip'},events=[event(team='right')])});s=MatchPresentation(a).score('c');self.assertEqual(s['right'],1);self.assertEqual(s['scope'],'recording')

class RangeSourceTests(unittest.TestCase):
 def setUp(self):
  self.data=b'0123456789'*100;self.blob=Mock();self.blob.get_blob_properties.return_value=SimpleNamespace(size=len(self.data),etag='etag-one');self.calls=[]
  def download(**kw):self.calls.append(kw);data=self.data[kw['offset']:kw['offset']+kw['length']];return SimpleNamespace(chunks=lambda:iter([data]))
  self.blob.download_blob.side_effect=download;client=Mock();client.get_blob_client.return_value=self.blob
  self.media=SimpleNamespace(register_half=Mock(),client=client,blob_name=lambda _: 'videos/118575/half.mp4')
 def test_range_and_metadata_pin_the_registered_etag(self):
  with azure_video_source(self.media,'h1',{'match_id':'118575','half':1},threading.Event()) as (url,fingerprint,origin,state):
   head=requests.head(url);self.assertEqual(head.status_code,200);self.assertEqual(len(self.calls),0)
   r=requests.get(url,headers={'Range':'bytes=2-5'});self.assertEqual(r.status_code,206);self.assertEqual(r.content,b'2345');self.assertEqual(self.calls[0]['etag'],'etag-one');self.assertIn('match_condition',self.calls[0]);self.assertEqual(state['bytes'],4);self.assertEqual(len(fingerprint),64)
   self.assertEqual(requests.get(url.replace('/source.mp4','/private')).status_code,404);self.assertEqual(requests.get(url,headers={'Range':'bytes=10000-'}).status_code,416)
 def test_read_budget_is_enforced_and_never_claims_source_sha(self):
  with self.assertRaises(HighlightError):
   with azure_video_source(self.media,'h1',{'match_id':'118575','half':1},threading.Event(),byte_budget=2) as (url,_,origin,state):
    try:requests.get(url)
    except requests.RequestException:pass
    self.assertEqual(state['failure'],'source_read_budget_exceeded');self.assertNotIn('source_sha256',origin)
 def test_changed_blob_is_rejected(self):
  error=RuntimeError('changed');error.status_code=412;self.blob.download_blob.side_effect=error
  with self.assertRaises(HighlightError):
   with azure_video_source(self.media,'h1',{'match_id':'118575','half':1},threading.Event()) as (url,_,_,state):self.assertEqual(requests.get(url).status_code,503);self.assertEqual(state['failure'],'source_changed')

class GoalWindowTests(unittest.TestCase):
 def test_three_goals_keep_fifteen_seconds_of_lead_in(self):
  rows=[{'event_id':str(t),'event_type':'goal','start':t,'end':t,'verification':'annotated','timebase':{'period':1}} for t in [400,1800,2600]];parts=ClipBoundaryEngine().build(rows,HighlightQueryPlan(),2800,25);self.assertEqual(len(parts),3)
  for row,part in zip(rows,parts):self.assertAlmostEqual(row['start']-part['start'],15,places=4);self.assertGreaterEqual(part['end']-row['end'],4.9)

class LanguageFirstTests(unittest.TestCase):
 def test_model_can_interpret_a_request_unrecognized_by_local_patterns(self):
  result={'operation':'actions','actions':['goal'],'video':True,'buildup':'more'};client=SimpleNamespace(client=SimpleNamespace(generate_structured=lambda *args:json.dumps(result)))
  with patch('src.providers.factory.build_provider',return_value=client):
   p,_,provider,_=IntentAndQueryPlanner().plan('Put the scoring finishes into a film with their lead-in',provider='gemini',roster=[]);self.assertEqual(provider,'gemini');self.assertTrue(p.video);self.assertEqual(p.actions,['goal'])
 def test_model_cannot_create_a_player_reference(self):
  result={'operation':'player_profile','players':['actor:wrong']};client=SimpleNamespace(client=SimpleNamespace(generate_structured=lambda *args:json.dumps(result)))
  with patch('src.providers.factory.build_provider',return_value=client):self.assertEqual(IntentAndQueryPlanner().plan('Tell me about this performance',provider='gemini',roster=[])[2],'local_fallback')
 def test_broad_player_statistics_are_supported_offline_too(self):
  p=IntentAndQueryPlanner().plan('Show me stats about Player A',players=['actor:1'],provider='local')[0];self.assertEqual(p.operation,'player_profile');self.assertFalse(p.video)

class BuildupBudgetTests(unittest.TestCase):
 def test_more_buildup_does_not_get_recentered_around_the_goal(self):
  rows=[{'event_id':str(t),'event_type':'goal','start':t,'end':t,'verification':'annotated','timebase':{'period':1}} for t in [400,1800,2600]]
  segments=ClipBoundaryEngine().build(rows,HighlightQueryPlan(buildup='more',duration=120),2800,25)
  self.assertEqual(len(segments),3)
  for s in segments:self.assertGreaterEqual(s['event_start']-s['start'],20);self.assertGreaterEqual(s['end']-s['event_end'],3)
  self.assertLessEqual(sum(s['end']-s['start'] for s in segments),120.000001)

class ProfileTests(unittest.TestCase):
 def make(self):
  from src.assistant.evidence import AssistantEventIndex
  from src.assistant.identity import ConfirmedIdentityStore
  from src.highlights.render import file_hash
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);cat=catalog();t=tools([event(player_id=10,actor_id=1)])
  completed=event(kind='PASS_COMPLETED',frame=200,player_id=10,actor_id=1);completed=completed.model_copy(update={'source':'transfer','metrics':completed.metrics.model_copy(update={'outcome':'completed','observed_step_fraction':1,'distance_m':15})});t.events.append(completed)
  t.summary.update(match_id='118575');t.players=[{'player_id':1,'measured_distance_m':100,'max_speed_mps':8}]
  t.transfers=[{'sender_id':1,'outcome':outcome,'observed_step_fraction':1,'start_x_m':10,'start_y_m':20,'end_x_m':20,'end_y_m':30,'start_frame':i*25,'pass_id':str(i)} for i,outcome in enumerate(['completed','intercepted'])];t._transfer=lambda _:(None,t.transfers[0]);cat.tools=t;idx=AssistantEventIndex('r',t,cat)
  a=SimpleNamespace(catalog=lambda _:cat,index=lambda _:idx,loaded={'r':t},identities=ConfirmedIdentityStore(self.tmp.name),portraits=SimpleNamespace(public=lambda _:{'players':{}}));return MatchPresentation(a)
 def test_complete_profile_and_map_are_from_explicit_actor_rows(self):
  p=self.make();p.trajectory=lambda *_:(np.array([[10.,20.],[10.,20.],[100.,60.]]),np.array([0,1,2]),'a'*64)
  result=p.profile('r','actor:1',uuid.uuid4());stats={r['key']:r['value'] for r in result['stats']};self.assertEqual(stats['goals'],1);self.assertEqual(stats['passes'],1);self.assertEqual(stats['pass_attempts'],2);self.assertEqual(stats['pass_accuracy'],50);self.assertEqual(len(result['pass_map']),2);self.assertAlmostEqual(float(np.sum(result['heatmap']['seconds'])),3/25);self.assertAlmostEqual(result['tracked_minutes'],3/25/60)
 def test_unknown_heatmap_is_not_filled_with_invented_locations(self):
  p=self.make();p.trajectory=lambda *_:None;result=p.profile('r','actor:1',uuid.uuid4());self.assertIsNone(result['heatmap']);self.assertIsNone(result['tracked_minutes'])
