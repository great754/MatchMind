import unittest,tempfile,threading,json,time,uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from fastapi import HTTPException
from pydantic import ValidationError
from src.assistant.schema import AssistantRequest,AssistantPlan,IdentityConfirmation
from src.assistant.identity import ConfirmedIdentityStore
from src.assistant.planner import IntentAndQueryPlanner
from src.assistant.composer import ResponseComposer
from src.assistant.portraits import quality
from src.assistant.catalog import letter
from src.assistant.evidence import AssistantEventIndex,PlayerIntelligenceService,TacticalExplanationService
from src.assistant.service import UnifiedMatchAssistant
from src.highlights.service import HighlightService
from tests.test_highlights_m17 import tools,event

def catalog():
 players=[{'player_ref':'actor:1','label':'Player A','actor_id':1,'mot_id':10,'team':'left','start':0.,'end':20.,'source_signature':'source'},{'player_ref':'actor:2','label':'Player B','actor_id':2,'mot_id':20,'team':'right','start':0.,'end':20.,'source_signature':'source'}]
 return SimpleNamespace(players=players,by_ref={p['player_ref']:p for p in players},recording='r',tools=None,boxes={},source_signature='source',fps=25,frames=500)

class IdentityTests(unittest.TestCase):
 def setUp(self):self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.store=ConfirmedIdentityStore(self.temp.name);self.cat=catalog();self.session=uuid.uuid4()
 def body(self,ref='actor:1',**kw):return IdentityConfirmation(session_id=self.session,player_ref=ref,jersey=7,name='Alex',team='left',confirmed=True,**kw)
 def test_no_tracker_to_jersey_alias(self):
  refs,msg=self.store.resolve('Show player 10 actions',{'players':{}},self.cat);self.assertFalse(refs);self.assertIn('confirm',msg)
 def test_confirmed_jersey_resolves(self):
  data=self.store.confirm('r',self.body(),self.cat);self.assertEqual(self.store.resolve('How many shots did player 7 take?',data,self.cat),(['actor:1'],None));self.assertEqual(self.store.label('actor:1',data,self.cat),'Alex #7')
 def test_duplicate_same_team_and_time_rejected(self):
  self.store.confirm('r',self.body(),self.cat)
  with self.assertRaises(HTTPException) as e:self.store.confirm('r',self.body('actor:2'),self.cat)
  self.assertEqual(e.exception.status_code,409)
 def test_same_number_other_team_is_ambiguous(self):
  self.store.confirm('r',self.body(),self.cat);b=self.body('actor:2').model_copy(update={'team':'right'});data=self.store.confirm('r',b,self.cat);self.assertIn('both teams',self.store.resolve('Show player 7',data,self.cat)[1])
 def test_nonoverlapping_segments_can_reuse_number(self):
  self.store.confirm('r',self.body(),self.cat);self.cat.by_ref['actor:2']['start']=21;self.cat.by_ref['actor:2']['end']=30;data=self.store.confirm('r',self.body('actor:2'),self.cat);self.assertEqual(len(data['players']),2)
 def test_wrong_photo_cannot_confirm_identity(self):
  with self.assertRaises(HTTPException):self.store.confirm('r',self.body(image_id='wrong'),self.cat,{'actor:2':[{'image_id':'wrong'}]})
 def test_change_source_invalidates_annotation(self):
  self.store.confirm('r',self.body(),self.cat);self.cat.by_ref['actor:1']['source_signature']='changed';self.assertFalse(self.store.load('r',self.session,self.cat)['players'])
 def test_reversible_and_session_match_scoped(self):
  self.store.confirm('r',self.body(),self.cat);self.assertFalse(self.store.load('other',self.session,self.cat)['players']);self.assertFalse(self.store.load('r',uuid.uuid4(),self.cat)['players']);self.assertFalse(self.store.remove('r',self.session,'actor:1',self.cat)['players'])
 def test_temporary_label_is_not_a_number(self):self.assertEqual(self.store.resolve('Show Player A',{'players':{}},self.cat)[0],['actor:1']);self.assertEqual(letter(26),'AA')

class PlanTests(unittest.TestCase):
 def setUp(self):self.p=IntentAndQueryPlanner()
 def plan(self,q,**kw):return self.p.plan(q,provider='local',**kw)[0]
 def test_combined_difficulty_is_not_longest(self):
  p=self.plan('What was the most difficult pass, why was it difficult, and show me the clip?');self.assertEqual(p.metric,'difficulty');self.assertTrue(p.video and p.explain)
  self.assertEqual(self.plan('Who made the longest pass and can I see it?').metric,'distance')
 def test_fastest_sprint_overlay(self):
  p=self.plan('Show me the fastest sprint and highlight the player.');self.assertEqual(p.operation,'sprint');self.assertTrue(p.video and p.overlay)
 def test_followups_do_not_rerank(self):
  p=self.plan('Why was that pass difficult?',previous={'players':['actor:1']});self.assertEqual(p.operation,'explain');self.assertIsNone(p.metric)
  p=self.plan('Show me that pass again',previous={'players':['actor:1']});self.assertEqual(p.operation,'followup');self.assertTrue(p.video)
 def test_pronoun_player_scope(self):self.assertEqual(self.plan('Show all his passes',previous={'players':['actor:2']}).players,['actor:2'])
 def test_unsupported_shot_speed(self):self.assertEqual(self.plan('Show fastest shot').operation,'unsupported')
 def test_no_raw_ids_or_times_allowed(self):
  with self.assertRaises(ValidationError):AssistantPlan(event_id='fake')
  with self.assertRaises(ValidationError):AssistantRequest(question='show',session_id=uuid.uuid4(),source='/etc/passwd')
 def test_compare_and_half(self):self.assertEqual(self.plan('Compare player 7 and player 10').operation,'compare');self.assertEqual(self.plan('Show his actions in the first half').period,1)

class ImageTests(unittest.TestCase):
 def test_bad_and_occluded_crops_rejected(self):
  image=np.zeros((200,200,3),np.uint8);self.assertIsNone(quality(image,[-5,10,30,50],[]));self.assertIsNone(quality(image,[10,10,30,50],[[10,10,30,50],[11,11,30,50]]))
 def test_clear_crop_outranks_blur(self):
  import cv2
  rng=np.random.default_rng(18);image=rng.integers(0,255,(200,200,3),dtype=np.uint8);sharp=quality(image,[20,20,70,140],[]);blur=quality(cv2.GaussianBlur(image,(21,21),9),[20,20,70,140],[]);self.assertGreater(sharp[1]['score'],blur[1]['score']);self.assertEqual(sharp[1]['readability'],'UNVERIFIED');self.assertEqual(sharp[1]['orientation'],'UNKNOWN')

class ComposerTests(unittest.TestCase):
 def test_codes_stay_out_of_plain_text_without_corrupting_based(self):
  with tempfile.TemporaryDirectory() as d:
   s=ConfirmedIdentityStore(d);text=ResponseComposer.readable('Actor 1 made a pass based on BAS data. Event evt_abc and MOT GSR.',catalog(),{'players':{}},s);self.assertIn('based',text);self.assertIn('Player A',text)
   for bad in ['Actor 1','BAS','MOT','GSR','evt_']:self.assertNotIn(bad,text)
 def test_difficulty_explanation_uses_only_present_factors(self):
  text=TacticalExplanationService.pass_explanation({'metrics':{'distance_m':30.4}});self.assertIn('30.4 meters',text);self.assertNotIn('two defenders',text);self.assertIn('estimate',text)

class ServiceTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);t=tools([event(player_id=10,actor_id=1)]);t.players=[{'player_id':1,'mot_id':10,'team':'left','measured_distance_m':100,'valid_motion_fraction':1}];t.transfers=[];t._transfer=lambda _: (None,None);self.t=t
  self.h=HighlightService({'r':t},{},root=Path(self.temp.name)/'h');self.addCleanup(self.h.executor.shutdown,wait=True);self.a=UnifiedMatchAssistant({'r':t},{},self.h,root=Path(self.temp.name)/'a');c=catalog();c.tools=t;self.a.catalogs['r']=c;self.session=uuid.uuid4()
 def ask(self,q,**kw):return self.a.ask('r',AssistantRequest(question=q,session_id=self.session,provider='local',**kw))
 def test_unconfirmed_number_clarifies_without_zero_count(self):
  r=self.ask('How many shots did player 7 take?');self.assertIn('confirm',r['text']);self.assertEqual(r['metrics'],[])
 def test_supported_count_after_manual_confirmation(self):
  self.a.identities.confirm('r',IdentityConfirmation(session_id=self.session,player_ref='actor:1',jersey=7,name='Alex',team='left',confirmed=True),self.a.catalog('r'))
  r=self.ask('How many goals did player 7 have?');self.assertIn('scored 1 goal',r['text']);self.assertIn('Alex #7',r['text']);self.assertEqual(r['events'][0]['actor_id'],1)
 def test_context_cannot_cross_session_or_match(self):
  r=self.ask('How many shots did player 7 take?')
  with self.assertRaises(HTTPException):self.a.ask('r',AssistantRequest(question='Who was that?',session_id=uuid.uuid4(),previous_turn=r['turn_id'],provider='local'))
 def test_followup_identity_uses_selected_event(self):
  r=self.ask('Show goals');second=self.ask('Who was that?',previous_turn=r['turn_id']);self.assertIn('Player A',second['text']);self.assertNotIn('10',second['text'])

class RouteTests(unittest.TestCase):
 def test_roster_does_not_collide_with_legacy_integer_player_route(self):
  from src.serving.app import create_app
  from fastapi.testclient import TestClient
  app=create_app({});cat=catalog();cat.public=lambda:{'players':cat.players};app.state.assistant.catalog=lambda recording:cat
  with TestClient(app) as client:
   r=client.get('/matches/r/players/roster',params={'session_id':str(uuid.uuid4())});self.assertEqual(r.status_code,200);self.assertEqual(len(r.json()['players']),2)
   self.assertEqual(client.get('/matches/r/players/-1').status_code,422)

class MotionTests(unittest.TestCase):
 def test_existing_motion_does_not_bridge_a_gap(self):
  from src.analytics.motion import calculate_motion
  p=np.column_stack([np.arange(100)*.32,np.zeros(100)]);p[50:55]=np.nan;m=calculate_motion(p,25);self.assertTrue(np.isnan(m.smoothed_speed_mps[50:55]).all())
 def test_constant_metric_speed_and_spike_are_distinct(self):
  from src.analytics.motion import calculate_motion
  p=np.column_stack([np.arange(100)*.32,np.zeros(100)]);m=calculate_motion(p,25);self.assertAlmostEqual(float(np.nanmedian(m.smoothed_speed_mps[10:90])),8,places=4)
  p[40]=[900,900];m=calculate_motion(p,25);self.assertTrue(m.raw_speed_flag[40]);self.assertLess(float(np.nanmedian(m.smoothed_speed_mps)),14)

if __name__=='__main__':unittest.main()

class SprintEligibilityTests(unittest.TestCase):
 def run_motion(self,positions):
  from src.analytics.motion import calculate_motion
  from src.assistant.sprints import qualifying_runs
  cfg=json.loads(Path('configs/assistant/m18.json').read_text());return list(qualifying_runs(calculate_motion(positions,25),25,cfg))
 def test_sustained_metric_run_is_eligible(self):
  r=self.run_motion(np.column_stack([np.arange(100)*.32,np.zeros(100)]));self.assertTrue(r);self.assertAlmostEqual(r[0][2],8,places=4)
 def test_short_motion_cannot_be_a_sprint(self):self.assertEqual(self.run_motion(np.column_stack([np.arange(20)*.32,np.zeros(20)])),[])
 def test_gap_is_never_bridged(self):
  p=np.column_stack([np.arange(150)*.32,np.zeros(150)]);p[70:80]=np.nan;r=self.run_motion(p);self.assertTrue(r);self.assertTrue(all(not(a<75<b) for a,b,*_ in r))
 def test_subthreshold_motion_rejected(self):self.assertEqual(self.run_motion(np.column_stack([np.arange(100)*.08,np.zeros(100)])),[])

class ProviderPlanTests(unittest.TestCase):
 def gemini(self,plan):return SimpleNamespace(client=SimpleNamespace(generate_structured=lambda *args:json.dumps(plan)))
 def test_gemini_structured_plan_is_used(self):
  with patch('src.providers.factory.build_provider',return_value=self.gemini({'operation':'actions','actions':['goal'],'video':True})):
   p,_,used,_=IntentAndQueryPlanner().plan('Show goals',provider='gemini');self.assertEqual(used,'gemini');self.assertEqual(p.actions,['goal'])
 def test_provider_cannot_drop_player_or_invent_time_filter(self):
  for result in [{'operation':'actions','actions':['pass'],'video':True,'players':[]},{'operation':'actions','actions':['pass'],'video':True,'players':['actor:1'],'start':42}]:
   with patch('src.providers.factory.build_provider',return_value=self.gemini(result)):
    p,_,used,_=IntentAndQueryPlanner().plan('Show passes',players=['actor:1'],provider='gemini');self.assertEqual(used,'local_fallback');self.assertEqual(p.players,['actor:1']);self.assertIsNone(p.start)
 def test_microsoft_closed_tool_plan_is_used(self):
  call=SimpleNamespace(name='assistant_query',arguments={'operation':'actions','actions':['goal'],'video':True});client=SimpleNamespace(plan=lambda *args:SimpleNamespace(calls=[call]))
  with patch('src.providers.factory.build_provider',return_value=client):self.assertEqual(IntentAndQueryPlanner().plan('Show goals',provider='microsoft')[2],'microsoft')
 def test_duration_recap_and_ranges(self):
  p=IntentAndQueryPlanner().plan('Make a 60-second highlight video',provider='local')[0];self.assertEqual(p.operation,'recap');self.assertEqual(p.duration,60);self.assertIsNone(p.ordinal)
  with self.assertRaises(ValidationError):AssistantPlan(start=20,end=10)

class CountAndFilterTests(unittest.TestCase):
 setUp=ServiceTests.setUp
 ask=ServiceTests.ask
 def test_global_count_does_not_become_a_player_ranking(self):
  r=self.ask('How many goals were there?');self.assertIn('1 recorded goal entries',r['text']);self.assertEqual(r['metrics'][0]['value'],1);self.assertIsNone(r['selected_player'])
 def test_player_filtered_count_reports_requested_window(self):
  self.a.identities.confirm('r',IdentityConfirmation(session_id=self.session,player_ref='actor:1',jersey=7,team='left',confirmed=True),self.a.catalog('r'))
  r=self.ask('How many goals did player 7 have between 0:00 and 0:01?');self.assertEqual(r['metrics'][0]['value'],0);self.assertEqual(r['metrics'][0]['window'],(0.,1.));self.assertEqual(r['events'],[])
 def test_unknown_named_filter_is_not_silently_dropped(self):
  r=self.ask('Show goals by Messi');self.assertIn('confirm',r['text']);self.assertFalse(r['events']);self.assertIsNone(r['highlight'])
 def test_receptions_are_not_duplicated_as_sender_passes(self):self.assertEqual(IntentAndQueryPlanner().plan('Show pass receptions',provider='local')[0].actions,['reception'])

class BundleMetadataTests(unittest.TestCase):
 def test_canonical_players_remain_available_without_optional_raw_video_mapping(self):
  from src.assistant.catalog import PlayerCatalog
  t=SimpleNamespace(summary={'fps':25,'frames':500,'match_id':'no-local-raw-data','source_files':[]},players=[{'player_id':1,'team':'left'}])
  c=PlayerCatalog('r',t);self.assertEqual(c.players[0]['player_ref'],'actor:1');self.assertIsNone(c.players[0]['mot_id']);self.assertFalse(c.boxes);self.assertIn('could not be checked',c.diagnostic)

class CVSegmentCatalogTests(unittest.TestCase):
 def test_validated_cv_source_is_scanned_once_without_merging_appearances(self):
  from src.assistant.catalog import PlayerCatalog
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);source=root/'frames.jsonl';video=root/'video.mp4';video.write_bytes(b'test')
   rows=[]
   for i in range(8):
    segment='first' if i<4 else 'second';rows.append({'entities':[{'track_id':'track','image_bbox':[10,20,30,40]}],'unassigned':[],'player_world':{'players':[{'current_segment_id':segment,'current_track_id':'track','visibility_state':'VISIBLE','identity_status':'PROVISIONAL','entity_id':'same-hypothesis','team':{'label':'TEAM_A'}}]}})
   source.write_text(''.join(json.dumps(r)+'\n' for r in rows));cv=SimpleNamespace(manifest={'fps':10,'frames':8,'frames_sha256':'verified-source'},data=source,video=video,data_stamp=source.stat().st_mtime_ns,video_stamp=video.stat().st_mtime_ns,frame=lambda _:self.fail('Do not reparse validated WorldState models'))
   c=PlayerCatalog('cv',cv=cv);self.assertEqual(len(c.players),2);self.assertEqual(c.players[0]['end'],.4);self.assertEqual(c.players[1]['start'],.4);self.assertEqual(set(c.boxes['segment:first']),set(range(4)));self.assertEqual(set(c.boxes['segment:second']),set(range(4,8)))
   cv.data_stamp-=1;c=PlayerCatalog('cv',cv=cv);self.assertEqual(c.players,[]);self.assertFalse(c.boxes)

class TeamQuestionTests(unittest.TestCase):
 def test_team_comparison_never_becomes_an_individual_ranking(self):
  planner=IntentAndQueryPlanner();p=planner.plan('Which team completed the most passes?',provider='local')[0];self.assertEqual(p.operation,'compare');self.assertEqual(p.metric,'passes')
  p=planner.plan('Which team had the most possession?',provider='local')[0];self.assertEqual(p.operation,'answer')
