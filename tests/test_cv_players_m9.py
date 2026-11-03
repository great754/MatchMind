import unittest
import numpy as np
from pydantic import ValidationError
from src.vision.world.player_identity import ShortTrack,candidate,globally_associate,build_segments
from src.vision.world.player_state import PlayerWorldConfig,PersistentPerson,PlayerWorldFrame,TeamIntelligence
from src.vision.world.player_teams import PersistentTeam,JerseyBelief,TeamAppearance
from src.vision.world.player_appearance import foreground

FPS=25

def segment(key,start,end,x=20,uncertainty=.2,team='TEAM_A',embedding=(1,0,0),epoch=0):
 s=ShortTrack(key,key,epoch,embedding=np.array(embedding,float),histogram=np.array([1,0],float),quality=.8,team_label=team,team_strength=.95,evidence_frames=[start,end])
 for f in range(start,end+1):s.observations.append({'frame':f,'track_id':key,'foot':[x*10,200],'bbox':[x*10-10,160,20,40],'pitch':[x,20],'projection':{'uncertainty_pitch_m_proxy':uncertainty},'feature':{'quality':.8,'lab':[90,150,120]}})
 return s

def visual(label='TEAM_A',quality=.8,strength=.95):return {'label':label,'quality':quality,'strength':strength}
def person(**kwargs):return PersistentPerson(**({'entity_id':'PLAYER_001','visibility_state':'VISIBLE','identity_status':'PROVISIONAL','identity_confidence':.5,'team':TeamIntelligence(),'camera_epoch':0,'camera_state':'UNKNOWN','camera_source':'test','last_seen_time':0,'visibility_reason':'test'}|kwargs))
class IdentityTests(unittest.TestCase):
 def test_same_player_fragment_reconnects(self):
  a,b=segment('a',0,10),segment('b',30,40,x=22);mapping,_,_,events,_=globally_associate([a,b],FPS);self.assertEqual(mapping['a'],mapping['b']);self.assertEqual(len(events),1)
 def test_plausible_metric_gate(self):self.assertTrue(candidate(segment('a',0,10),segment('b',30,40,x=23),FPS).eligible)
 def test_impossible_metric_rejected(self):
  c=candidate(segment('a',0,10),segment('b',12,20,x=100),FPS);self.assertFalse(c.eligible);self.assertEqual(c.reason,'PHYSICALLY_IMPLAUSIBLE_REASSOCIATION')
 def test_uncertain_geometry_cannot_reject_identity_as_impossible(self):
  c=candidate(segment('a',0,10),segment('b',12,20,x=100,uncertainty=15),FPS);self.assertEqual(c.position_weight,0);self.assertNotEqual(c.reason,'PHYSICALLY_IMPLAUSIBLE_REASSOCIATION')
 def test_unknown_geometry_requires_independent_continuity(self):
  a,b=segment('a',0,10),segment('b',30,40);a.observations[-1]['pitch']=None;self.assertFalse(candidate(a,b,FPS).eligible)
 def test_close_image_gap_can_support_unknown_geometry(self):
  a,b=segment('a',0,10),segment('b',12,20);a.observations[-1]['pitch']=None;self.assertTrue(candidate(a,b,FPS).eligible)
 def test_weak_embedding_rejected(self):self.assertFalse(candidate(segment('a',0,10),segment('b',30,40,embedding=(0,1,0)),FPS).eligible)
 def test_missing_embedding_rejected(self):
  a=segment('a',0,10);a.embedding=None;self.assertFalse(candidate(a,segment('b',30,40),FPS).eligible)
 def test_team_conflict_rejected(self):self.assertEqual(candidate(segment('a',0,10),segment('b',30,40,team='TEAM_B'),FPS).reason,'STRONG_TEAM_CONFLICT')
 def test_unknown_team_does_not_erase_identity_candidate(self):self.assertTrue(candidate(segment('a',0,10),segment('b',30,40,team='UNKNOWN'),FPS).eligible)
 def test_simultaneous_tracks_cannot_merge(self):self.assertFalse(candidate(segment('a',0,20),segment('b',10,30),FPS).eligible)
 def test_global_assignment_one_to_one(self):
  ss=[segment('a',0,10,x=10),segment('b',0,10,x=70,embedding=(0,1,0)),segment('c',30,40,x=12),segment('d',30,40,x=72,embedding=(0,1,0))];m,_,_,e,_=globally_associate(ss,FPS);self.assertEqual(m['a'],m['c']);self.assertEqual(m['b'],m['d']);self.assertNotEqual(m['c'],m['d']);self.assertEqual(len(e),2)
 def test_two_similar_teammates_remain_ambiguous(self):
  m,ev,a,e,_=globally_associate([segment('a',0,10,x=20),segment('b',0,10,x=21),segment('c',30,40,x=20.5)],FPS);self.assertIn('c',a);self.assertEqual(len(e),0);self.assertGreaterEqual(len(ev['c']),2)
 def test_crossing_competitors_do_not_force_assignment(self):
  m,_,a,e,_=globally_associate([segment('a',0,10,x=20),segment('b',0,10,x=20.1),segment('c',30,40,x=20),segment('d',30,40,x=20.1)],FPS);self.assertEqual(len(e),0)
 def test_cut_epoch_prevents_reconnect(self):self.assertFalse(candidate(segment('a',0,10),segment('b',30,40,epoch=1),FPS).eligible)
 def test_stale_reappearance_rejected(self):self.assertFalse(candidate(segment('a',0,10),segment('b',300,310),FPS).eligible)
 def test_delayed_resolution_is_explicit(self):
  _,_,_,events,_=globally_associate([segment('a',0,10),segment('b',30,40)],FPS);self.assertGreater(events[0]['resolved_at_frame'],events[0]['reappeared_at_frame']);self.assertTrue(events[0]['offline_future_evidence'])
 def test_continuous_track_separate_from_entity(self):
  features=[[{'track_id':'t','frame':i,'observation_id':f'{i}:t','feature':{'quality':0}}] for i in range(3)];ss,_=build_segments(features,[{'camera':{'epoch':0}}]*3,{});self.assertEqual(len(ss),1);self.assertEqual(len(ss[0].observations),3)
 def test_fragment_and_epoch_create_separate_segments(self):
  f=[[{'track_id':'t','frame':0,'observation_id':'0:t','feature':{'quality':0}}],[],[],[{'track_id':'t','frame':3,'observation_id':'3:t','feature':{'quality':0}}]];ss,_=build_segments(f,[{'camera':{'epoch':0}}]*4,{});self.assertEqual(len(ss),2)
class TeamTests(unittest.TestCase):
 def establish(self):
  t=PersistentTeam()
  for i in range(8):t.update(visual(),i,i*.16)
  self.assertEqual(t.label,'TEAM_A');return t
 def test_persistent_established_team(self):self.assertEqual(self.establish().snapshot().label,'TEAM_A')
 def test_weak_crop_cannot_erase_team(self):
  t=self.establish();self.assertEqual(t.update(visual('UNKNOWN',0,0),9,2).label,'TEAM_A')
 def test_one_contradiction_cannot_switch(self):
  t=self.establish();self.assertEqual(t.update(visual('TEAM_B'),9,2).label,'TEAM_A');self.assertEqual(t.events[-1]['kind'],'TEAM_SWITCH_CANDIDATE')
 def test_candidate_rejection_recorded(self):
  t=self.establish();t.update(visual('TEAM_B'),9,2);t.update(visual(),10,2.2);self.assertEqual(t.events[-1]['kind'],'TEAM_SWITCH_REJECTED')
 def test_sustained_contradiction_required(self):
  t=self.establish()
  for i in range(10):t.update(visual('TEAM_B'),20+i,3+i*.16)
  self.assertEqual(t.label,'TEAM_B');self.assertTrue(any(e['kind']=='TEAM_SWITCH_ACCEPTED' for e in t.events))
 def test_official_is_not_a_third_team(self):
  t=PersistentTeam()
  for i in range(8):t.update(visual('OFFICIAL'),i,i*.16)
  self.assertEqual(t.label,'OFFICIAL');self.assertAlmostEqual(sum(t.snapshot().probabilities.values()),1)
 def test_unknown_new_player(self):self.assertEqual(PersistentTeam().update(visual('UNKNOWN',0,0),0,0).label,'UNKNOWN')
 def test_independent_sampling(self):
  t=PersistentTeam()
  for i in range(10):t.update(visual(),i,i*.001)
  self.assertEqual(t.samples,1);self.assertEqual(t.label,'UNKNOWN')
 def test_foreground_keeps_dark_and_white_pixels(self):
  image=np.zeros((100,80,3),np.uint8);image[:]=[20,140,20];image[15:65,25:45]=[30,30,30];f,mask=foreground(image,[20,0,30,80]);self.assertGreater(f['quality'],0);self.assertLess(f['lab'][0],100)
 def test_number_needs_repeated_good_evidence(self):
  j=JerseyBelief();j.update('7',.9,.8,1);self.assertIsNone(j.snapshot()['number']);j.update('7',.9,.8,2);j.update('7',.9,.8,3);self.assertEqual(j.snapshot()['number'],'7')
 def test_bad_ocr_ignored(self):
  j=JerseyBelief()
  for i in range(8):j.update('7',.4,.8,i)
  self.assertIsNone(j.snapshot()['number']);self.assertEqual(j.samples,[])
class InvariantTests(unittest.TestCase):
 def test_off_camera_position_is_unknown(self):
  with self.assertRaises(ValidationError):person(current_position=(20,20)).model_copy(update={'visibility_state':'OFF_CAMERA'}) and PersistentPerson.model_validate({**person(current_position=(20,20)).model_dump(),'visibility_state':'OFF_CAMERA'})
 def test_separate_confidence_channels(self):
  p=person(identity_confidence=.9,current_position=None,team=TeamIntelligence(label='TEAM_A',confidence=.95));self.assertEqual(p.identity_confidence,.9);self.assertIsNone(p.position_confidence)
 def test_known_position_does_not_imply_identity(self):
  p=person(current_position=(20,20),position_confidence=.9,position_provenance='DIRECT_PROJECTED');self.assertEqual(p.identity_status,'PROVISIONAL')
 def test_duplicate_visible_track_rejected(self):
  a=person(current_track_id='t');b=a.model_copy(update={'entity_id':'PLAYER_002'})
  with self.assertRaises(ValidationError):PlayerWorldFrame(frame=0,timestamp=0,players=[a,b])
 def test_duplicate_entity_rejected(self):
  with self.assertRaises(ValidationError):PlayerWorldFrame(frame=0,timestamp=0,players=[person(),person()])
 def test_unknown_position_has_no_fabricated_confidence(self):
  with self.assertRaises(ValidationError):person(position_confidence=.8)

class VisibilityTests(unittest.TestCase):
 def test_camera_pan_off_player(self):
  from src.vision.world.player_identity import absence_state
  c={'epoch':0,'calibration_valid':True,'visible_pitch_region':[[40,0],[80,0],[80,68],[40,68]]};self.assertEqual(absence_state(c,[20,20],.5,0,0,1)[0],'OFF_CAMERA')
 def test_camera_pan_back_missing_detection_is_occlusion_hypothesis(self):
  from src.vision.world.player_identity import absence_state
  c={'epoch':0,'calibration_valid':True,'visible_pitch_region':[[0,0],[80,0],[80,68],[0,68]]};self.assertEqual(absence_state(c,[20,20],.5,0,0,1)[0],'OCCLUDED')
 def test_unknown_camera_does_not_establish_off_camera(self):
  from src.vision.world.player_identity import absence_state
  self.assertEqual(absence_state({'epoch':0,'calibration_valid':False},[20,20],1,0,0,1)[0],'OCCLUDED')
 def test_long_gap_is_lost_without_exact_prediction(self):
  from src.vision.world.player_identity import absence_state
  self.assertEqual(absence_state({'epoch':0,'calibration_valid':False},[20,20],1,0,0,20)[0],'LOST')
 def test_epoch_cut_is_lost(self):
  from src.vision.world.player_identity import absence_state
  self.assertEqual(absence_state({'epoch':1,'calibration_valid':True},[20,20],1,0,0,1)[0],'LOST')

class LongGapTests(unittest.TestCase):
 def test_long_gap_appearance_is_not_confident_identity(self):
  a=segment('a',0,10);b=segment('b',190,200);c=candidate(a,b,FPS);self.assertLess(c.score,.78)
 def test_weak_metric_weight_is_lower_than_reliable_metric(self):
  a,b=segment('a',0,10,uncertainty=4),segment('b',30,40,uncertainty=4)
  a.evidence_frames=[0,3,6,9];b.evidence_frames=[30,33,36,39];c=candidate(a,b,FPS);self.assertGreater(c.position_weight,0);self.assertLess(c.position_weight,.35)

class EvidenceSafetyTests(unittest.TestCase):
 def test_moderate_contradiction_cannot_switch_established_team(self):
  t=TeamTests().establish()
  for i in range(30):t.update(visual('TEAM_B',.3,.4),30+i,3+i*.16)
  self.assertEqual(t.label,'TEAM_A');self.assertFalse(any(e['kind']=='TEAM_SWITCH_ACCEPTED' for e in t.events))
 def test_official_population_is_learned_separately(self):
  ss=[]
  for i,lab in enumerate([[80,130,135]]*3+[[160,180,150]]*3+[[230,95,210]]):
   s=segment(str(i),0,19 if i<6 else 9,x=50)
   for o in s.observations:o['feature']['lab']=lab
   ss.append(s)
  m=TeamAppearance.fit(ss);self.assertIn('OFFICIAL',m.labels);self.assertEqual(set(m.labels),{'TEAM_A','TEAM_B','OFFICIAL'})
 def test_duplicate_unresolved_and_persistent_track_rejected(self):
  from src.vision.world.player_state import UnresolvedPlayerDetection
  u=UnresolvedPlayerDetection(observation_id='1:t',track_id='t',bbox=(0,0,10,20),team=TeamIntelligence())
  with self.assertRaises(ValidationError):PlayerWorldFrame(frame=1,timestamp=1,players=[person(current_track_id='t')],unresolved_detections=[u])
 def test_checkpoint_is_validated_before_neural_loading(self):
  import tempfile
  from pathlib import Path
  from src.vision.world.player_appearance import BodyEncoder
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'bad.pth';p.write_bytes(b'corrupt')
   with self.assertRaisesRegex(ValueError,'hash mismatch'):BodyEncoder(p)

class WeakTeamIdentityTests(unittest.TestCase):
 def test_tiny_team_evidence_cannot_hard_gate_reid(self):
  s=segment('a',0,2)
  for o in s.observations:o['visual_team']=visual('TEAM_B',.2,.2)
  s.team_summary();self.assertLess(s.team_strength,.8)

class InputLineageTests(unittest.TestCase):
 def test_wrong_recording_is_rejected_before_features_or_model_loading(self):
  import tempfile,json
  from pathlib import Path
  from src.vision.world.player_replay import run
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'manifest.json').write_text(json.dumps({'recording_id':'other','camera_model_version':'m8'}))
   with self.assertRaisesRegex(ValueError,'matching recording'):run('wanted',p/'output',baseline=p)

if __name__=='__main__':unittest.main()
