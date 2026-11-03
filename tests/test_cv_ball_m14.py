import unittest,json,tempfile,math
from pathlib import Path
import numpy as np
from pydantic import ValidationError
from src.vision.ball.schema import BallCandidate,BallFrame,canonical,VideoBallObservation
from src.vision.ball.tracker import BallTracker,BallTrackerConfig
from src.vision.ball.geometry import camera_motion,plane_intersection,reconstruct

def candidate(x=100,y=100,confidence=.9,name='a',w=8,h=8):return BallCandidate(candidate_id=name,bbox=(x,y,w,h),confidence=confidence)
def camera(epoch=0,cut=False,valid=False,h=None):
 return {'epoch':epoch,'continuity':'cut' if cut else 'continuous','calibration_valid':valid,'calibration':{'image_to_pitch':h or [.1,0,0,0,.1,0,0,0,1],'validity_region':[(0,0),(640,0),(640,360),(0,360)]} if valid else None,'motion_evidence':{'metric_camera':{'uncertainty_pitch_m_proxy':1.}}}
def observed(frame=1,p=None):
 c=p or candidate();return BallFrame(frame=frame,timestamp=frame/25,camera_epoch=0,candidates=[c],candidate_count=1,tracking_status='TRACKED',image_position=c.center,image_bbox=c.bbox,detection_confidence=c.confidence,directly_observed=True)
def acquired(config=None):
 t=BallTracker(640,360,config);t.step(0,0,[candidate()],camera());t.step(1,.04,[candidate(101)],camera());return t

class SchemaTests(unittest.TestCase):
 def test_valid_candidate(self):self.assertEqual(candidate().center,(104,104))
 def test_box_dimensions(self):
  for b in [(0,0,0,2),(0,0,2,-1),(-1,0,2,2)]:
   with self.assertRaises(ValidationError):BallCandidate(candidate_id='a',bbox=b,confidence=.5)
 def test_nonfinite_box_rejected(self):
  with self.assertRaises(ValidationError):candidate(float('nan'))
 def test_confidence_range(self):
  with self.assertRaises(ValidationError):candidate(confidence=1.01)
 def test_candidates_count_exact(self):
  with self.assertRaises(ValidationError):BallFrame(frame=0,timestamp=0,camera_epoch=0,candidates=[candidate()],candidate_count=0)
 def test_prediction_not_current_observation(self):
  with self.assertRaises(ValidationError):BallFrame(frame=0,timestamp=0,camera_epoch=0,image_position=(1,2),tracking_status='PREDICTED')
 def test_observed_requires_retained_neural_candidate(self):
  with self.assertRaises(ValidationError):BallFrame(frame=0,timestamp=0,camera_epoch=0,image_position=(1,2),image_bbox=(0,0,2,4),detection_confidence=.9,directly_observed=True,tracking_status='DETECTED')
 def test_airborne_cannot_supply_metric(self):
  with self.assertRaises(ValidationError):BallFrame(**{**observed().model_dump(),'coordinate_state':'AIRBORNE_SUSPECTED','pitch_position':(1,2),'pitch_position_confidence':.5})
 def test_unknown_height_cannot_supply_metric(self):
  with self.assertRaises(ValidationError):BallFrame(**{**observed().model_dump(),'coordinate_state':'HEIGHT_UNKNOWN','pitch_position':(1,2),'pitch_position_confidence':.5})
 def test_ambiguous_has_no_current_prediction(self):
  with self.assertRaises(ValidationError):BallFrame(frame=0,timestamp=0,camera_epoch=0,tracking_status='AMBIGUOUS',predicted_image_position=(1,2))
 def test_canonical_provenance_and_clock(self):
  r=canonical('clip',25,observed(),'abc');self.assertEqual(r.source,'VIDEO_CV');self.assertEqual(VideoBallObservation.model_validate_json(r.model_dump_json()),r)
 def test_canonical_bad_clock(self):
  r=canonical('clip',25,observed(),'abc').model_dump();r['timestamp']=2
  with self.assertRaises(ValidationError):VideoBallObservation(**r)
 def test_canonical_prediction_separate(self):
  s=BallFrame(frame=2,timestamp=.08,camera_epoch=0,tracking_status='PREDICTED',predicted_image_position=(1,2));r=canonical('clip',25,s,'abc');self.assertIsNone(r.image);self.assertEqual(r.predicted_image,(1,2))

class TrackerTests(unittest.TestCase):
 def test_empty_is_lost(self):self.assertEqual(BallTracker(640,360).step(0,0,[],camera()).tracking_status,'LOST')
 def test_initial_detection_not_auto_confirmed(self):self.assertFalse(BallTracker(640,360).step(0,0,[candidate()],camera()).directly_observed)
 def test_repeated_detection_acquired(self):self.assertEqual(acquired().step(2,.08,[candidate(102)],camera()).tracking_status,'TRACKED')
 def test_lower_confidence_propagates(self):
  s=acquired().step(2,.08,[candidate(102,confidence=.55)],camera());self.assertEqual(s.detection_confidence,.55);self.assertLessEqual(s.tracking_confidence,.55)
 def test_candidates_below_operating_point_not_observed(self):self.assertFalse(BallTracker(640,360).step(0,0,[candidate(confidence=.1)],camera()).directly_observed)
 def test_multiple_repeated_births_stay_ambiguous(self):
  t=BallTracker(640,360);ps=[candidate(),candidate(200,confidence=.99,name='b')];t.step(0,0,ps,camera());s=t.step(1,.04,ps,camera());self.assertEqual(s.tracking_status,'AMBIGUOUS');self.assertIsNone(s.image_position)
 def test_active_competing_candidates_ambiguous(self):
  s=acquired().step(2,.08,[candidate(102),candidate(104,name='b')],camera());self.assertEqual(s.tracking_status,'AMBIGUOUS')
 def test_missing_prediction_separate(self):
  s=acquired().step(2,.08,[],camera());self.assertEqual(s.tracking_status,'PREDICTED');self.assertIsNone(s.image_position);self.assertIsNotNone(s.predicted_image_position)
 def test_prediction_uncertainty_grows(self):
  t=acquired();a=t.step(2,.08,[],camera());b=t.step(3,.12,[],camera());self.assertGreater(b.prediction_uncertainty_px,a.prediction_uncertainty_px)
 def test_prediction_confidence_decays(self):
  t=acquired();a=t.step(2,.08,[],camera());b=t.step(3,.12,[],camera());self.assertLess(b.tracking_confidence,a.tracking_confidence)
 def test_player_overlap_is_occlusion_hypothesis(self):
  s=acquired().step(2,.08,[],camera(),[{'bbox':[90,80,30,60]}]);self.assertEqual(s.tracking_status,'OCCLUDED');self.assertIn('HYPOTHESIS',s.reason)
 def test_short_occlusion_recovers(self):
  t=acquired();t.step(2,.08,[],camera());s=t.step(3,.12,[candidate(103)],camera());self.assertTrue(s.directly_observed);self.assertEqual(s.track_id,1)
 def test_scene_cut_resets(self):
  s=acquired().step(2,.08,[candidate(102)],camera(cut=True));self.assertEqual(s.tracking_status,'LOST');self.assertIsNone(s.image_position)
 def test_camera_epoch_resets(self):
  s=acquired().step(2,.08,[candidate(102)],camera(epoch=1));self.assertEqual(s.tracking_status,'LOST')
 def test_impossible_jump_rejected(self):
  s=acquired().step(2,.08,[candidate(590,300)],camera());self.assertFalse(s.directly_observed);self.assertIn('REJECTED',s.reason)
 def test_kick_can_change_velocity(self):
  s=acquired().step(2,.08,[candidate(125,108)],camera());self.assertTrue(s.directly_observed)
 def test_prediction_expires(self):
  s=acquired().step(12,.48,[],camera());self.assertEqual(s.tracking_status,'LOST');self.assertIsNone(s.predicted_image_position)
 def test_long_gap_does_not_reuse_identity(self):
  t=acquired();t.step(12,.48,[candidate()],camera());s=t.step(13,.52,[candidate()],camera());self.assertEqual(s.track_id,2)
 def test_last_confirmed_is_historical_only(self):
  s=acquired().step(12,.48,[],camera());self.assertIsNotNone(s.last_confirmed_image);self.assertIsNone(s.image_position)
 def test_trail_breaks_on_missing(self):
  t=acquired();t.step(2,.08,[],camera());s=t.step(3,.12,[candidate(102)],camera());self.assertEqual(len(s.trail),1)
 def test_clock_reversal_rejected(self):
  with self.assertRaises(ValueError):acquired().step(2,0,[],camera())
 def test_source_bounds_enforced(self):
  with self.assertRaises(ValueError):BallTracker(640,360).step(0,0,[candidate(639)],camera())

class GeometryTests(unittest.TestCase):
 def test_no_geometry_means_no_pitch(self):self.assertIsNone(plane_intersection((10,10),camera())[0])
 def test_existing_m8_projection_reused(self):self.assertEqual(plane_intersection((100,100),camera(valid=True))[0],(10,10))
 def test_outside_validity_region_rejected(self):self.assertIsNone(plane_intersection((700,100),camera(valid=True))[0])
 def test_height_unknown_keeps_only_plane_hypothesis(self):
  r,_=reconstruct(candidate(),camera(valid=True),[],0);self.assertIsNone(r['pitch_position']);self.assertEqual(r['coordinate_state'],'HEIGHT_UNKNOWN');self.assertIsNotNone(r['plane_hypothesis'])
 def test_repeated_foot_contact_requires_good_camera(self):
  c=candidate(100,100,w=8,h=8);people=[{'bbox':[94,28,20,80]}];r,n=reconstruct(c,camera(valid=True),people,2);self.assertEqual(r['coordinate_state'],'GROUND_SUPPORTED');self.assertLessEqual(r['pitch_position_confidence'],.35);self.assertEqual(n,3)
 def test_unknown_camera_uncertainty_cannot_authorize_plane(self):
  cam=camera(valid=True);cam['motion_evidence']['metric_camera']['uncertainty_pitch_m_proxy']=None;r,_=reconstruct(candidate(),cam,[{'bbox':[94,28,20,80]}],2);self.assertIsNone(r['pitch_position'])
 def test_airborne_suspicion_above_supported_region(self):
  cam=camera(valid=True);cam['calibration']['validity_region']=[(0,100),(640,100),(640,360),(0,360)];r,_=reconstruct(candidate(100,10),cam,[],0);self.assertEqual(r['coordinate_state'],'AIRBORNE_SUSPECTED');self.assertIsNone(r['pitch_position'])
 def test_camera_motion_composition(self):
  before=camera(valid=True);after=camera(valid=True,h=[.1,0,-2,0,.1,0,0,0,1]);h,source=camera_motion(before,after);p=h@np.array([100,100,1]);self.assertAlmostEqual(p[0]/p[2],120)
 def test_camera_cut_disallows_warp(self):self.assertIsNone(camera_motion(camera(valid=True),camera(cut=True,valid=True))[0])
 def test_adjacent_field_motion_reused_without_metric(self):
  after=camera();after['motion_evidence']['field_motion']={'accepted':True,'interval_frames':1,'homography':[[1,0,20],[0,1,0],[0,0,1]]};self.assertIsNotNone(camera_motion(camera(),after)[0])
 def test_nonadjacent_field_motion_not_assumed_current(self):
  after=camera();after['motion_evidence']['field_motion']={'accepted':True,'interval_frames':5,'homography':np.eye(3).tolist()};self.assertIsNone(camera_motion(camera(),after)[0])
 def test_m8_camera_compensation_carries_stationary_ball(self):
  t=BallTracker(640,360);a=camera(valid=True);b=camera(valid=True,h=[.1,0,-2,0,.1,0,0,0,1]);t.step(0,0,[candidate()],a);t.step(1,.04,[candidate(120)],b);s=t.step(2,.08,[candidate(120)],b);self.assertTrue(s.directly_observed)


class SidecarTests(unittest.TestCase):
 def fixture(self,root):
  from src.serving.cv import digest
  states=[BallFrame(frame=0,timestamp=0,camera_epoch=0),observed()];data=root/'ball_frames.jsonl';canon=root/'video_ball_observations.jsonl';offsets=[]
  with data.open('wb') as f:
   for s in states:offsets.append(f.tell());f.write((s.model_dump_json()+'\n').encode())
  canon.write_text(''.join(canonical('clip',25,s,'abc').model_dump_json()+'\n' for s in states));base={'recording_id':'clip','frames':2,'fps':25,'video_sha256':'video','frames_sha256':'m9','width':640,'height':360};manifest={**base,'detector_sha256':'abc','source_m9_frames_sha256':'m9','ball_frames_sha256':digest(data),'canonical_sha256':digest(canon),'frame_offsets':offsets};(root/'manifest.json').write_text(json.dumps(manifest));return base,manifest
 def test_validated_sidecar_queries_exact_index(self):
  from src.serving.ball import BallReview
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);base,_=self.fixture(root);r=BallReview(root,base);self.assertTrue(r.frame(1).directly_observed);self.assertEqual(r.frame(0).tracking_status,'LOST')
 def test_sidecar_rejects_other_video(self):
  from src.serving.ball import BallReview
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);base,_=self.fixture(root)
   with self.assertRaises(ValueError):BallReview(root,{**base,'video_sha256':'other'})
 def test_sidecar_rejects_corruption(self):
  from src.serving.ball import BallReview
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);base,_=self.fixture(root);(root/'ball_frames.jsonl').write_text('{}\n')
   with self.assertRaises(ValueError):BallReview(root,base)
 def test_sidecar_detects_change_after_validation(self):
  from src.serving.ball import BallReview
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);base,_=self.fixture(root);r=BallReview(root,base);p=root/'video_ball_observations.jsonl';p.write_text(p.read_text()+'\n')
   with self.assertRaises(ValueError):r.frame(0)
 def test_sidecar_canonical_mismatch_rejected(self):
  from src.serving.ball import BallReview
  from src.serving.cv import digest
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);base,manifest=self.fixture(root);p=root/'video_ball_observations.jsonl';rows=[json.loads(x) for x in p.read_text().splitlines()];rows[1]['image']=[3,4];p.write_text(''.join(json.dumps(r)+'\n' for r in rows));manifest['canonical_sha256']=digest(p);(root/'manifest.json').write_text(json.dumps(manifest))
   with self.assertRaises(ValueError):BallReview(root,base)
 def test_bad_optional_sidecar_does_not_replace_m9(self):
  from src.serving.cv import CVReview
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);base,_=self.fixture(root);r=object.__new__(CVReview);r.manifest={**base,'video_sha256':'other'};r.attach_ball(root);self.assertIsNone(r.ball_review);self.assertIsNotNone(r.ball_diagnostic)

class OptionalRegistryTests(unittest.TestCase):
 def test_malformed_optional_registry_is_empty(self):
  from src.serving.ball import read_ball_registry
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'registry.json';p.write_text('[]');self.assertEqual(read_ball_registry(p),{});p.write_text('{bad');self.assertEqual(read_ball_registry(p),{})
 def test_registry_keeps_only_valid_paths(self):
  from src.serving.ball import read_ball_registry
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'registry.json';p.write_text(json.dumps({'a':'path','b':None}));self.assertEqual(read_ball_registry(p),{'a':'path'})

if __name__=='__main__':unittest.main()
