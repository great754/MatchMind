"""Known-geometry temporal/kit invariants; no model inference or cloud access."""
import unittest
import cv2,numpy as np
from src.vision.world.temporal_camera import TemporalCameraFilter,transform,compatible_path,strongly_observed
from src.vision.world.static_geometry import line_measurements,measure
from src.vision.world.field_motion import field_image,validate_motion
from src.vision.world.persistent_camera import PersistentCamera
from src.vision.world.pitch_geometry import PitchGeometry
from src.vision.world.schema import CameraState,KitObservation,PlayerEntity
from src.vision.world.team_belief import TeamBelief
from src.vision.world.player_geometry import project_player
from src.vision.world.temporal_review import position_diagnostic

W,HEIGHT,FPS=1050,680,25
H=np.diag([.1,.1,1.]);G=PitchGeometry()
def groups(h=H,names=('Circle central','Middle line','Side line top')):
 out={}
 for name in names:
  if name in G.segments:
   a,b=map(np.array,G.segments[name]);p=a+(b-a)*np.linspace(0,1,80)[:,None]
  else:p=G.curves()[name]
  out[name]=cv2.perspectiveTransform(p[None].astype(float),np.linalg.inv(h))[0]/[W,HEIGHT]
 return out

def camera(i,h=H,valid=True,epoch=0,cut=False):
 base=CameraState(frame=i,timestamp=i/FPS,epoch=epoch,continuity='cut' if cut else 'continuous')
 return PersistentCamera(W,HEIGHT).update(base,groups(h),{'homography':h.tolist()} if valid else None,{}, {'accepted':False,'reason':'NO_PRIOR'})

def motion(dx=0,dy=0,scale=1):return {'accepted':True,'homography':[[scale,0,dx],[0,scale,dy],[0,0,1]],'heldout_p90_small_px':.1,'scale_to_source':1,'matches':80,'reason':'VALIDATED_PITCH_FEATURE_MOTION'}

class TemporalCameraTests(unittest.TestCase):
 def setUp(self):self.f=TemporalCameraFilter(W,HEIGHT,FPS)
 def test_current_motion_budget_is_not_inherited_from_baseline(self):
  rows=[camera(0),camera(1,valid=False)];out,d=self.f.run(rows,[groups(),groups()],[motion(),motion()],[motion(),motion()]);self.assertAlmostEqual(out[1].motion_evidence['localization']['relative_motion_error_budget_px'],.1);self.assertEqual(out[1].motion_evidence['localization']['relative_motion_error_budget_px'],out[1].motion_evidence['temporal_camera']['relative_motion_error_budget_px'])
 def test_forward_pan_updates_camera_not_player(self):
  h=H@np.linalg.inv(np.array(motion(2)['homography']));out,d=self.f.run([camera(0),camera(1,h)],[groups(),groups(h)],[motion(),motion(2)],[motion(-2),motion()]);self.assertEqual(out[1].motion_evidence['localization']['status'],'TEMPORAL_CORRECTED');np.testing.assert_allclose(project_player((502,300),out[1])[0],(50,30),atol=.02)
 def test_tilt_and_small_zoom_are_current_measurements(self):
  for m in (motion(0,2),motion(0,0,1.005)):
   h=H@np.linalg.inv(np.array(m['homography']));out,_=self.f.run([camera(0),camera(1,h)],[groups(),groups(h)],[motion(),m],[motion(),motion()]);self.assertTrue(out[1].calibration_valid)
 def test_backward_boundary_prediction(self):
  cameras=[camera(0,valid=False),camera(1)];out,attempts=self.f.fill_short_gaps(cameras,[groups(),groups()],[motion(),motion()],[motion(),motion()],list(cameras));self.assertEqual(out[0].motion_evidence['localization']['status'],'TEMPORAL_PREDICTED');self.assertEqual(out[0].motion_evidence['temporal_camera']['direction'],'backward')
 def test_bidirectional_agreement(self):
  rows=[camera(0),camera(1,valid=False),camera(2)];out,attempts=self.f.fill_short_gaps(rows,[groups()]*3,[motion()]*3,[motion()]*3,list(rows));self.assertEqual(out[1].motion_evidence['localization']['status'],'BIDIRECTIONAL_INTERPOLATED');self.assertTrue(attempts[0]['accepted'])
 def test_bidirectional_disagreement_keeps_unknown(self):
  h=H.copy();h[0,2]=2.;rows=[camera(0),camera(1,valid=False),camera(2,h)];gs=[groups(),groups(),groups(h)];out,attempts=self.f.fill_short_gaps(rows,gs,[motion()]*3,[motion()]*3,list(rows));self.assertFalse(out[1].calibration_valid);self.assertEqual(attempts[0]['reason'],'BIDIRECTIONAL_DISAGREEMENT')
 def test_long_gap_never_filled(self):
  rows=[camera(i,valid=i in (0,4)) for i in range(5)];out,attempts=self.f.fill_short_gaps(rows,[groups()]*5,[motion()]*5,[motion()]*5,list(rows));self.assertEqual(attempts,[]);self.assertFalse(out[2].calibration_valid)
 def test_no_gap_across_cut_or_epoch(self):
  for cut,epoch in ((True,1),(False,2)):
   rows=[camera(0),camera(1,valid=False,epoch=epoch,cut=cut),camera(2,epoch=epoch)];out,attempts=self.f.fill_short_gaps(rows,[groups()]*3,[motion()]*3,[motion()]*3,list(rows));self.assertFalse(out[1].calibration_valid);self.assertFalse(compatible_path(rows,0,2,FPS))
 def test_two_measured_directions_can_bridge_semantic_dropout(self):
  rows=[camera(0),camera(1,valid=False),camera(2)];out,_=self.f.fill_short_gaps(rows,[groups(),{},groups()],[motion()]*3,[motion()]*3,list(rows));self.assertEqual(out[1].motion_evidence['localization']['status'],'BIDIRECTIONAL_INTERPOLATED');self.assertFalse(out[1].motion_evidence['localization']['current_semantic_support'])
 def test_no_measurement_no_interpolation(self):
  rows=[camera(0),camera(1,valid=False),camera(2)];bad=[{'accepted':False,'reason':'ZOOM_UNSUPPORTED'}]*3;out,_=self.f.fill_short_gaps(rows,[groups()]*3,bad,bad,list(rows));self.assertFalse(out[1].calibration_valid)
 def test_temporal_geometry_inconsistency(self):
  wrong=H.copy();wrong[0,2]=12;rows=[camera(0),camera(1,wrong)];out,diag=self.f.run(rows,[groups(),groups()],[motion(),motion()],[motion(),motion()]);self.assertIn(1,diag['geometry_inconsistent_frames']);self.assertTrue(out[1].motion_evidence['temporal_camera']['geometry_inconsistent']);np.testing.assert_allclose(project_player((500,300),out[1])[0],(50,30),atol=.1)
 def test_one_line_never_independent_anchor(self):self.assertFalse(strongly_observed(H,groups(names=('Side line top',)),W,HEIGHT)[0])
 def test_propagated_raw_pose_cannot_become_independent_anchor(self):
  raw=camera(0).model_copy(update={'calibration_source_kind':'partial_constrained'});out,d=self.f.run([raw],[groups()],[motion()],[motion()]);self.assertFalse(out[0].calibration_valid)
 def test_full_center_and_midfield_is_observed(self):self.assertTrue(strongly_observed(H,groups(),W,HEIGHT)[0])
 def test_named_touchline_and_multiple_line_motion(self):
  m=np.array(motion(2,1)['homography']);h=H@np.linalg.inv(m);a=groups();b=groups(h);lines=line_measurements(a,b,m,W,HEIGHT);self.assertEqual(len(lines),2);self.assertTrue(all(l['accepted'] for l in lines));self.assertTrue(any(l['kind']=='Side line top' for l in lines))
 def test_wrong_line_motion_rejected(self):
  lines=line_measurements(groups(),groups(),np.array(motion(0,20)['homography']),W,HEIGHT);self.assertFalse(next(l['accepted'] for l in lines if l['kind']=='Side line top'))
 def test_ransac_outliers_not_accepted_as_all_inliers(self):
  rng=np.random.default_rng(8);a=rng.uniform([20,20],[620,620],(160,2)).astype(np.float32);b=a+[2,0];b[1::13]+=[30,30];field=(np.zeros((640,640),np.uint8),np.full((640,640),255,np.uint8),1.);r=validate_motion(a,b,field,field);self.assertLess(r.get('inlier_ratio',1),1)
 def test_unsupported_zoom_rejected_by_measurement(self):
  rng=np.random.default_rng(7);a=rng.uniform([20,20],[620,620],(160,2)).astype(np.float32);field=(np.zeros((640,640),np.uint8),np.full((640,640),255,np.uint8),1.);self.assertFalse(validate_motion(a,a*1.1,field,field)['accepted'])

 def test_two_frame_gap_with_two_directions(self):
  rows=[camera(i,valid=i in (0,3)) for i in range(4)];out,a=self.f.fill_short_gaps(rows,[groups()]*4,[motion()]*4,[motion()]*4,list(rows));self.assertTrue(all(r.calibration_valid for r in out));self.assertTrue(all(t['accepted'] for t in a))
 def test_one_sided_boundary_without_semantics_stays_unknown(self):
  rows=[camera(0,valid=False),camera(1)];out,a=self.f.fill_short_gaps(rows,[{},groups()],[motion()]*2,[motion()]*2,list(rows));self.assertFalse(out[0].calibration_valid)
 def test_epoch_reset_clears_old_absolute_age(self):
  rows=[camera(0),camera(1,valid=False,epoch=1,cut=True)];out,d=self.f.run(rows,[groups(),{}],[motion(),{'accepted':False,'reason':'HARD_CUT'}],[motion()]*2);self.assertIsNone(out[1].motion_evidence['localization']['last_absolute_age_seconds']);self.assertIsNone(out[1].memory.last_direct)

class PositionTests(unittest.TestCase):
 def entity(self,foot,pitch):return PlayerEntity(entity_id='e',track_id='t',visibility='visible',last_seen=0,last_seen_frame=0,image_foot=foot,current_pitch=pitch)
 def test_legitimate_movement_preserved(self):
  old=self.entity((500,300),(50,30));current=self.entity((503,300),(50.3,30));d=position_diagnostic(old,current,(50.3,30),camera(1),FPS);self.assertFalse(d['large_jump']);self.assertAlmostEqual(d['distance_m'],.3);self.assertEqual(current.current_pitch,(50.3,30))
 def test_impossible_jump_decomposes_foot_error(self):
  old=self.entity((500,300),(50,30));current=self.entity((600,300),(60,30));d=position_diagnostic(old,current,(60,30),camera(1),FPS);self.assertTrue(d['large_jump']);self.assertAlmostEqual(d['camera_component_m'],0);self.assertAlmostEqual(d['changed_foot_component_m'],10)
 def test_camera_error_decomposes_without_smoothing(self):
  old=self.entity((500,300),(50,30));current=self.entity((500,300),(60,30));h=H.copy();h[0,2]=10;d=position_diagnostic(old,current,(60,30),camera(1,h),FPS);self.assertTrue(d['large_jump']);self.assertAlmostEqual(d['camera_component_m'],10);self.assertAlmostEqual(d['changed_foot_component_m'],0)
 def test_unknown_position_has_no_invented_jump(self):self.assertIsNone(position_diagnostic(self.entity((500,300),(50,30)),self.entity((500,300),None),None,camera(1,valid=False),FPS))

class TeamBeliefTests(unittest.TestCase):
 def kit(self,q=.9):return KitObservation(lab=(100,150,150),quality=q,foreground_fraction=.8,field_context=.8)
 def feed(self,b,team,i,q=.9):return b.update(team,.95,self.kit(q),i,i*.1)
 def learned(self,team):
  b=TeamBelief()
  for i in range(8):self.feed(b,team,i)
  self.assertEqual(b.team,team);return b
 def test_strong_a_bad_crop_retains_a(self):
  b=self.learned('team_a');r,_=self.feed(b,None,9,.05);self.assertEqual(r['current_team'],'team_a');self.assertEqual(r['source'],'temporal_memory');self.assertFalse(r['current_strong'])
 def test_strong_b_bad_crop_retains_b(self):
  b=self.learned('team_b');r,_=self.feed(b,None,9,.05);self.assertEqual(r['current_team'],'team_b')
 def test_isolated_contradiction_does_not_switch(self):
  b=self.learned('team_a');self.feed(b,'team_b',9);self.assertEqual(b.team,'team_a')
 def test_sustained_strong_evidence_switches_with_audit(self):
  b=self.learned('team_a')
  for i in range(9,100):self.feed(b,'team_b',i)
  self.assertEqual(b.team,'team_b');switch=[e for e in b.switches if e['kind']=='team_changed'];self.assertEqual(len(switch),1);self.assertGreaterEqual(switch[0]['supporting_observations'],6);self.assertGreaterEqual(switch[0]['candidate_span_seconds'],.4)
 def test_weak_contradiction_never_switches(self):
  b=self.learned('team_a')
  for i in range(9,100):self.feed(b,'team_b',i,.1)
  self.assertEqual(b.team,'team_a')
 def test_samples_need_temporal_independence(self):
  b=TeamBelief()
  for i in range(20):b.update('team_a',.95,self.kit(),i,i*.001)
  self.assertEqual(b.samples,1);self.assertIsNone(b.team)
 def test_identity_unknown_not_forced(self):
  b=TeamBelief();r,_=b.update(None,None,None,0,0,False);self.assertIsNone(r['current_team']);self.assertFalse(r['identity_resolved']);self.assertEqual(r['unknown_reason'],'TEAM_UNKNOWN_NO_EVIDENCE')
 def test_short_occlusion_retains_history(self):
  b=self.learned('team_a');old=b.samples;r,_=b.update(None,None,None,9,.9);self.assertEqual(r['current_team'],'team_a');self.assertEqual(b.samples,old)
 def test_ambiguous_color_is_distinct(self):
  b=TeamBelief();r,_=b.update(None,None,self.kit(),0,0);self.assertEqual(r['unknown_reason'],'TEAM_UNKNOWN_AMBIGUOUS_COLOR')
 def test_low_quality_is_distinct(self):
  b=TeamBelief();r,_=b.update('team_a',.9,self.kit(.1),0,0);self.assertEqual(r['unknown_reason'],'TEAM_UNKNOWN_LOW_QUALITY_CROP')
