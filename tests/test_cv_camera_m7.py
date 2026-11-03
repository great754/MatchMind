"""Reference-evaluated synthetic field-camera behavior, not real-video accuracy."""
import unittest
import cv2,numpy as np
from src.vision.world.world_field_camera import WorldFieldTracks,WorldFieldCamera
from tests.test_cv_temporal_m6 import camera,groups,motion,H,W,HEIGHT,FPS

class PersistentWorldTests(unittest.TestCase):
 def setUp(self):
  self.rng=np.random.default_rng(17);self.points=self.rng.uniform([80,60],[950,610],(160,2));self.field=(np.zeros((HEIGHT,W),np.uint8),np.full((HEIGHT,W),255,np.uint8),1.);self.f=WorldFieldCamera(W,HEIGHT,FPS)
 def measurements(self,points=None):return {'ids':np.arange(160),'points':self.points.copy() if points is None else points.copy()}
 def trajectory(self,n=80,dx=1,zoom=1,names=None,missing=False,reverse=False,noise=0):
  poses=[];gs=[];ms=[];measurements=[]
  for i in range(n):
   m=np.array([[zoom**i,0,dx*i],[0,zoom**i,0],[0,0,1.]]);h=H@np.linalg.inv(m);poses.append(camera(i,h,valid=(i==n-1 if reverse else i==0)));gs.append({} if missing and i else groups(h,names=names) if names else groups(h));ms.append(motion(-dx if reverse else dx,scale=1/zoom if reverse else zoom));p=cv2.perspectiveTransform(self.points[None],m)[0]+self.rng.normal(0,noise,self.points.shape);measurements.append(self.measurements(p))
  out=self.f.directional(poses,gs,ms,[self.field]*n,measurements,reverse);return poses,gs,ms,measurements,out
 def test_long_constant_pan_and_multisecond_continuation(self):
  *_,out=self.trajectory();self.assertTrue(all(c.calibration_valid for c in out.values()));np.testing.assert_allclose(out[79].calibration.image_to_pitch,np.ravel(H@np.linalg.inv(np.array([[1,0,79],[0,1,0],[0,0,1.]]))),atol=1e-4)
 def test_pan_and_zoom(self):
  *_,out=self.trajectory(60,zoom=1.001);self.assertTrue(out[59].calibration_valid)
 def test_line_disappearance_world_features_continue(self):
  *_,out=self.trajectory(missing=True);self.assertTrue(out[79].calibration_valid);self.assertGreater(out[79].motion_evidence['temporal_camera']['world_field']['active_world_anchors'],40)
 def test_top_touchline_only_support(self):
  *_,out=self.trajectory(names=('Side line top',));self.assertTrue(out[79].calibration_valid)
 def test_circle_temporal_support_not_absolute_seed(self):
  poses,gs,ms,meas,out=self.trajectory(names=('Circle central',));self.assertTrue(out[79].calibration_valid)
  no_anchor=[c.model_copy(update={'calibration_valid':False,'calibration':None}) for c in poses];out=self.f.directional(no_anchor,gs,ms,[self.field]*80,meas);self.assertFalse(any(c.calibration_valid for c in out.values()))
 def test_fixed_world_coordinates_and_feature_lifetime(self):
  t=WorldFieldTracks(W,HEIGHT);current=t.observe(0,self.measurements(),1);t.seed(0,current,H,0,0,0,self.field);old=t.tracks[2]['world_coordinate'].copy();t.observe(1,self.measurements(self.points+[3,0]),1);self.assertEqual(old,t.tracks[2]['world_coordinate']);self.assertEqual(t.tracks[2]['observations'],2)
 def test_anchor_loss_expires_tracks(self):
  t=WorldFieldTracks(W,HEIGHT);c=t.observe(0,self.measurements(),1);t.seed(0,c,H,0,0,0,self.field);t.observe(1,None,1);self.assertFalse(t.tracks)
 def test_contamination_rejection(self):
  t=WorldFieldTracks(W,HEIGHT);c=t.observe(0,self.measurements(),1);t.seed(0,c,H,0,0,0,self.field);bad=self.points.copy();bad[:80]+=self.rng.uniform(-80,80,(80,2));current=t.observe(1,self.measurements(bad),1);h,d=t.fit(current,self.field);self.assertIsNone(h)
 def test_backward_measured_trajectory(self):
  *_,out=self.trajectory(reverse=True);self.assertTrue(out[0].calibration_valid);self.assertEqual(out[0].motion_evidence['temporal_camera']['direction'],'backward')
 def test_forward_backward_long_gap_convergence(self):
  poses,gs,ms,meas,_=self.trajectory(n=30);poses[-1]=camera(29,H@np.linalg.inv(np.array([[1,0,29],[0,1,0],[0,0,1.]])));out,d=self.f.run(poses,gs,ms,[motion(-1)]*30,[self.field]*30,meas,meas);self.assertTrue(all(c.calibration_valid for c in out));self.assertEqual(out[15].motion_evidence['localization']['status'],'BIDIRECTIONAL_INTERPOLATED')
 def test_uncertainty_grows_without_reobserved_world_tracks(self):
  poses,gs,ms,meas,_=self.trajectory(n=20);out=self.f.directional(poses,gs,ms,[self.field]*20,[None]*20);a=out[1].motion_evidence['temporal_camera']['trajectory_uncertainty_px'];b=out[19].motion_evidence['temporal_camera']['trajectory_uncertainty_px'];self.assertGreater(b,a)
 def test_uncertainty_recovers_on_world_reobservation(self):
  *_,out=self.trajectory(n=20);self.assertLess(out[19].motion_evidence['temporal_camera']['trajectory_uncertainty_px'],.1)
 def test_no_evidence_rejects(self):
  poses,gs,ms,meas,_=self.trajectory(n=5);ms[1]={'accepted':False,'reason':'NO_FEATURES'};out=self.f.directional(poses,gs,ms,[self.field]*5,[None]*5);self.assertFalse(out[1].calibration_valid)
 def test_hard_cut_reset(self):
  poses,gs,ms,meas,_=self.trajectory(n=5);poses[2]=poses[2].model_copy(update={'epoch':1,'continuity':'cut'});out=self.f.directional(poses,gs,ms,[self.field]*5,meas);self.assertFalse(out[2].calibration_valid)
 def test_independent_reacquisition_corrects_small_drift(self):
  poses,gs,ms,meas,_=self.trajectory(n=10);fresh=H@np.linalg.inv(np.array([[1,0,9.5],[0,1,0],[0,0,1.]]));poses[9]=camera(9,fresh);gs[9]=groups(fresh);out=self.f.directional(poses,gs,ms,[self.field]*10,meas);self.assertEqual(out[9].motion_evidence['localization']['status'],'RELOCALIZED');np.testing.assert_allclose(out[9].calibration.image_to_pitch,fresh.flat,atol=1e-6)

 def test_zoom_reset_drops_world_anchors(self):
  poses,gs,ms,meas,_=self.trajectory(n=5);ms[2]={'accepted':False,'reason':'ZOOM_UNSUPPORTED'};out=self.f.directional(poses,gs,ms,[self.field]*5,meas);self.assertFalse(out[2].calibration_valid);self.assertEqual(out[2].motion_evidence['temporal_camera']['world_field']['active_world_anchors'],0)
 def test_uncertainty_budget_rejects_even_with_current_lines(self):
  poses,gs,ms,meas,_=self.trajectory(n=8)
  for m in ms:m['heldout_p90_small_px']=10
  out=self.f.directional(poses,gs,ms,[self.field]*8,[None]*8);self.assertFalse(out[6].calibration_valid)
 def test_two_sided_disagreement_is_not_averaged(self):
  poses,gs,ms,meas,_=self.trajectory(n=7);wrong=H.copy();wrong[0,2]=4;poses[-1]=camera(6,wrong);gs=[{}]*7;out,d=self.f.run(poses,gs,ms,[motion(-1)]*7,[self.field]*7,meas,meas);self.assertFalse(out[3].calibration_valid);self.assertEqual(out[3].source,'BIDIRECTIONAL_DISAGREEMENT')

 def test_unknown_frames_do_not_mint_world_anchors(self):
  poses,gs,ms,meas,_=self.trajectory(n=5);poses=[c.model_copy(update={'calibration_valid':False,'calibration':None}) for c in poses];out=self.f.directional(poses,gs,ms,[self.field]*5,meas);self.assertTrue(all(c.motion_evidence['temporal_camera']['new_world_anchors']==0 for c in out.values()))
 def test_large_zoom_cannot_bypass_relative_motion_gate_via_world_fit(self):
  poses,gs,ms,meas,_=self.trajectory(n=3);meas[1]=self.measurements(self.points*1.1);ms[1]={'accepted':False,'reason':'FIELD_MOTION_VALIDATION_FAILED'};out=self.f.directional(poses,gs,ms,[self.field]*3,meas);self.assertFalse(out[1].calibration_valid)
 def test_measured_line_name_migration_is_ambiguity_not_new_absolute_evidence(self):
  from src.vision.world.static_geometry import semantic_line_migrations
  old=groups(names=('Side line top',));new={'Big rect. left top':old['Side line top'].copy()};m=motion();m['inlier_ratio']=1.;found=semantic_line_migrations(old,new,m,W,HEIGHT,1.);self.assertIn('Big rect. left top',found);self.assertFalse(found['Big rect. left top']['independent_evidence'])
 def test_preserved_m6_independent_anchor_is_not_current_partial_transform(self):
  raw=camera(0);partial=H.copy();partial[0,2]=.2
  raw=raw.model_copy(update={'calibration_source_kind':'partial_constrained','calibration':raw.calibration.model_copy(update={'image_to_pitch':tuple(partial.flat)})})
  debug={'groups':groups(partial),'field_motion':motion(),'direction':'forward','absolute_anchor_frame':0,'relative_motion_error_budget_px':1}
  current=self.f.publish(raw,partial,'PARTIAL_CONSTRAINED','preserved M6',debug,0);out=self.f.attach_memory([current],[raw])[0]
  np.testing.assert_allclose(out.memory.last_direct.transform,H.flat);self.assertEqual(out.motion_evidence['temporal_camera']['independent_anchor_proof']['kind'],'M6_PRESERVED_INDEPENDENT');np.testing.assert_allclose(out.calibration.image_to_pitch,partial.flat)

 def test_observed_touchline_masks_advertising_half_plane(self):
  from src.vision.world.static_geometry import pitch_interior_field
  field=(np.zeros((100,200),np.uint8),np.full((100,200),255,np.uint8),1.)
  boundary={'Side line top':np.column_stack([np.linspace(.1,.9,80),np.full(80,.2)])}
  gray,mask,scale=pitch_interior_field(field,boundary,200,100);self.assertEqual(np.count_nonzero(mask[:20]),0);self.assertGreater(np.count_nonzero(mask[30:]),0)
