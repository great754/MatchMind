"""Known-geometry M8 camera tests; no real-video accuracy claims."""
import unittest
import cv2,numpy as np
from src.vision.world.metric_motion import metric_update,MetricTrajectory,project
from src.vision.world.metric_landmarks import conic_fit,solve_midfield,pixel_statistics
from src.vision.world.pitch_geometry import PitchGeometry

W,H=1000,680
BASE=np.diag([.105,.1,1.]);G=PitchGeometry()
def groups(h=BASE,names=('Circle central','Middle line','Side line top')):
 result={}
 for name in names:
  if name in G.segments:
   a,b=np.array(G.segments[name]);world=a+(b-a)*np.linspace(0,1,180)[:,None]
  else:world=G.curves()[name]
  result[name]=project(np.linalg.inv(h),world)/[W,H]
 return result

def moved(dx=0,dy=0,zoom=1):
 m=np.array([[zoom,0,W/2*(1-zoom)+dx],[0,zoom,H/2*(1-zoom)+dy],[0,0,1.]])
 return BASE@np.linalg.inv(m)

class MetricMotionTests(unittest.TestCase):
 def test_known_pan(self):np.testing.assert_allclose(metric_update(BASE,moved(-2),W,H)['incremental_view_shift_m'],[.21,0],atol=1e-8)
 def test_opposite_pan(self):self.assertLess(metric_update(BASE,moved(2),W,H)['incremental_view_shift_m'][0],0)
 def test_pan_accumulates_through_projective_maps(self):
  t=MetricTrajectory()
  for i in range(10):d=t.update(i,0,moved(-i),W,H)
  np.testing.assert_allclose(d['accumulated_view_shift_m'],[.945,0],atol=1e-8)
 def test_perspective_scale_is_local(self):
  h=np.array([[.1,0,0],[0,.1,0],[0,.001,1.]])
  from src.vision.world.metric_motion import jacobians
  j=jacobians(h,[[100,50],[100,500]]);self.assertGreater(abs(j[0,0,0]-j[1,0,0]),.02)
 def test_zoom_is_not_lateral_translation(self):
  d=metric_update(BASE,moved(zoom=1.01),W,H);np.testing.assert_allclose(d['incremental_view_shift_m'],[0,0],atol=1e-8);self.assertGreater(d['zoom_singular_values'][0],1)
 def test_pan_zoom(self):self.assertGreater(metric_update(BASE,moved(-2,zoom=1.01),W,H)['incremental_view_shift_m'][0],0)
 def test_vertical_framing(self):self.assertGreater(metric_update(BASE,moved(0,-2),W,H)['vertical_framing_shift_m'],0)
 def test_uncertainty_grows_with_evidence_budget(self):self.assertGreater(metric_update(BASE,BASE,W,H,pixel_budget=4)['uncertainty_pitch_m_proxy'],metric_update(BASE,BASE,W,H,pixel_budget=1)['uncertainty_pitch_m_proxy'])
 def test_landmark_budget_reduces_uncertainty(self):self.assertLess(metric_update(BASE,moved(),W,H,pixel_budget=.2)['uncertainty_pitch_m_proxy'],metric_update(BASE,moved(),W,H,pixel_budget=10)['uncertainty_pitch_m_proxy'])
 def test_cut_resets_accumulated_path(self):
  t=MetricTrajectory();t.update(0,0,BASE,W,H);t.update(1,0,moved(-10),W,H);d=t.update(2,1,BASE,W,H,cut=True);self.assertEqual(d['accumulated_view_shift_m'],[0,0])
 def test_unknown_has_no_stale_current_transform(self):
  t=MetricTrajectory();t.update(0,0,BASE,W,H);d=t.update(1,0,None,W,H);self.assertIsNone(d['current_transform']);self.assertIsNone(d['accumulated_view_shift_m']);d=t.update(2,0,moved(-30),W,H);self.assertEqual(d['accumulated_view_shift_m'],[0,0])
 def test_static_field_point_is_constant_after_camera_pan(self):
  h=moved(-2);m=np.linalg.inv(h)@BASE;a=[500,300];b=project(m,[a])[0];d=metric_update(BASE,h,W,H,{'vectors':[{'previous':a,'current':b.tolist(),'inlier':True}]});self.assertLess(d['metric_field_correspondences'][0]['fixed_point_residual_m'],1e-8)
 def test_footpoint_uses_current_mapping(self):np.testing.assert_allclose(project(moved(-2),[[498,300]])[0],[52.5,30],atol=1e-8)

class LandmarkTests(unittest.TestCase):
 def test_circle_conic_fit(self):
  affine,d=conic_fit(groups()['Circle central'],W,H);self.assertTrue(d['accepted']);self.assertGreater(d['arc_span_degrees'],300)
 def test_no_prior_absolute_relocalization_with_epoch_orientation(self):
  h,d=solve_midfield(groups(),W,H,1);self.assertIsNotNone(h);np.testing.assert_allclose(project(h,[[500,340]]),[[52.5,34]],atol=.02)
 def test_reflection_ambiguity_without_orientation_or_asymmetric_landmark(self):
  h,d=solve_midfield(groups(),W,H);self.assertIsNone(h);self.assertEqual(d['reason'],'UNRESOLVED_LEFT_RIGHT_REFLECTION')
 def test_circle_only_insufficient(self):self.assertIsNone(solve_midfield(groups(names=('Circle central',)),W,H,1)[0])
 def test_touchline_only_insufficient(self):self.assertIsNone(solve_midfield(groups(names=('Side line top',)),W,H,1)[0])
 def test_short_arc_rejected(self):
  arc=G.curves()['Circle central'][:25];raw=project(np.linalg.inv(BASE),arc)/[W,H];self.assertFalse(conic_fit(raw,W,H)[1]['accepted'])
 def test_perspective_conic_relocalization(self):
  h=BASE.copy();h[2,0]=.00008;h[2,1]=.0002
  estimate,d=solve_midfield(groups(h),W,H,1);self.assertIsNotNone(estimate);np.testing.assert_allclose(project(estimate,[[400,300],[600,400]]),project(h,[[400,300],[600,400]]),atol=.05)
 def test_wrong_named_line_not_silently_accepted(self):
  g=groups();g['Middle line']=g['Middle line']+[.2,0];h,d=solve_midfield(g,W,H,1);self.assertIsNone(h)
 def test_penalty_geometry_residuals_use_same_gate(self):
  g=groups(names=('Big rect. left main','Big rect. left top','Side line left','Side line top'));self.assertTrue(pixel_statistics(BASE,g,W,H)[0]);wrong=BASE.copy();wrong[0,2]=20;self.assertFalse(pixel_statistics(wrong,g,W,H)[0])

 def test_penalty_box_absolute_relocalization_without_prior(self):
  from src.vision.world.metric_landmarks import solve_penalty
  h,d=solve_penalty(groups(names=('Big rect. left main','Big rect. left top','Side line left','Side line top')),W,H)
  self.assertIsNotNone(h);np.testing.assert_allclose(project(h,[[100,100],[500,300]]),project(BASE,[[100,100],[500,300]]),atol=.01)
 def test_center_and_asymmetric_penalty_line_resolve_reflection_without_epoch_prior(self):
  h,d=solve_midfield(groups(names=('Circle central','Middle line','Side line top','Big rect. left main')),W,H)
  self.assertIsNotNone(h)

 def test_odd_semantic_pixel_counts_have_separate_holdout_weights(self):
  g=groups();g['Middle line']=g['Middle line'][:65];h,d=solve_midfield(g,W,H,1);self.assertIsNotNone(h)

class FusionTests(unittest.TestCase):
 def test_landmarks_correct_accumulated_metric_drift(self):
  drift=BASE.copy();drift[0,2]=2
  estimate,d=solve_midfield(groups(),W,H,1,initial_h=drift)
  self.assertIsNotNone(estimate);np.testing.assert_allclose(project(estimate,[[400,300]]),project(BASE,[[400,300]]),atol=.02)
 def test_circle_and_measured_pan_prior(self):
  current=moved(-3,zoom=1.002);estimate,d=solve_midfield(groups(current),W,H,1,initial_h=current)
  self.assertIsNotNone(estimate);np.testing.assert_allclose(project(estimate,[[400,300]]),project(current,[[400,300]]),atol=.02)
 def test_landmark_reset_reduces_trajectory_uncertainty(self):
  t=MetricTrajectory();t.update(0,0,BASE,W,H,budget=10);old=t.update(1,0,moved(-2),W,H,budget=12);fresh=t.update(2,0,moved(-4),W,H,budget=.5,anchor=2);self.assertLess(fresh['uncertainty_pitch_m_proxy'],old['uncertainty_pitch_m_proxy'])
 def test_unknown_camera_does_not_use_historical_metric_position(self):
  from src.vision.world.schema import CameraState
  from src.vision.world.player_geometry import project_player
  c=CameraState(frame=1,timestamp=.04,last_reliable_transform=tuple(BASE.flat),last_reliable_timestamp=0)
  self.assertIsNone(project_player((500,300),c)[0])
 def test_teleport_is_diagnostic_not_pose_correction(self):
  from tests.test_cv_temporal_m6 import camera,PositionTests
  from src.vision.world.temporal_review import position_diagnostic
  old=PositionTests().entity((500,300),(50,30));new=PositionTests().entity((500,300),(65,30));h=np.diag([.1,.1,1.]);h[0,2]=15;c=camera(1,h)
  result=position_diagnostic(old,new,(65,30),c,25);self.assertTrue(result['large_jump']);self.assertEqual(new.current_pitch,(65,30))
