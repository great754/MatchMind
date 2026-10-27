"""Deterministic camera tracking and failure provenance; no pretrained inference."""
import unittest
import cv2
import numpy as np
from src.vision.world.localization import CameraLocalizer
from src.vision.world.pitch import PitchEvidence,LineEvidence,TemporalCamera
from src.vision.world.schema import CameraState,TrackObservation
from src.vision.world.engine import WorldModel
from src.vision.world.camera_motion import projective_motion
from src.vision.world.diagnosis import absolute_outcome

class CameraMilestone3Tests(unittest.TestCase):
    def setUp(self):
        self.localizer=CameraLocalizer(1050,680)
        self.fit={'homography':np.diag([.1,.1,1]).tolist(),'image_region':[[0,0],[1050,0],[1050,680],[0,680]],'pitch_region':[[0,0],[105,0],[105,68],[0,68]]}
        self.evidence=PitchEvidence(status='partial',grass_fraction=None,reason='synthetic identified pitch',lines=[LineEvidence(start=(0,0),end=(1050,0),kind='Side line top',support=.9),LineEvidence(start=(0,680),end=(1050,680),kind='Side line bottom',support=.9)])
    def state(self,frame,fit=None,**updates):
        motion={'image_affine':[[1,0,0],[0,1,0]],'feature_matches':100,'affine_inliers':95,'affine_median_error_small_px':.1}
        c=CameraState(frame=frame,timestamp=frame/25,continuity='continuous',motion_evidence=motion).model_copy(update=updates)
        return self.localizer.apply(c,self.evidence,fit)
    def info(self,c):return c.motion_evidence['localization']
    def test_acquire_track_relocalize_and_correct(self):
        a=self.state(0,self.fit);self.assertEqual(self.info(a)['transition'],'acquisition')
        b=self.state(1);self.assertEqual(self.info(b)['status'],'TRACKED');self.assertEqual(b.calibration.provenance.quality,'propagated')
        fit={**self.fit,'homography':[[.1,0,1],[0,.1,0],[0,0,1]]}
        c=self.state(2,fit);self.assertEqual(self.info(c)['status'],'RELOCALIZED');self.assertEqual(c.last_reliable_timestamp,.08)
        self.assertAlmostEqual(c.motion_evidence['relocalization_correction_p90_m'],1)
    def test_expiry_loss_and_reacquisition(self):
        self.state(0,self.fit);self.state(1);lost=self.state(7)
        self.assertFalse(lost.calibration_valid);self.assertEqual(self.info(lost)['transition'],'loss');self.assertIn('PROPAGATION_EXPIRED',self.info(lost)['tracking_blockers'])
        self.assertIsNone(TemporalCamera.project((100,100),lost));self.assertEqual(self.info(self.state(8,self.fit))['status'],'RELOCALIZED')
    def test_cut_invalidates_and_starts_new_epoch(self):
        self.state(0,self.fit);cut=self.state(1,continuity='cut',epoch=1)
        self.assertFalse(cut.calibration_valid);self.assertEqual(self.info(cut)['transition'],'cut_reset')
        self.assertFalse(self.state(2,epoch=1).calibration_valid)
        self.assertTrue(self.state(3,self.fit,epoch=1).calibration_valid)
    def test_bad_transform_and_bad_evidence_are_rejected(self):
        for h in (np.zeros((3,3)),np.full((3,3),np.nan)):
            c=self.state(0,{**self.fit,'homography':h.tolist()});self.assertFalse(c.calibration_valid);self.assertEqual(self.info(c)['absolute_outcome'],'INVALID_VISIBLE_REGION')
        self.state(1,self.fit);self.evidence=self.evidence.model_copy(update={'lines':[]})
        c=self.state(2);self.assertFalse(c.calibration_valid);self.assertIn('INSUFFICIENT_GEOMETRIC_CONSTRAINTS',self.info(c)['tracking_blockers'])
    def test_measured_unsupported_zoom_cannot_propagate(self):
        self.state(0,self.fit)
        c=self.state(1,motion_evidence={'image_affine':[[1.1,0,0],[0,1.1,0]],'feature_matches':100,'affine_inliers':100,'affine_median_error_small_px':0,'projective_motion':{'accepted':False,'reason':'ZOOM_OR_SCALE_CHANGE_UNSUPPORTED','matches':100}})
        self.assertFalse(c.calibration_valid);self.assertIn('ZOOM_OR_SCALE_CHANGE_UNSUPPORTED',self.info(c)['tracking_blockers'])
    def test_world_loss_history_and_tracking_never_authorizes_identity(self):
        world=WorldModel('synthetic');t=TrackObservation(track_id='T',hits=2,bbox=(100,100,20,30),detection_confidence=.9,pitch=(11,13),appearance=(1.,0.))
        a=world.update([t],self.state(0,self.fit)).entities[0]
        c=self.state(1);b=world.update([t.model_copy(update={'pitch':(12,13)})],c).entities[0]
        self.assertEqual(b.last_reliable_pitch,a.last_reliable_pitch);self.assertIsNone(world.compare(t,b,.08,c).movement_feasible)
        lost=world.update([t],self.state(7)).entities[0];self.assertIsNone(lost.current_pitch);self.assertEqual(lost.last_reliable_pitch,a.last_reliable_pitch)
        self.assertEqual(lost.track_history,['T']);self.assertEqual(lost.last_reliable_camera_epoch,0)
    def test_projective_pan_is_measured_and_blank_is_rejected(self):
        rng=np.random.default_rng(42);image=np.zeros((180,320),np.uint8)
        for x,y in rng.integers([10,10],[310,170],size=(250,2)):cv2.circle(image,(int(x),int(y)),2,255,-1)
        pan=cv2.warpAffine(image,np.array([[1.,0,2],[0,1.,0]]),(320,180));motion=projective_motion(image,pan)
        self.assertTrue(motion['accepted'],motion);self.assertAlmostEqual(motion['homography'][0][2],2,delta=.15)
        zoom=cv2.warpAffine(image,cv2.getRotationMatrix2D((160,90),0,1.08),(320,180));self.assertFalse(projective_motion(image,zoom)['accepted'])
        self.assertFalse(projective_motion(np.zeros_like(image),np.zeros_like(image))['accepted'])
    def test_tracked_pan_changes_region_without_moving_fixed_world(self):
        a=self.state(0,self.fit)
        c=self.state(1,motion_evidence={'image_affine':[[1,0,1],[0,1,0]],'feature_matches':100,'affine_inliers':100,'affine_median_error_small_px':0})
        self.assertTrue(c.calibration_valid);self.assertNotEqual(a.visible_pitch_region,c.visible_pitch_region)
        np.testing.assert_allclose(TemporalCamera.project((100+1050/320,100),c),(10,10),atol=1e-8)
    def test_epoch_change_rejects_history_even_without_cut_flag(self):
        self.state(0,self.fit);c=self.state(1,epoch=1)
        self.assertFalse(c.calibration_valid);self.assertEqual(self.info(c)['transition'],'cut_reset')
    def test_zoom_can_reacquire_from_fresh_absolute_evidence(self):
        self.state(0,self.fit)
        c=self.state(1,motion_evidence={'image_affine':[[1.1,0,0],[0,1.1,0]],'feature_matches':100,'affine_inliers':100,'affine_median_error_small_px':0})
        self.assertFalse(c.calibration_valid)
        direct=self.state(2,{**self.fit,'homography':np.diag([.1/1.1,.1/1.1,1]).tolist()})
        self.assertTrue(direct.calibration_valid);self.assertEqual(self.info(direct)['status'],'RELOCALIZED')
        np.testing.assert_allclose(TemporalCamera.project((110,110),direct),(10,10),atol=1e-8)
    def test_taxonomy_uses_measured_gates(self):
        diag={'heldout_mean_px':{'lines':[11,12,13],'circles':[None]*3},'heldout_p90_px':{'lines':[15,16,17],'circles':[None]*3}}
        self.assertEqual(absolute_outcome(None,diag),'POSE_RESIDUAL_TOO_HIGH')
        self.assertEqual(absolute_outcome(None,{'reason':'arbitrary undocumented failure'}),'UNKNOWN_FAILURE')
        c=self.localizer.apply(CameraState(frame=0,timestamp=0),self.evidence,None,{'solver_attempted':True,'outcome':'POSE_OPTIMIZATION_FAILED','exception_class':'RuntimeError'})
        self.assertTrue(self.info(c)['absolute_attempted']);self.assertEqual(self.info(c)['absolute_outcome'],'POSE_OPTIMIZATION_FAILED')
