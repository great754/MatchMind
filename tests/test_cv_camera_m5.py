"""Known-geometry tests: they validate behavior, not real-video metric accuracy."""
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from src.vision.world.persistent_camera import PersistentCamera,corrected,safe_region
from src.vision.world.pitch_geometry import PitchGeometry
from src.vision.world.schema import CameraState,WorldState,TrackObservation,PlayerEntity
from src.vision.world.field_motion import validate_motion,assess_motion,FieldMotionTracker,field_image
from src.vision.world.texture_anchors import TextureAnchors
from src.vision.world.player_geometry import project_player
from src.vision.world.provider import canonical_observation

G=PitchGeometry();H=np.diag([.1,.1,1.]);W,HGT=1050,680

def groups(names=('Circle central','Middle line','Side line top'),h=H):
    result={}
    for name in names:
        if name in G.segments:
            a,b=map(np.array,G.segments[name]);points=a+(b-a)*np.linspace(0,1,80)[:,None]
        else:points=G.curves()[name]
        result[name]=cv2.perspectiveTransform(points[None].astype(float),np.linalg.inv(h))[0]/[W,HGT]
    return result

def motion(dx=0,scale=1):
    return {'accepted':True,'homography':[[scale,0,dx],[0,scale,0],[0,0,1]],'heldout_p90_small_px':.1,'scale_to_source':1.,'reason':'VALIDATED_PITCH_FEATURE_MOTION'}

class PersistentCameraTests(unittest.TestCase):
    def setUp(self):self.camera=PersistentCamera(W,HGT)
    def state(self,frame=0,fit=False,sem=None,flow=None,**updates):
        base=CameraState(frame=frame,timestamp=frame/25,continuity='continuous').model_copy(update=updates)
        return self.camera.update(base,groups() if sem is None else sem,{'homography':H.tolist()} if fit is True else fit or None,{},motion() if flow is None else flow)
    def info(self,c):return c.motion_evidence['localization']
    def test_direct_absolute(self):
        c=self.state(fit=True);self.assertTrue(c.calibration_valid);self.assertEqual(self.info(c)['status'],'DIRECT');self.assertTrue(c.projected_pitch_geometry)
    def test_pan_updates_transform_and_fixed_world_foot(self):
        self.state(fit=True);h=H@np.linalg.inv(np.array(motion(2)['homography']));c=self.state(1,sem=groups(h=h),flow=motion(2))
        self.assertEqual(self.info(c)['status'],'PARTIAL_CONSTRAINED');np.testing.assert_allclose(project_player((502,300),c)[0],(50,30),atol=.01)
    def test_one_frame_semantic_dropout(self):
        self.state(fit=True);c=self.state(1,sem={});self.assertEqual(self.info(c)['status'],'TEMPORAL_TRACKED');self.assertFalse(self.info(c)['current_semantic_support'])
    def test_several_dropouts_are_short_not_indefinite(self):
        self.state(fit=True)
        for i in range(1,7):self.assertTrue(self.state(i,sem={}).calibration_valid)
        self.assertFalse(self.state(7,sem={}).calibration_valid)
    def test_continuous_current_constraints_outlive_direct_lifetime(self):
        self.state(fit=True)
        for i in range(1,15):c=self.state(i)
        self.assertTrue(c.calibration_valid);self.assertAlmostEqual(self.info(c)['last_absolute_age_seconds'],.56);self.assertEqual(self.camera.memory.last_direct.frame,0)
    def test_center_circle_plus_sideline_uses_recent_prior(self):
        self.state(fit=True);c=self.state(1,sem=groups(('Circle central','Side line top')))
        self.assertEqual(self.info(c)['status'],'PARTIAL_CONSTRAINED');self.assertEqual(self.info(c)['semantic_constraint_upper_bound'],7)
    def test_center_circle_alone_cannot_seed(self):self.assertFalse(self.state(sem=groups(('Circle central',))).calibration_valid)
    def test_circle_and_sideline_cannot_seed(self):self.assertFalse(self.state(sem=groups(('Circle central','Side line top'))).calibration_valid)
    def test_circle_halfway_prior(self):
        self.state(fit=True);self.assertTrue(self.state(1,sem=groups(('Circle central','Middle line'))).calibration_valid)
    def test_penalty_end_constraints(self):
        self.state(fit=True);c=self.state(1,sem=groups(('Big rect. right main','Side line right','Big rect. right top')))
        self.assertEqual(self.info(c)['status'],'PARTIAL_CONSTRAINED')
    def test_touchline_is_weak_not_fresh_pose(self):
        sem=groups(('Side line top',));self.assertFalse(self.state(sem=sem).calibration_valid);self.state(1,True)
        self.assertEqual(self.info(self.state(2,sem=sem))['status'],'TEMPORAL_TRACKED')
    def test_ambiguous_local_candidates_stay_unknown(self):
        self.state(fit=True)
        with patch.object(self.camera,'refine',return_value=(None,{'reason':'AMBIGUOUS_GEOMETRY'})):
            c=self.state(1,sem=groups(),flow={'accepted':False,'reason':'AMBIGUOUS_GEOMETRY'})
        self.assertFalse(c.calibration_valid);self.assertEqual(self.info(c)['outcome'],'AMBIGUOUS_GEOMETRY')
    def test_distinct_passing_candidates_are_not_arbitrarily_selected(self):
        from types import SimpleNamespace
        h=np.diag([.5,.5,1.]);sem=groups(h=h)
        candidates=[SimpleNamespace(x=np.array(p,float)) for p in ((0,0,0,0),(8,-8,0,0),(-8,8,0,0))]
        with patch('src.vision.world.persistent_camera.least_squares',side_effect=candidates):
            candidate,why=self.camera.refine(h,sem,0,0)
        self.assertIsNone(candidate);self.assertEqual(why['reason'],'AMBIGUOUS_GEOMETRY');self.assertGreater(why['hypothesis_spread_p90_m'],4)
    def test_bad_semantics_do_not_refine_to_arbitrary_pose(self):
        self.state(fit=True);bad={k:v+[.3,.3] for k,v in groups().items()};self.assertFalse(self.state(1,sem=bad).calibration_valid)
    def test_cut_resets_current_not_historical_direct(self):
        self.state(fit=True);c=self.state(1,epoch=1,continuity='cut');self.assertFalse(c.calibration_valid);self.assertIsNone(c.memory.last_accepted);self.assertIsNone(self.info(c)['last_absolute_age_seconds']);self.assertIsNotNone(c.memory.last_direct)
    def test_epoch_change_resets_without_flag(self):
        self.state(fit=True);self.assertFalse(self.state(1,epoch=2).calibration_valid)
    def test_relocalize_after_unknown(self):
        self.state(fit=True);self.state(8);self.assertEqual(self.info(self.state(9,True))['status'],'RELOCALIZED')
    def test_consecutive_direct_is_not_relocalization(self):
        self.state(fit=True);self.assertEqual(self.info(self.state(1,True))['status'],'DIRECT')
    def test_unsupported_zoom_rejected_can_reacquire(self):
        self.state(fit=True);self.assertFalse(self.state(1,flow={'accepted':False,'reason':'ZOOM_UNSUPPORTED'}).calibration_valid);self.assertTrue(self.state(2,True).calibration_valid)
    def test_reflection_rejected_even_with_accepted_flag(self):
        self.state(fit=True);m=motion();m['homography']=[[-1,0,W],[0,1,0],[0,0,1]];self.assertFalse(self.state(1,flow=m).calibration_valid)
    def test_invalid_direct_transforms(self):
        for bad in (np.zeros((3,3)),np.full((3,3),np.nan),np.ones((2,3))):
            with self.subTest(shape=bad.shape):self.assertFalse(self.state(fit={'homography':bad.tolist()}).calibration_valid)
    def test_error_budget_expires_without_geometry_weakening(self):
        self.state(fit=True);m=motion();m['heldout_p90_small_px']=21;self.assertFalse(self.state(1,flow=m).calibration_valid)
    def test_history_bounded_and_current_distinct(self):
        self.state(fit=True)
        for i in range(1,16):self.state(i)
        self.assertLessEqual(len(self.camera.memory.accepted_history),12);self.assertEqual(self.camera.memory.last_direct.frame,0);self.assertEqual(self.camera.memory.last_accepted.frame,15)
    def test_unknown_has_no_stale_current_geometry(self):
        self.state(fit=True);c=self.state(8);self.assertIsNone(project_player((500,300),c)[0]);self.assertIsNone(c.calibration);self.assertEqual(c.projected_pitch_geometry,[])
    def test_filter_reduces_correction_against_noisy_semantics(self):
        self.state(fit=True);noisy={k:v+[.002,0] for k,v in groups().items()};c=self.state(1,sem=noisy)
        self.assertTrue(c.calibration_valid);self.assertLess(self.info(c)['landmark_validation']['correction_p90_source_px'],2.1)
    def test_canonical_partial_is_propagated_not_direct(self):
        self.state(fit=True);c=self.state(1);person=PlayerEntity(entity_id='p',team='team_a',visibility='visible',last_seen=.04,last_seen_frame=1,current_pitch=(50,30),position_source='partial_constrained');s=WorldState(recording_id='synthetic',frame=1,timestamp=.04,camera=c,entities=[person],unassigned=[])
        obs=canonical_observation(s,25,'synthetic');self.assertEqual(obs.calibration.provenance.quality,'propagated')
        from src.vision.world.provider import analytics_recording
        obs=obs.model_copy(update={'frame':0,'timestamp':0,'calibration':None});data=analytics_recording([obs],{'team_a':'left'},1);self.assertFalse(np.isfinite(data.positions).any())
    def test_invalid_and_outside_foot_points(self):
        c=self.state(fit=True)
        for point in ((float('nan'),30),(-100,300),(W+100,300)):
            with self.subTest(point=point):self.assertIsNone(project_player(point,c)[0])
    def test_no_measured_motion_cannot_use_current_semantics_as_prior(self):
        self.state(fit=True);c=self.state(1,flow={'accepted':False,'reason':'NO_CURRENT_MEASUREMENT'});self.assertFalse(c.calibration_valid)
    def test_current_and_historical_entity_positions_separate(self):
        from src.vision.world.provider import canonical_observation
        c=self.state();e=PlayerEntity(entity_id='history',visibility='off_camera',last_seen=0,last_seen_frame=0,current_pitch=None,last_reliable_pitch=(50,30))
        s=WorldState(recording_id='synthetic',frame=0,timestamp=0,camera=c,entities=[e],unassigned=[]);p=canonical_observation(s,25,'synthetic').players[0];self.assertIsNone(p.pitch);self.assertEqual(e.last_reliable_pitch,(50,30))
    def test_polygon_iou_stays_geometric(self):
        from scripts.evaluate_camera_placement import polygon_iou
        square=[[0,0],[10,0],[10,10],[0,10]];self.assertAlmostEqual(polygon_iou(square,square),1);self.assertAlmostEqual(polygon_iou(square,[[5,0],[15,0],[15,10],[5,10]]),1/3)
    def test_configured_geometry_and_overlay(self):
        other=PitchGeometry(length=110,width=70);self.assertEqual(other.center,(55,35));self.assertEqual(other.points['top_right'],(110,0));self.assertTrue(G.overlay(H,W,HGT));self.assertIsNone(safe_region(np.zeros((3,3)),W,HGT))

class FieldEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.field=(np.zeros((640,640),np.uint8),np.full((640,640),255,np.uint8),1.)
        self.points=np.array([(x,y) for x in np.linspace(40,590,12) for y in np.linspace(40,590,9)],np.float32)
    def test_known_pan_small_shake_and_zoom(self):
        for scale,angle in ((1,0),(1,.1),(1.005,0)):
            h=cv2.getRotationMatrix2D((320,320),angle,scale);h[:,2]+=[2,0];b=cv2.transform(self.points[None],h)[0];r=validate_motion(self.points,b,self.field,self.field);self.assertTrue(r['accepted'],r)
    def test_large_zoom_rejected(self):
        r=validate_motion(self.points,self.points*1.1,self.field,self.field);self.assertFalse(r['accepted']);self.assertEqual(r['reason'],'ZOOM_UNSUPPORTED')
    def test_collinear_features_constrain_similarity_not_projective(self):
        a=np.column_stack([np.linspace(30,600,60),np.ones(60)*320]).astype(np.float32);r=validate_motion(a,a+[2,0],self.field,self.field);self.assertTrue(r['accepted'],r);self.assertEqual(r['model'],'similarity');self.assertFalse(r['models'][-1]['accepted'])
    def test_few_matches_and_blank_are_unknown(self):
        self.assertFalse(validate_motion(self.points[:20],self.points[:20],self.field,self.field)['accepted']);self.assertFalse(FieldMotionTracker().update(self.field)['accepted'])
    def test_flow_failure_clears_stale_current_measurements(self):
        t=FieldMotionTracker();t.previous=self.field;t.points=self.points[:40,None];t.ids=np.arange(40);t.ages=np.ones(40,int)*3;t.static=np.ones(40,bool);t.measurements={'ids':np.arange(40),'points':self.points[:40]}
        with patch('cv2.calcOpticalFlowPyrLK',return_value=(None,None,None)):t.update(self.field)
        self.assertIsNone(t.measurements)
    def test_anchors_require_reobserved_current_features_and_epoch(self):
        c=PersistentCamera(W,HGT);a=c.update(CameraState(frame=0,timestamp=0),groups(),{'homography':H.tolist()},{},motion());keys=TextureAnchors();keys.add(0,0,a.memory.last_accepted,self.field,None);self.assertEqual(keys.keys,[])
        measured={'ids':np.arange(len(self.points)),'points':self.points};keys.add(0,0,a.memory.last_accepted,self.field,measured,True)
        self.assertIsNone(keys.propose(1,0,self.field,None,motion(),W,HGT)[0]);self.assertIsNone(keys.propose(1,1,self.field,measured,motion(),W,HGT)[0]);self.assertIsNotNone(keys.propose(1,0,self.field,measured,motion(),W,HGT)[0])
